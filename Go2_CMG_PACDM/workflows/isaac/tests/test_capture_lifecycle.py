"""API-shaped fakes only. These tests do not execute an NVIDIA renderer."""
from types import SimpleNamespace
import numpy as np
import pytest

from isaac_validation.capture import PhysicsFrameCapture
from isaac_validation.lifecycle import cleanup_actions
from isaac_validation.runtime import shutdown_configuration, close_simulation_app
from test_runtime import install_module


def capture_fixture(monkeypatch, mode='render'):
    calls = []
    robot = SimpleNamespace(q=np.zeros(18), v=np.zeros(18))
    robot.state = lambda: (robot.q.copy(), robot.v.copy())
    world = SimpleNamespace(current_time=0.0, block=False)
    world.get_block_on_render = lambda: world.block
    def set_block(value):
        calls.append(('block', value))
        world.block = value
    world.set_block_on_render = set_block
    settings = {'/omni/replicator/captureOnPlay': True}
    install_module(monkeypatch, 'carb.settings', get_settings=lambda: SimpleNamespace(get=settings.get))
    world.render_count = 0
    def render():
        calls.append('render')
        world.render_count += 1
    world.render = render
    rgb = SimpleNamespace(data=np.zeros((2, 3, 4), dtype=np.uint8))
    rgb.attach = lambda products: calls.append('attach')
    rgb.detach = lambda: calls.append('detach')
    rgb.ready_after = 0
    rgb.get_data = lambda: (rgb.data if world.render_count >= rgb.ready_after
                           and (settings['/omni/replicator/captureOnPlay'] or mode == 'replicator')
                           else np.empty(0, dtype=np.uint8))
    product = SimpleNamespace(destroy=lambda: calls.append('destroy'))
    def set_capture(value):
        calls.append(('capture_on_play', value))
        settings['/omni/replicator/captureOnPlay'] = value
    def step(**kwargs):
        calls.append(('rep.step', kwargs))
        world.render_count += 1
    world.capture_settings = settings
    rep = SimpleNamespace(orchestrator=SimpleNamespace(
        set_capture_on_play=set_capture, step=step),
        create=SimpleNamespace(render_product=lambda *a: product),
        AnnotatorRegistry=SimpleNamespace(get_annotator=lambda *a, **kw: rgb))
    install_module(monkeypatch, 'omni.replicator.core', **vars(rep))
    capture = PhysicsFrameCapture(world, robot, '/Camera', mode=mode, resolution=(3, 2))
    return capture, world, robot, rgb, product, calls


def test_default_render_drains_lag_without_replicator_scheduler(monkeypatch):
    capture, world, robot, rgb, product, calls = capture_fixture(monkeypatch)
    capture.start()
    frame = capture.frame(0.0)
    assert calls.count('render') == 6  # four warmup + two at the frame state
    assert not any(isinstance(call, tuple) and call[0] == 'rep.step' for call in calls)
    assert frame.shape == (2, 3, 3)
    rgb.data[:] = 99
    assert np.max(frame) == 0  # native backing buffer is not reused
    assert capture.audit()['physics_unchanged'] is True
    capture.close()
    assert calls[-4:] == ['detach', 'destroy', ('block', False), ('capture_on_play', True)]
    capture.close()  # idempotent
    assert calls.count('detach') == calls.count('destroy') == 1


@pytest.mark.parametrize('changed', ['q', 'v', 'time', 'nonfinite'])
def test_render_state_guard_rejects_changes_without_restoring_physics(monkeypatch, changed):
    capture, world, robot, *_ = capture_fixture(monkeypatch)
    capture.start()
    def bad_render():
        if changed == 'q': robot.q[0] += 0.001
        elif changed == 'v': robot.v[0] += 0.001
        elif changed == 'time': world.current_time += 0.001
        else: robot.q[0] = np.nan
    world.render = bad_render
    with pytest.raises(RuntimeError, match='Rendering advanced/changed physics'):
        capture.frame(0.0)
    assert capture.audit()['frame_count'] == 0
    if changed == 'q': assert robot.q[0] == 0.001
    if changed == 'v': assert robot.v[0] == 0.001
    if changed == 'time': assert world.current_time == 0.001
    capture.close()


@pytest.mark.parametrize('shape,dtype', [((2, 3),np.uint8), ((2,3,2),np.uint8), ((2,3,3),float), ((0,),np.uint8)])
def test_bad_native_rgb_is_not_a_frame(monkeypatch, shape, dtype):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    rgb.data = np.zeros(shape, dtype=dtype)
    with pytest.raises(RuntimeError, match='Invalid live RGB|Live RGB not ready'):
        capture.frame(0.0)
    assert not capture.audit()['physics_unchanged']
    capture.close()


def test_replicator_is_explicit_zero_time_single_subframe_alternative(monkeypatch):
    capture, *_, calls = capture_fixture(monkeypatch, mode='replicator')
    capture.start()
    capture.frame(0.0)
    assert 'render' not in calls
    steps = [item[1] for item in calls if isinstance(item, tuple) and item[0]=='rep.step']
    assert len(steps) == 5
    assert all(step == dict(rt_subframes=1, delta_time=0.0, pause_timeline=False, wait_for_render=True) for step in steps)
    capture.close()


def test_capture_partial_start_still_detaches_and_destroys(monkeypatch):
    capture, world, robot, rgb, product, calls = capture_fixture(monkeypatch)
    def fail_attach(products): raise RuntimeError('test native attach failure')
    rgb.attach = fail_attach
    with pytest.raises(RuntimeError, match='attach failure'):
        capture.start()
    capture.close()
    assert calls[-3:] == ['detach', 'destroy', ('capture_on_play', True)]


def test_cleanup_continues_after_detach_error(monkeypatch):
    capture, world, robot, rgb, product, calls = capture_fixture(monkeypatch)
    capture.start()
    def fail_detach(): raise RuntimeError('test detach failure')
    rgb.detach = fail_detach
    with pytest.raises(RuntimeError, match='detach failure'):
        capture.close()
    assert calls[-3:] == ['destroy', ('block', False), ('capture_on_play', True)]


def test_every_resource_cleanup_attempted_and_failure_recorded():
    calls=[]
    def fail(): raise RuntimeError('cannot release test resource')
    result=cleanup_actions([('a',lambda:calls.append('a')),('b',fail),('c',lambda:calls.append('c'))])
    assert calls == ['a','c']
    assert not result['passed']
    assert [item['passed'] for item in result['steps']] == [True,False,True]


@pytest.mark.parametrize('mode,fast', [('auto',True),('fast',True),('graceful',False)])
def test_modern_shutdown_policy_preserves_status_api(mode, fast):
    class App:
        def close(self, exit_code=0): pass
    config=shutdown_configuration(App,mode)
    assert config['fast_shutdown'] is fast
    assert config['skip_cleanup'] is False


def test_legacy_auto_preserves_failure_without_unsafe_fast_close():
    class App:
        def close(self): pass
    assert shutdown_configuration(App)['fast_shutdown'] is False
    with pytest.raises(RuntimeError, match='requires'):
        shutdown_configuration(App,'fast')


def test_python_shutdown_exception_is_not_swallowed():
    class App:
        def close(self,exit_code=0): raise RuntimeError('test shutdown error')
    with pytest.raises(RuntimeError, match='shutdown error'):
        close_simulation_app(App(),0)


def test_render_mode_enables_playback_capture_before_warmup(monkeypatch):
    capture, world, *_, calls = capture_fixture(monkeypatch)
    world.capture_settings['/omni/replicator/captureOnPlay'] = False
    capture.start()
    assert calls[0] == ('capture_on_play', True)
    assert capture.audit()['camera_ready'] is True
    assert capture.audit()['warmup']['rgb_ready'] is True
    capture.close()
    assert world.capture_settings['/omni/replicator/captureOnPlay'] is False


def test_replicator_disables_automatic_capture_and_restores_it(monkeypatch):
    capture, world, *_, calls = capture_fixture(monkeypatch, 'replicator')
    capture.start()
    assert calls[0] == ('capture_on_play', False)
    capture.close()
    assert world.capture_settings['/omni/replicator/captureOnPlay'] is True


@pytest.mark.parametrize('ready_after', [4, 7, 64])
def test_delayed_startup_waits_for_actual_data_without_physics(monkeypatch, ready_after):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    rgb.ready_after = ready_after
    capture.start()
    assert world.render_count == ready_after
    warmup = capture.audit()['warmup']
    assert warmup['passed'] and warmup['rgb_ready']
    assert warmup['render_passes'] == ready_after
    assert warmup['last_rgb_shape'] == [2, 3, 4]
    assert world.current_time == 0 and np.all(robot.q == 0) and np.all(robot.v == 0)
    assert capture.audit()['frame_count'] == 0  # warmup is not a video frame
    capture.close()


def test_empty_startup_buffer_is_a_bounded_failure_not_successful_warmup(monkeypatch):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    rgb.ready_after = 65
    with pytest.raises(RuntimeError, match='not ready during warmup after 64'):
        capture.start()
    audit = capture.audit()
    assert world.render_count == 64 and world.current_time == 0
    assert not audit['camera_ready'] and not audit['physics_unchanged']
    assert audit['failed'] and not audit['warmup']['passed']
    assert audit['warmup']['last_rgb_shape'] == [0]
    assert not audit['frames']
    capture.close()


def test_transient_empty_frame_recovers_without_reusing_previous_pixels(monkeypatch):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    capture.frame(0.0)
    rgb.data[:] = 123
    rgb.ready_after = world.render_count + 5
    image = capture.frame(0.04)
    assert np.all(image == 123)
    assert capture.audit()['frames'][-1]['render_passes'] == 5
    assert len(capture.audit()['frames'][-1]['samples']) == 4
    capture.close()


def test_capture_failure_invalidates_earlier_successful_frames(monkeypatch):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    capture.frame(0.0)
    rgb.data = np.empty(0, dtype=np.uint8)
    with pytest.raises(RuntimeError, match='not ready during frame after 16'):
        capture.frame(0.04)
    assert capture.audit()['frame_count'] == 1
    assert not capture.audit()['physics_unchanged']
    with pytest.raises(RuntimeError, match='has failed'):
        capture.frame(0.08)
    capture.close()


def test_drift_is_compared_to_start_of_whole_readiness_loop(monkeypatch):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    rgb.data = np.empty(0, dtype=np.uint8)
    def drift():
        robot.q[0] += 0.6e-7  # two renders together exceed the unchanged-state tolerance
    world.render = drift
    with pytest.raises(RuntimeError, match='advanced/changed physics'):
        capture.frame(0.0)
    assert robot.q[0] > 1e-7  # not silently restored
    assert capture.audit()['last_attempt']['render_passes'] == 2
    capture.close()


def test_data_read_cannot_change_physics(monkeypatch):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    def changed_read():
        world.current_time += 0.001
        return rgb.data
    rgb.get_data = changed_read
    with pytest.raises(RuntimeError, match='advanced/changed physics'):
        capture.frame(0.0)
    assert capture.audit()['frame_count'] == 0
    capture.close()


@pytest.mark.parametrize('time_s', [-1, float('nan'), float('inf'), 0])
def test_invalid_or_duplicate_frame_time_is_rejected(monkeypatch, time_s):
    capture, *_, calls = capture_fixture(monkeypatch)
    capture.start()
    capture.frame(0.0)
    before = len(calls)
    with pytest.raises(ValueError, match='timestamps'):
        capture.frame(time_s)
    assert len(calls) == before
    capture.close()


def test_valid_dict_rgb_is_copied(monkeypatch):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    image = np.ones((2, 3, 3), dtype=np.uint8)
    rgb.get_data = lambda: {'data': image}
    frame = capture.frame(0.0)
    image[:] = 9
    assert np.all(frame == 1)
    capture.close()


def test_missing_rgb_mapping_data_is_not_retried(monkeypatch):
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    rgb.get_data = lambda: {'info': {}}
    with pytest.raises(RuntimeError, match="missing 'data'"):
        capture.frame(0.0)
    assert capture.audit()['last_attempt']['render_passes'] == 2
    capture.close()


@pytest.mark.parametrize('invalid', ['q', 'time'])
def test_nonfinite_pre_render_state_cannot_leave_a_valid_audit(monkeypatch, invalid):
    import json
    capture, world, robot, rgb, *_ = capture_fixture(monkeypatch)
    capture.start()
    capture.frame(0.0)
    if invalid == 'q': robot.q[0] = np.nan
    else: world.current_time = float('inf')
    with pytest.raises(RuntimeError, match='Nonfinite physics state'):
        capture.frame(0.04)
    audit = capture.audit()
    assert audit['failed'] and not audit['physics_unchanged']
    json.dumps(audit, allow_nan=False)
    capture.close()
