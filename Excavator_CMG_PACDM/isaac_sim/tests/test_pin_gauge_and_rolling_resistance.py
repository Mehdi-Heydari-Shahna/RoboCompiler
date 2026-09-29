"""Regression tests for the q22 pin-spin gauge, grain rolling resistance and CPU replay.

1. q22 pin-spin gauge (the 1.5 rad numerical guard must not stop a PhysX run)
2. soil-grain rolling resistance (PhysX has no rolling friction)
3. CPU renderer for the post-run replay (independent of the Isaac RTX workers)

Tests marked "model" load the real MuJoCo/Pinocchio controller model and are
skipped when those packages are unavailable. None of these tests run PhysX.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = json.loads((ROOT/'generated/soil_final/manifest.json').read_text(encoding='utf-8'))
SCENE_XML = (ROOT/'generated/soil_final'/MANIFEST['source_scene']).resolve()
HAVE_MUJOCO = importlib.util.find_spec('mujoco') is not None
HAVE_MODEL = HAVE_MUJOCO and importlib.util.find_spec('pinocchio') is not None
needs_mujoco = pytest.mark.skipif(not HAVE_MUJOCO, reason='mujoco unavailable')
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason='mujoco/pinocchio controller runtime unavailable')


# ---------------------------------------------------------------- q22 gauge
def test_controller_state_view_only_gauges_listed_address():
    from bridge import ControllerStateView
    data = SimpleNamespace(qpos=np.arange(5.), qvel=np.arange(4.), qacc=-np.arange(4.), time=1.25)
    view = ControllerStateView(data, ((2, 9.),))
    assert view.qpos.tolist() == [0., 1., 9., 3., 4.]
    assert data.qpos[2] == 2.  # plant/mapper data never modified
    view.qvel[0] = 99.
    assert data.qvel[0] == 0.  # copies, not aliases
    assert view.time == 1.25 and np.array_equal(view.qacc, data.qacc)


def _observation(model, qpos):
    import mujoco
    from bridge import kinematic_observation
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.
    return kinematic_observation(model, data)


@pytest.fixture(scope='module')
def bridge_factory():
    if not HAVE_MODEL:
        pytest.skip('mujoco/pinocchio controller runtime unavailable')
    from bridge import ControllerBridge
    source = (ROOT/'generated/soil_final'/MANIFEST['source_root']).resolve()

    def make():
        return ControllerBridge(source, SCENE_XML, MANIFEST['initial_qpos'], dt=.001)
    return make


@needs_model
def test_spin_gauge_is_verified_at_startup(bridge_factory):
    bridge = bridge_factory()
    gauge = bridge.metadata()['passive_spin_gauge']
    record, = gauge['coordinates']
    assert record['joint'] == 'q22' and record['body'] == 'body_59' and record['independent_index'] == 6
    assert all(value < 1e-10 for value in record['checks'].values())
    assert gauge['plant_state_written'] is False and gauge['velocity_modified'] is False
    # A pin with an off-axis COM is not an ignorable spin: refuse to gauge it.
    body = bridge.m.body('body_59').id
    saved = bridge.m.body_ipos[body].copy()
    try:
        bridge.m.body_ipos[body] = saved+np.array([.01, 0., 0.])
        with pytest.raises(ValueError, match='not a verified ignorable pin spin'):
            bridge._configure_passive_spin_gauge(['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'q21', 'q23'])
    finally:
        bridge.m.body_ipos[body] = saved


@needs_model
def test_tick_survives_q22_beyond_numerical_guard(bridge_factory):
    """A q22 offset above the 1.5 rad numerical guard must not stop the controller."""
    import mujoco
    reference = bridge_factory()
    drifted = bridge_factory()
    model = reference.m
    q0 = np.asarray(MANIFEST['initial_qpos'], float)
    address = model.jnt_qposadr[model.joint('q22').id]
    q_drift = q0.copy()
    q_drift[address] += 2.0  # the pin spun 2 rad further than the source guard allows
    a = _observation(model, q0)
    b = _observation(model, q_drift)
    moved = np.flatnonzero(np.any(np.abs(a['quaternions_wxyz']-b['quaternions_wxyz']) > 1e-12, axis=1))
    assert [reference.mapper.body_names[i] for i in moved] == ['body_59']  # only the pin differs
    ra = reference.tick(0., a['positions'], a['quaternions_wxyz'], a['linear_velocities'], a['angular_velocities'])
    rb = drifted.tick(0., b['positions'], b['quaternions_wxyz'], b['linear_velocities'], b['angular_velocities'])
    assert rb['passive_pin_gauge']['q22']['measured_offset_rad'] == pytest.approx(2.0, abs=1e-6)
    assert np.allclose(ra['efforts'], rb['efforts'], rtol=1e-9, atol=1e-6)
    # The unmodified controller input would stop exactly as in the Isaac run.
    raw = SimpleNamespace(qpos=drifted.d.qpos.copy(), qvel=drifted.d.qvel.copy(),
                          qacc=drifted.d.qacc.copy(), time=0.)
    with pytest.raises(ValueError, match='Independent coordinate exceeded its numerical guard'):
        drifted.arm.evaluate(*drifted.mission.arm_reference(0.), raw)
    summary = drifted.summary()
    assert summary['passive_spin_gauge']['coordinates'][0]['max_abs_measured_offset_rad'] == pytest.approx(2.0, abs=1e-6)


@needs_model
def test_efforts_invariant_to_q22_inside_guard(bridge_factory):
    """Inside the guard the raw and gauge-fixed controller inputs agree to rounding."""
    from tracked_arm import TrackedArmController
    from bridge import ControllerStateView
    bridge = bridge_factory()
    model = bridge.m
    address = model.jnt_qposadr[model.joint('q22').id]
    arm_raw = TrackedArmController(*bridge.context, bridge.mission.initial_tree, bridge.m, bridge.d)
    arm_fixed = TrackedArmController(*bridge.context, bridge.mission.initial_tree, bridge.m, bridge.d)
    rng = np.random.default_rng(3)
    q0 = np.asarray(MANIFEST['initial_qpos'], float)
    for step, offset in enumerate((0.3, 0.9, 1.4)):
        q = q0.copy(); q[address] += offset
        obs = _observation(model, q)
        bridge.mapper.update(.001*(step+1), obs['positions'], obs['quaternions_wxyz'],
                             obs['linear_velocities'], obs['angular_velocities'])
        bridge.d.qvel[:] = rng.normal(scale=.05, size=model.nv)
        bridge.d.qacc[:] = rng.normal(scale=.2, size=model.nv)
        desired = bridge.mission.arm_reference(0.)
        raw = arm_raw.evaluate(*desired, ControllerStateView(bridge.d))
        fixed = arm_fixed.evaluate(*desired, ControllerStateView(bridge.d, bridge.spin_gauge))
        assert np.max(np.abs(raw-fixed)) <= 1e-9*max(1., np.max(np.abs(raw)))


# ------------------------------------------------------ grain rolling resistance
def _grain(manifest):
    names = [b['name'] for b in manifest['bodies']]
    return names.index('grain_0000'), manifest['bodies'][names.index('grain_0000')]


def test_rolling_friction_derived_from_manifest():
    from rolling_resistance import GrainRollingResistance, manifest_rolling_friction
    assert manifest_rolling_friction(MANIFEST) == pytest.approx(.008)
    model = GrainRollingResistance(MANIFEST, .001)
    assert model.enabled and len(model.index) == 105 and model.mu == pytest.approx(.008)
    assert not GrainRollingResistance(MANIFEST, .001, 0.).enabled
    assert GrainRollingResistance(MANIFEST, .001, .004).mu == .004
    with pytest.raises(ValueError):
        GrainRollingResistance(MANIFEST, .001, -1.)


def test_rolling_torque_equals_mujoco_coulomb_for_grain_on_ground():
    from rolling_resistance import GrainRollingResistance
    model = GrainRollingResistance(MANIFEST, .001)
    k, grain = _grain(MANIFEST)
    n = len(MANIFEST['bodies'])
    weight = grain['mass']*9.81
    forces = np.zeros((n, 3)); forces[k] = [0., 0., weight]
    model.update_contacts(forces)
    omega = np.zeros((n, 3)); omega[k] = [0., 10., 0.]   # rolling at 1 m/s (r = 0.1 m)
    omega[0] = [1., 2., 3.]                                # robot body: must stay untouched
    tau = model.torques(omega)
    assert np.allclose(tau[k], [0., -.008*weight, 0.])
    assert np.count_nonzero(np.any(tau != 0., axis=1)) == 1
    # Solid sphere rolling without slip: deceleration mu_r*g/(1.4*r), as in MuJoCo.
    radius, inertia = .1, max(grain['inertia_diagonal'])
    decel = .008*weight*radius/(inertia+grain['mass']*radius**2)
    assert decel == pytest.approx(.008*9.81/(1.4*radius), rel=1e-6)
    # No contact, no torque (free flight is unaffected).
    model.update_contacts(np.zeros((n, 3)))
    assert not np.any(model.torques(omega))


def test_torsional_component_uses_mujoco_torsional_coefficient():
    from rolling_resistance import GrainRollingResistance
    model = GrainRollingResistance(MANIFEST, .001)
    assert model.mu_t == pytest.approx(.003)
    k, grain = _grain(MANIFEST)
    n = len(MANIFEST['bodies'])
    weight = grain['mass']*9.81
    forces = np.zeros((n, 3)); forces[k] = [0., 0., weight]
    model.update_contacts(forces)
    omega = np.zeros((n, 3)); omega[k] = [0., 4., 3.]   # rolling about y plus spin about the normal
    tau = model.torques(omega)[k]
    assert tau == pytest.approx([0., -.008*weight, -.003*weight])


def test_rolling_torque_regularized_and_stability_capped():
    from rolling_resistance import GrainRollingResistance
    model = GrainRollingResistance(MANIFEST, .001)
    k, grain = _grain(MANIFEST)
    n = len(MANIFEST['bodies'])
    forces = np.zeros((n, 3)); forces[k] = [0., 0., 100.]
    model.update_contacts(forces)
    slow = np.zeros((n, 3)); slow[k] = [.1, 0., 0.]
    assert model.torques(slow)[k] == pytest.approx([-.008*100.*.1/.5, 0., 0.])
    forces[k] = [0., 0., 1e7]                              # absurd squeeze: capped gain
    model.update_contacts(forces)
    fast = np.zeros((n, 3)); fast[k] = [0., 0., 2.]
    tau = model.torques(fast)[k]
    inertia = max(grain['inertia_diagonal'])
    assert tau[2] == pytest.approx(-.5*inertia/.001*2.)
    # One explicit step can at most halve the spin; it never reverses it.
    assert 2.+tau[2]*.001/inertia == pytest.approx(1.)
    assert model.report()['steps_with_stability_cap'] == 1


def test_rolling_work_is_dissipative():
    from rolling_resistance import GrainRollingResistance
    model = GrainRollingResistance(MANIFEST, .001)
    k, grain = _grain(MANIFEST)
    n = len(MANIFEST['bodies'])
    forces = np.zeros((n, 3)); forces[k] = [0., 0., grain['mass']*9.81]
    model.update_contacts(forces)
    w0 = np.zeros((n, 3)); w0[k] = [5., 0., 0.]
    w1 = w0*.99
    tau = model.torques(w0)
    model.account(tau, w0, w1)
    assert model.report()['applied_torque_work_J'] < 0.


def test_runner_applies_grain_torques_after_actuator_power_and_records_them():
    text = (ROOT/'run_isaac.py').read_text(encoding='utf-8')
    loop = text[text.index('for step in range(steps):'):]
    power = loop.index('actual_actuator_power = float(')
    add = loop.index('torque = torque + grain_torque')
    apply = loop.index('rb.apply_forces_and_torques_at_position(')
    record = loop.index('controls.append((t, efforts.copy(), force.copy(), torque.copy()))')
    assert power < add < apply < record
    assert loop.index('rolling.update_contacts(contact_now)') < record
    launch = (ROOT/'launch.py').read_text(encoding='utf-8')
    assert "'--grain-rolling-friction'" in launch and "renderer=args.video_renderer" in launch
    assert 'angularDamping' not in text and 'grain_angular_damping' not in text


# --------------------------------------------------------------- CPU video
def test_cpu_renderer_is_default():
    from replay_video import render_video
    assert inspect.signature(render_video).parameters['renderer'].default == 'cpu'


def test_frames_from_another_renderer_are_not_reused(tmp_path):
    from test_reliability import create_replay_result, fake_render_worker
    from replay_video import prepare_plan, verified_frame, RENDERERS
    out, scene = create_replay_result(tmp_path)
    plan = prepare_plan(out, scene)
    fake_render_worker(['--result', str(out), '--start', '0', '--stop', '2'])
    assert verified_frame(out, plan, 0, RENDERERS['isaac']) is not None
    assert verified_frame(out, plan, 0, RENDERERS['cpu']) is None


def test_cpu_render_refuses_to_replace_finished_isaac_video(tmp_path):
    from test_reliability import create_replay_result, fake_render_worker
    from replay_video import prepare_plan, render_video
    out, scene = create_replay_result(tmp_path)
    plan = prepare_plan(out, scene)
    fake_render_worker(['--result', str(out), '--start', '0', '--stop', str(len(plan['frames']))])
    before = sorted((p.name, p.read_bytes()) for p in (out/'frames').glob('*.png'))
    with pytest.raises(ValueError, match='complete verified frame set'):
        render_video(out, None, scene, renderer='cpu')
    (out/'isaac_native.mp4').write_bytes(b'\x00\x00\x00\x18ftypmp42'+b'0'*2048)
    (out/'video_encoding.json').write_text(json.dumps({'mp4_created': True}))
    with pytest.raises(ValueError, match='verified MP4'):
        render_video(out, None, scene, renderer='cpu')
    assert sorted((p.name, p.read_bytes()) for p in (out/'frames').glob('*.png')) == before
    assert json.loads((out/'video_encoding.json').read_text(encoding='utf-8')) == {'mp4_created': True}


def test_incomplete_isaac_attempt_may_be_rerendered_on_cpu(tmp_path):
    from test_reliability import create_replay_result, fake_render_worker
    from replay_video import _refuse_to_replace_finished_video, prepare_plan, RENDERERS
    out, scene = create_replay_result(tmp_path)
    plan = prepare_plan(out, scene)
    fake_render_worker(['--result', str(out), '--start', '0', '--stop', '2'])   # like the 3/167 frames
    (out/'video_encoding.json').write_text(json.dumps({'mp4_created': False}))
    _refuse_to_replace_finished_video(out, plan, RENDERERS['cpu'])  # does not raise


def test_replay_cli_keeps_isaac_meaning_of_old_command(tmp_path):
    import subprocess
    old = subprocess.run([sys.executable, str(ROOT/'replay_video.py'), '--result', str(tmp_path/'missing'),
                          '--isaac-python', sys.executable], capture_output=True, text=True, cwd=ROOT)
    assert 'Renderer: isaac' in old.stdout
    new = subprocess.run([sys.executable, str(ROOT/'replay_video.py'), '--result', str(tmp_path/'missing')],
                         capture_output=True, text=True, cwd=ROOT)
    assert 'Renderer: cpu' in new.stdout


@needs_mujoco
def test_cpu_renderer_draws_measured_pose():
    from software_replay import Camera, ReplayGeometry, render_pose
    geometry = ReplayGeometry(MANIFEST, SCENE_XML)
    camera = Camera([7., -20., 12.], [4., 1., .8], 160, 90)
    p = np.array([b['initial_pos'] for b in MANIFEST['bodies']], float)
    q = np.array([b['initial_quat_wxyz'] for b in MANIFEST['bodies']], float)
    first = render_pose(geometry, camera, p, q)
    assert first.shape == (90, 160, 3) and first.dtype == np.uint8 and len(np.unique(first.reshape(-1, 3), axis=0)) > 20
    moved = p.copy()
    moved[[b['name'] for b in MANIFEST['bodies']].index('body_53')] += [0., 0., 1.5]
    assert np.any(render_pose(geometry, camera, moved, q) != first)


@needs_mujoco
def test_cpu_video_pipeline_on_real_model(tmp_path):
    from measurement_store import atomic_json, atomic_npz
    from replay_video import render_video
    from assess_results import assess_video
    out = tmp_path/'results'/'run'
    out.mkdir(parents=True)
    p = np.array([b['initial_pos'] for b in MANIFEST['bodies']], float)
    q = np.array([b['initial_quat_wxyz'] for b in MANIFEST['bodies']], float)
    times = np.array([0., .001, .1, .2, .25])
    positions = np.stack([p+[0., 0., 2.*t] for t in times])
    atomic_npz(out/'states.npz', time=times, positions=positions, quaternions_wxyz=np.stack([q]*len(times)),
               body_names=np.array([b['name'] for b in MANIFEST['bodies']]))
    atomic_json(out/'manifest.json', MANIFEST)
    atomic_json(out/'status.json', {'dt': .001, 'native_steps_completed': 250,
                                    'status': 'NATIVE_RUN_COMPLETED_UNASSESSED'})
    fake_mp4 = {'mp4_created': True}

    def encoder(frames, movie, fps, expected_count):
        Path(movie).write_bytes(b'\x00\x00\x00\x18ftypmp42'+b'0'*2048)
        return dict(fake_mp4)
    with patch('replay_video.encode_png_sequence', side_effect=encoder):
        capture, encoding = render_video(out, None, ROOT/'generated/soil_final/scene.usda', renderer='cpu')
    assert capture['png_complete'] and capture['captured_frames'] == 4 and encoding['mp4_created']
    assert capture['renderer'] == 'cpu_zbuffer_v1' and encoding['renderer'] == 'cpu_zbuffer_v1'
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((out/'frame_records').glob('*.json'))]
    assert [r['native_step'] for r in records] == [1, 100, 200, 250]
    assert all(r['renderer'] == 'cpu_zbuffer_v1' for r in records)
    issues = []
    status = json.loads((out/'status.json').read_text(encoding='utf-8'))
    video = assess_video(out, status, {'video_requested': True}, issues)
    assert video['video_evidence_available'] and video['replay_provenance_verified'], issues
