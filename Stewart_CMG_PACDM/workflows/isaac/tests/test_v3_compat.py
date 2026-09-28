"""Tensor boundary and lifecycle checks. Warp doubles are not native Warp tests."""
import sys
import types
import numpy as np
import pytest
from isaac_validation.compat import BodyView, ManagerContext


class NumpyRaw:
    prim_paths = ['/one', '/two']
    def __init__(self, result=True):
        self.transforms = np.zeros((2, 7), np.float32); self.transforms[:, 6] = 1
        self.velocities = np.zeros((2, 6), np.float32)
        self.result = result; self.calls = 0
    def get_transforms(self): return self.transforms
    def get_velocities(self): return self.velocities
    def set_transforms(self, x, i): self.calls += 1; return self.result
    def set_velocities(self, x, i): self.calls += 1; return self.result
    def apply_forces_and_torques_at_position(self, f, t, p, i, world):
        assert i.dtype == np.int32 and world is True
        assert all(x.dtype == np.float32 for x in (f, t, p))
        self.calls += 1; return self.result


@pytest.mark.parametrize('result', [False, np.bool_(False)])
@pytest.mark.parametrize('operation', ['transform', 'velocity', 'force'])
def test_false_native_write_results_raise(result, operation):
    raw = NumpyRaw(result); view = BodyView(raw)
    with pytest.raises(RuntimeError, match='failed'):
        if operation == 'transform': view.set_transforms(raw.transforms, [0, 1])
        elif operation == 'velocity': view.set_velocities(raw.velocities, [0, 1])
        else: view.apply_forces_and_torques_at_position(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), [0, 1], True)


@pytest.mark.parametrize('result', [None, True, np.bool_(True)])
def test_supported_native_success_return_values(result):
    raw = NumpyRaw(result); view = BodyView(raw)
    view.set_transforms(raw.transforms, [0, 1])
    view.set_velocities(raw.velocities, [0, 1])
    view.apply_forces_and_torques_at_position(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), [0, 1], True)
    assert raw.calls == 3


@pytest.mark.parametrize('indices', [[-1], [2], [0, 0], [0.0], [], [[0, 1]]])
def test_invalid_body_indices_never_reach_native_api(indices):
    raw = NumpyRaw(); view = BodyView(raw)
    with pytest.raises(ValueError): view.set_velocities(raw.velocities, indices)
    assert raw.calls == 0


@pytest.mark.parametrize('bad', [np.nan, np.inf, 1e100])
def test_invalid_force_never_reaches_native_api(bad):
    raw = NumpyRaw(); view = BodyView(raw); forces = np.zeros((2, 3)); forces[0, 0] = bad
    with pytest.raises(ValueError):
        view.apply_forces_and_torques_at_position(forces, np.zeros((2, 3)), np.zeros((2, 3)), [0, 1], True)
    assert raw.calls == 0


def test_readbacks_are_not_aliases_of_mutable_native_buffers():
    raw = NumpyRaw(); view = BodyView(raw)
    first = view.get_transforms(); raw.transforms[0, 0] = 3
    assert first[0, 0] == 0 and view.get_transforms()[0, 0] == 3


def test_torch_cpu_boundary_with_real_torch():
    torch = pytest.importorskip('torch')
    class Raw(NumpyRaw):
        def get_transforms(self): return torch.from_numpy(self.transforms)
        def set_velocities(self, values, indices):
            assert values.dtype == torch.float32 and indices.dtype == torch.int32
            assert str(values.device) == 'cpu'
            assert tuple(values.shape) == (2, 6)
            return True
    view = BodyView(Raw())
    assert view.frontend == 'torch'
    assert view.get_transforms().shape == (2, 7)
    view.set_velocities(np.zeros((2, 6)), [0, 1])


def test_warp_boundary_protocol_double(monkeypatch):
    class TestWarpArray:
        __module__ = 'warp._src.types'
        def __init__(self, a, dtype, device='cpu'):
            self.data = np.asarray(a, dtype=dtype); self.dtype = dtype; self.device = device
        def numpy(self): return self.data.copy()
    wp = types.ModuleType('warp'); wp.float32 = np.float32; wp.int32 = np.int32; wp.array = TestWarpArray
    monkeypatch.setitem(sys.modules, 'warp', wp)
    class Raw(NumpyRaw):
        def get_transforms(self): return TestWarpArray(self.transforms, np.float32)
        def set_velocities(self, values, indices):
            assert isinstance(values, TestWarpArray) and isinstance(indices, TestWarpArray)
            assert values.data.shape == (2, 6) and values.dtype == np.float32
            assert indices.dtype == np.int32 and values.device == 'cpu'
            return True
    view = BodyView(Raw())
    assert view.frontend == 'warp'
    assert view.get_transforms().shape == (2, 7)
    view.set_velocities(np.zeros((2, 6)), [0, 1])


@pytest.mark.parametrize('fabric_enabled', [True, False])
def test_manager_step_uses_supported_keywords(fabric_enabled):
    class Manager:
        time = 0.; steps = 0
        def get_simulation_time(self): return self.time
        def get_num_physics_steps(self): return self.steps
        def is_fabric_enabled(self): return fabric_enabled
        def step(self, *, steps, update_fabric):
            assert steps == 1 and update_fabric == fabric_enabled
            self.time += .002; self.steps += steps
    manager = Manager(); context = ManagerContext(None, manager)
    context.step()
    assert context.current_time_step_index == 1 and context.current_time == .002


@pytest.mark.parametrize('advance', [False, True])
def test_render_guard_and_settings_restoration(monkeypatch, advance):
    state = {'time': 0., 'steps': 0, '/app/player/playSimulations': True}
    class Settings:
        def get(self, key): return state[key]
        def set_bool(self, key, value): state[key] = value
    carb = types.ModuleType('carb'); carb.settings = types.SimpleNamespace(get_settings=lambda: Settings())
    monkeypatch.setitem(sys.modules, 'carb', carb)
    manager = types.SimpleNamespace(get_simulation_time=lambda: state['time'], get_num_physics_steps=lambda: state['steps'])
    class App:
        def update(self):
            assert state['/app/player/playSimulations'] is False
            if advance: state['time'] += .002; state['steps'] += 1
    context = ManagerContext(App(), manager)
    if advance:
        with pytest.raises(RuntimeError, match='advanced'): context.render()
    else: context.render()
    assert state['/app/player/playSimulations'] is True
