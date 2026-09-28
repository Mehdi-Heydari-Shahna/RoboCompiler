"""Execute production run_case against deterministic in-process API doubles.

This checks control flow, API argument shapes, logging, clock/failure handling
and the uploaded inertia regression. It is NOT PhysX or Isaac execution; the
stationary body fixture is deliberately not a physics integrator. Artifacts
remain in pytest temporary directories and are never release mission results.
"""
import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from isaac_validation import compat
from isaac_validation.runtime import run_case
from isaac_validation.model_checks import physical_case_cmg
from isaac_validation.worker_check import check_worker, smoke_checks
from vendor.pacdm_original import PointGraph

ROOT = Path(__file__).resolve().parents[1]


def expected_bodies(cmg, seed):
    graph = PointGraph(cmg, seed)
    poses, _ = graph.poses(graph.augment(seed))
    source = {body['id']: body for body in cmg['bodies']}
    names = sorted({'platform'} | {f'leg_{i}_{part}' for i in range(6) for part in ('yoke', 'barrel', 'rod')})
    props = {key: dict(mass_kg=source[key]['mass_kg'], com_m=source[key]['com_m'],
                      inertia_kg_m2=source[key]['inertia_kg_m2']) for key in names}
    pieces = []
    for key in ('platform', 'payload'):
        T = np.linalg.inv(poses['platform']) @ poses[key]
        body = source[key]
        pieces.append((body['mass_kg'], T[:3, :3] @ body['com_m'] + T[:3, 3],
                       T[:3, :3] @ np.asarray(body['inertia_kg_m2']) @ T[:3, :3].T))
    total = sum(m for m, _, _ in pieces)
    center = sum(m * c for m, c, _ in pieces) / total
    inertia = sum(I + m * (np.dot(c-center, c-center)*np.eye(3) - np.outer(c-center, c-center))
                  for m, c, I in pieces)
    props['platform'] = dict(mass_kg=total, com_m=center.tolist(), inertia_kg_m2=inertia.tolist())
    return names, props, poses


class ProtocolHarness:
    def __init__(self, monkeypatch, case='nominal', *, bad_inertia=False, bad_clock=False,
                 rejected_force=False, stop_at=None, fake_video=False):
        cmg = physical_case_cmg(ROOT, case)
        with np.load(ROOT/'baseline/reference.npz') as z: seed = z['q'][0].copy()
        self.names, props, poses = expected_bodies(cmg, seed)
        self.prim_paths = [f'/World/Stewart/Bodies/{name}' for name in self.names]
        self.mass = np.array([props[key]['mass_kg'] for key in self.names], np.float32)
        self.inertia = np.array([props[key]['inertia_kg_m2'] for key in self.names], np.float32).reshape(19, 9)
        if bad_inertia: self.inertia[0, 0] += .003
        coms = []
        for key in self.names:
            _, axes = np.linalg.eigh(np.array(props[key]['inertia_kg_m2']))
            if np.linalg.det(axes) < 0: axes[:, 0] *= -1
            coms.append(np.r_[props[key]['com_m'], Rotation.from_matrix(axes).as_quat()])
        self.com = np.asarray(coms, np.float32)
        self.transforms = np.array([np.r_[poses[key][:3, 3], Rotation.from_matrix(poses[key][:3, :3]).as_quat()]
                                    for key in self.names], np.float32)
        self.velocities = np.zeros((19, 6), np.float32)
        self.current_time = 0.; self.current_time_step_index = 0
        self.stopped = False; self.assignments = []; self.force_calls = 0
        self.bad_clock = bad_clock; self.rejected_force = rejected_force
        self.stop_at = stop_at; self.dt = .002; self.render_calls = 0
        self.video_init_clock = None
        harness = self
        class Stage:
            def GetRootLayer(self): return self
            def Export(self, path):
                Path(path).write_text('# TEST PROTOCOL FIXTURE, NOT A USD SIMULATION\n')
                return True
        stage = Stage()
        scene = dict(stage=stage, body_paths=dict(zip(self.names, self.prim_paths)),
                     audit=dict(expected_body_properties=props,
                                body_properties={key: dict(initial_transform=poses[key].tolist()) for key in self.names}))
        def build_scene(actual_cmg, q, stage, dt):
            harness.dt = dt
            assert actual_cmg == cmg
            np.testing.assert_array_equal(q, seed)
            return scene
        settings = types.SimpleNamespace(set_bool=lambda *args: None)
        carb = types.ModuleType('carb'); carb.settings = types.SimpleNamespace(get_settings=lambda: settings)
        omni = types.ModuleType('omni'); omni.__path__ = []
        usd = types.ModuleType('omni.usd')
        usd.get_context = lambda: types.SimpleNamespace(new_stage=lambda: None, get_stage=lambda: stage)
        omni.usd = usd
        compiler = types.ModuleType('isaac_validation.usd_scene'); compiler.build_scene = build_scene
        monkeypatch.setitem(sys.modules, 'carb', carb)
        monkeypatch.setitem(sys.modules, 'omni', omni)
        monkeypatch.setitem(sys.modules, 'omni.usd', usd)
        monkeypatch.setitem(sys.modules, 'isaac_validation.usd_scene', compiler)
        monkeypatch.setattr(compat, 'start_physics', lambda app, dt: (self, self, 'TEST_PROTOCOL_ONLY'))
        if fake_video:
            class TestVideo:
                def __init__(self, stage, context, path, run_id):
                    harness.video_init_clock = context.current_time
                    self.context = context; self.times = []
                def capture(self, t, *args):
                    self.context.render(); self.times.append(t)
                def finish(self):
                    return dict(passed=False, source='TEST_PROTOCOL_ONLY', frames=len(self.times))
            video_module = types.ModuleType('isaac_validation.video'); video_module.LiveVideo = TestVideo
            monkeypatch.setitem(sys.modules, 'isaac_validation.video', video_module)

    def create_rigid_body_view(self, pattern):
        assert pattern == '/World/Stewart/Bodies/*'; return self
    def get_transforms(self): return self.transforms
    def get_velocities(self): return self.velocities
    def get_coms(self): return self.com
    def get_masses(self): return self.mass
    def get_inertias(self): return self.inertia
    def get_gravity(self): return np.array([0, 0, -9.81], np.float32)
    def set_transforms(self, data, indices):
        assert data.dtype == np.float32 and indices.dtype == np.int32
        self.transforms[indices] = data[indices]; self.assignments.append(('transforms', self.current_time))
        return True
    def set_velocities(self, data, indices):
        self.velocities[indices] = data[indices]; self.assignments.append(('velocities', self.current_time))
        return True
    def apply_forces_and_torques_at_position(self, forces, torques, positions, indices, world):
        assert world is True and indices.dtype == np.int32
        assert forces.shape == torques.shape == positions.shape == (19, 3)
        assert all(a.dtype == np.float32 for a in (forces, torques, positions))
        np.testing.assert_allclose(np.sum(forces, axis=0), [0, 0, 0], atol=1e-4)
        self.force_calls += 1
        return np.bool_(not self.rejected_force)
    def step(self, render=False):
        self.current_time_step_index += 1
        self.current_time += self.dt * (2 if self.bad_clock else 1)
    def is_running(self):
        return self.stop_at is None or self.current_time_step_index < self.stop_at
    def stop(self): self.stopped = True
    def render(self): self.render_calls += 1


def test_production_runtime_completes_1001_sample_smoke_protocol(monkeypatch, tmp_path):
    h = ProtocolHarness(monkeypatch)
    result = run_case(h, ROOT, tmp_path, 'nominal', 'TEST_PROTOCOL_ONLY', duration=2.)
    assert result['completed'] and result['native_inertial_audit_passed']
    assert result['startup_route'] == 'TEST_PROTOCOL_ONLY'
    assert h.force_calls == 1000 and h.stopped
    assert h.assignments == [('transforms', 0.), ('velocities', 0.)]
    assert check_worker(tmp_path, 'nominal', 'TEST_PROTOCOL_ONLY', 0, 2., .002)['passed']
    assert smoke_checks(tmp_path, 'TEST_PROTOCOL_ONLY', False)['passed']
    with np.load(tmp_path/'nominal.npz') as z:
        assert z['time'].shape == (1001,)
        assert z['body_transforms'].shape == (1001, 19, 7)
        assert z['q'].shape == (1001, 24)
        assert np.max(z['force']) > 10
    # A diagnostic never grants a complete mission pass.
    assert not result['passed'] and not result['eligible_for_mission_acceptance']


@pytest.mark.parametrize('case', ['nominal', 'fine', 'heavy_payload', 'no_feedforward', 'zero_actuation'])
def test_every_case_keeps_its_name_and_force_mode(monkeypatch, tmp_path, case):
    h = ProtocolHarness(monkeypatch, case)
    dt = .001 if case == 'fine' else .002
    result = run_case(h, ROOT, tmp_path, case, 'TEST_PROTOCOL_ONLY', dt=dt, duration=4*dt)
    assert result['name'] == case
    assert (tmp_path/f'{case}.native_inertial_audit.json').is_file()
    assert (tmp_path/f'{case}.json').is_file()
    assert not list(tmp_path.glob('leg_*.json'))
    with np.load(tmp_path/f'{case}.npz') as z:
        if case in ('no_feedforward', 'zero_actuation'):
            assert np.max(abs(z['force'])) < .01
        else: assert np.max(abs(z['force'])) > 10
    if case == 'heavy_payload':
        cmg = json.loads((tmp_path/f'{case}.cmg.json').read_text())
        assert next(b['mass_kg'] for b in cmg['bodies'] if b['id'] == 'payload') == 14.


def test_rejected_native_inertia_is_saved_before_exception(monkeypatch, tmp_path):
    h = ProtocolHarness(monkeypatch, bad_inertia=True)
    with pytest.raises(RuntimeError, match='PhysX inertial import mismatch'):
        run_case(h, ROOT, tmp_path, 'nominal', 'TEST_PROTOCOL_ONLY', duration=.004)
    audit = json.loads((tmp_path/'nominal.native_inertial_audit.json').read_text())
    assert not audit['passed'] and 'raw' in audit and audit['errors']
    assert (tmp_path/'nominal.model_audit.json').is_file()
    assert (tmp_path/'nominal.usda').is_file()
    assert not (tmp_path/'nominal.json').exists()
    assert h.force_calls == 0


@pytest.mark.parametrize('fault, message, samples', [
    ('bad_clock', 'Physics clock mismatch', 1),
    ('rejected_force', 'force application failed', 1),
    ('stop_at', 'closed before case completion', 2),
])
def test_mid_run_failures_preserve_only_recorded_samples(monkeypatch, tmp_path, fault, message, samples):
    h = ProtocolHarness(monkeypatch, **{fault: 2 if fault == 'stop_at' else True})
    with pytest.raises(RuntimeError, match=message):
        run_case(h, ROOT, tmp_path, 'nominal', 'TEST_PROTOCOL_ONLY', duration=.02)
    assert h.stopped
    assert not (tmp_path/'nominal.json').exists()
    with np.load(tmp_path/'nominal.npz') as z:
        assert len(z['time']) == samples
        assert np.all(np.isfinite(z['q']))
    assert not check_worker(tmp_path, 'nominal', 'TEST_PROTOCOL_ONLY', 0, .02, .002)['passed']


def test_optional_video_interface_does_not_advance_physics(monkeypatch, tmp_path):
    h = ProtocolHarness(monkeypatch, fake_video=True)
    run_case(h, ROOT, tmp_path, 'nominal', 'TEST_PROTOCOL_ONLY', duration=.10, video=True)
    assert h.video_init_clock == 0.
    assert h.current_time_step_index == 50 and h.force_calls == 50
    assert h.render_calls == 4
