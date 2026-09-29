"""CPU finite-difference checks of state recovery and virtual work.

Uses the source graph, not an independent physics engine. Complements rather
than replaces the optional MuJoCo tests.
"""
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from isaac_validation.mechanics import StateMapping
from isaac_validation.model_checks import physical_case_cmg
from vendor.pacdm_original import PointGraph
from test_v3_runtime_protocol import expected_bodies

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('row', [0, 100, 150, 275, 350, 500, 565, 650, 720, 800, 900, 1000, 1100])
def test_mapping_and_force_power_using_finite_difference_body_poses(row):
    cmg = physical_case_cmg(ROOT, 'nominal')
    with np.load(ROOT/'baseline/reference.npz') as ref:
        q = ref['q'][row].copy(); qd = ref['velocity'][row].copy()
    names, props, poses = expected_bodies(cmg, q)
    graph = PointGraph(cmg, q); eps = 1e-5
    pp, _ = graph.poses(graph.augment(q + eps*qd))
    pm, _ = graph.poses(graph.augment(q - eps*qd))
    transforms = []; velocities = []
    for key in names:
        P = poses[key]; R = P[:3, :3]; com = np.asarray(props[key]['com_m'])
        cp = pp[key][:3, 3] + pp[key][:3, :3] @ com
        cm = pm[key][:3, 3] + pm[key][:3, :3] @ com
        omega_matrix = (pp[key][:3, :3] - pm[key][:3, :3])/(2*eps) @ R.T
        omega = np.array([omega_matrix[2, 1], omega_matrix[0, 2], omega_matrix[1, 0]])
        transforms.append(np.r_[P[:3, 3], Rotation.from_matrix(R).as_quat()])
        velocities.append(np.r_[(cp-cm)/(2*eps), omega])
    mapping = StateMapping(cmg, names, [props[key]['com_m'] for key in names])
    state = mapping.read(np.array(transforms), np.array(velocities))
    np.testing.assert_allclose(state['q'], q, atol=1e-12)
    np.testing.assert_allclose(state['velocity'], qd, atol=1e-8)
    assert state['closure_error_m'] < 1e-8
    assert state['joint_error_m'] < 1e-12 and state['joint_error_rad'] < 1e-12
    rng = np.random.default_rng(123 + row)
    force = rng.normal(0, 100, 6); wrench = rng.normal(0, 10, 6)
    f, tau = mapping.forces(state, force, wrench)
    ip = mapping.idx['platform']; vcom = state['com_velocities'][ip, :3]
    w = state['com_velocities'][ip, 3:]
    vo = vcom - np.cross(w, state['com'][ip]-state['p'][ip])
    native_power = np.sum(f*state['com_velocities'][:, :3]) + np.sum(tau*state['com_velocities'][:, 3:])
    expected_power = force @ qd[mapping.active] + wrench[:3] @ vo + wrench[3:] @ w
    assert native_power == pytest.approx(expected_power, abs=2e-7)
    np.testing.assert_allclose(np.sum(f, axis=0), wrench[:3], atol=1e-12)
    np.testing.assert_allclose(np.sum(tau + np.cross(state['com']-state['p'][ip], f), axis=0), wrench[3:], atol=1e-11)
    # The prismatic-frame separation is its allowed length, not a transverse gap.
    assert np.min(state['q'][mapping.active]) > .1
