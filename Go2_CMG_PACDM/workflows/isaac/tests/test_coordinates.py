"""Independent finite-difference checks of the PhysX chart/COM boundary."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2.model import native_qpos_from_chart, velocity_map
from isaac_validation.coordinates import from_chart, to_chart, rotation


@pytest.mark.parametrize("seed", [0, 5, 91])
def test_base_com_velocity_matches_independent_pose_derivative(seed):
    rng = np.random.default_rng(seed)
    for _ in range(12):
        q = rng.uniform(-0.6, 0.6, 18)
        q[3] = rng.uniform(-2.8, 2.8)
        q[4] = rng.uniform(-1.25, 1.25)
        v = rng.uniform(-0.7, 0.7, 18)
        com_local = rng.uniform(-0.15, 0.15, 3)
        pose, velocity = from_chart(q, v, com_local)
        step = 2e-6
        q_plus, q_minus = q + step * v, q - step * v
        r_plus = Rotation.from_euler("ZYX", q_plus[3:6]).as_matrix()
        r_minus = Rotation.from_euler("ZYX", q_minus[3:6]).as_matrix()
        com_plus = q_plus[:3] + r_plus @ com_local
        com_minus = q_minus[:3] + r_minus @ com_local
        np.testing.assert_allclose(velocity[:3], (com_plus - com_minus) / (2 * step), atol=2e-10)
        matrix = Rotation.from_euler("ZYX", q[3:6]).as_matrix()
        derivative = (r_plus - r_minus) / (2 * step)
        omega_skew = derivative @ matrix.T
        expected_omega = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
        np.testing.assert_allclose(velocity[3:], expected_omega, atol=2e-10)
        # Verify source native angular velocity (body axes) and Isaac's world
        # angular velocity describe the same motion; they cannot be interchanged.
        native_velocity = velocity_map(q) @ v
        np.testing.assert_allclose(velocity[3:], matrix @ native_velocity[3:6], atol=1e-13)
        native_pose = native_qpos_from_chart(q)
        np.testing.assert_allclose(pose[:3], native_pose[:3], atol=1e-13)
        np.testing.assert_allclose(rotation(pose[3:7]), matrix, atol=1e-13)
        recovered_q, recovered_v = to_chart(pose, velocity, q[6:], v[6:], com_local)
        np.testing.assert_allclose(recovered_q, q, atol=1e-13)
        np.testing.assert_allclose(recovered_v, v, atol=2e-13)


def test_nonzero_com_offset_is_removed_from_base_origin_velocity():
    # A base rotating about its actor origin has a translating COM; the chart
    # origin must remain stationary even though get_root_velocities is nonzero.
    q, v = np.zeros(18), np.zeros(18)
    q[:6] = [0.2, 0.1, 0.4, 0.4, -0.3, 0.2]
    v[3:6] = [0.9, -0.5, 0.2]
    com = np.array([0.1, -0.07, 0.05])
    pose, physical_velocity = from_chart(q, v, com)
    assert np.linalg.norm(physical_velocity[:3]) > 0.05
    _, actual = to_chart(pose, physical_velocity, q[6:], v[6:], com)
    np.testing.assert_allclose(actual[:3], 0, atol=1e-14)
    np.testing.assert_allclose(actual[3:6], v[3:6], atol=1e-14)


def test_declared_chart_boundary_is_rejected():
    q, v = np.zeros(18), np.zeros(18)
    q[4] = 1.47
    pose, velocity = from_chart(q, v, np.zeros(3))
    with pytest.raises(ValueError, match="outside its declared domain"):
        to_chart(pose, velocity, q[6:], v[6:], np.zeros(3))
