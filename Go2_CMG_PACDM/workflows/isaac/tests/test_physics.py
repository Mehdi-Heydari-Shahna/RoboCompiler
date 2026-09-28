"""Tensor-boundary tests with scrambled native names (not a PhysX rollout).

The fake supplies independently computed rigid poses; it makes no claim to
simulate contact, articulation dynamics, native API existence, or stability.
"""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2.contact import FootKinematics
from go2.model import chart_from_native, load_model
from isaac_validation.physics import PhysicsRobot


class FakeView:
    count, max_dofs, max_links = 1, 12, 13

    def __init__(self, cmg):
        self.cmg = cmg
        self.expected_joints = cmg["coordinate_ids"][6:]
        physical_bodies = [body for body in cmg["bodies"] if body["mass_kg"] > 0]
        # Deliberately put the base away from link index zero and scramble the
        # hip/thigh/calf ordering so accidental positional indexing cannot pass.
        permutation = [4, 9, 11, 1, 7, 0, 8, 5, 2, 12, 3, 10, 6]
        links = [physical_bodies[index]["id"] for index in permutation]
        dof_permutation = [9, 1, 11, 6, 2, 5, 8, 0, 10, 7, 4, 3]
        dofs = [self.expected_joints[index] for index in dof_permutation]
        self.shared_metatype = SimpleNamespace(link_names=links, dof_names=dofs)
        self.jp, self.jv = np.zeros((1, 12)), np.zeros((1, 12))
        self.root_pose = np.array([[0., 0., 0., 0., 0., 0., 1.]])
        self.root_velocity = np.zeros((1, 6))
        records = {body["id"]: body for body in physical_bodies}
        self.coms = np.asarray([[records[name]["com_m"] + [0., 0., 0., 1.] for name in links]])
        self.kin = FootKinematics(cmg, cmg["q_reference"])
        self.actuation = np.zeros((1, 12))
        self.force_call = None

    def get_coms(self): return self.coms.copy()
    def get_root_transforms(self): return self.root_pose.copy()
    def get_root_velocities(self): return self.root_velocity.copy()
    def get_dof_positions(self): return self.jp.copy()
    def get_dof_velocities(self): return self.jv.copy()
    def set_root_transforms(self, values, indices): self.root_pose = values.copy()
    def set_root_velocities(self, values, indices): self.root_velocity = values.copy()
    def set_dof_positions(self, values, indices): self.jp = values.copy()
    def set_dof_velocities(self, values, indices): self.jv = values.copy()
    def set_dof_actuation_forces(self, values, indices): self.actuation = values.copy()

    def physical_q(self):
        xyzw = self.root_pose[0, 3:]
        joint_values = dict(zip(self.shared_metatype.dof_names, self.jp[0]))
        native = np.r_[self.root_pose[0, :3], xyzw[[3, 0, 1, 2]],
                       [joint_values[name] for name in self.expected_joints]]
        return chart_from_native(native)

    def poses(self):
        return self.kin.tree.poses(self.physical_q())[0]

    def get_link_transforms(self):
        poses = self.poses()
        return np.asarray([[np.r_[poses[name][:3, 3], Rotation.from_matrix(poses[name][:3, :3]).as_quat()]
                            for name in self.shared_metatype.link_names]])

    def apply_forces_and_torques_at_position(self, force, torque, positions, indices, is_global):
        self.force_call = dict(force=force.copy(), torque=torque,
                               positions=positions.copy(), indices=indices.copy(), is_global=is_global)


@pytest.fixture
def tensor_robot(monkeypatch):
    cmg = load_model()
    view = FakeView(cmg)
    simulation = SimpleNamespace(set_subspace_roots=lambda path: None,
                                 create_articulation_view=lambda path: view,
                                 update_articulations_kinematic=lambda: None)
    omni, physics, tensors = (ModuleType(name) for name in ("omni", "omni.physics", "omni.physics.tensors"))
    def create_view(frontend, stage_id=-1, backend="physx"):
        assert frontend == "numpy" and stage_id == 17 and backend == "physx"
        return simulation
    tensors.create_simulation_view = create_view
    omni.physics = physics
    physics.tensors = tensors
    for module in (omni, physics, tensors):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    robot = PhysicsRobot({"base_path": "/World/Go2/base"}, cmg,
                         cmg["coordinate_ids"][6:], stage_id=17)
    assert robot.base_index == 5
    return robot, view, cmg


def test_scrambled_dof_state_and_effort_mapping(tensor_robot):
    robot, view, cmg = tensor_robot
    q = np.asarray(cmg["q_reference"], float)
    q[:6] = [0.1, -0.2, 0.4, 0.7, -0.3, 0.2]
    q[6:] += np.linspace(-0.03, 0.03, 12)
    v = np.linspace(-0.15, 0.18, 18)
    robot.set_state(q, v)
    for index, name in enumerate(cmg["coordinate_ids"][6:]):
        native_index = view.shared_metatype.dof_names.index(name)
        assert view.jp[0, native_index] == pytest.approx(q[6 + index], abs=1e-7)
        assert view.jv[0, native_index] == pytest.approx(v[6 + index], abs=1e-8)
    actual_q, actual_v = robot.state()
    np.testing.assert_allclose(actual_q, q, atol=1e-7)
    np.testing.assert_allclose(actual_v, v, atol=3e-8)
    efforts = np.arange(12, dtype=float) - 4.5
    robot.effort(efforts)
    for index, name in enumerate(cmg["coordinate_ids"][6:]):
        native_index = view.shared_metatype.dof_names.index(name)
        assert view.actuation[0, native_index] == efforts[index]


def test_link_pose_foot_fk_matches_original_cmg_with_scrambled_links(tensor_robot):
    robot, view, cmg = tensor_robot
    q = np.asarray(cmg["q_reference"], float)
    q[:6] = [0.7, -0.4, 0.6, -0.65, 0.22, -0.12]
    q[6:] += np.linspace(-0.07, 0.07, 12)
    robot.set_state(q, np.zeros(18))
    expected = FootKinematics(cmg, q).points(q)
    np.testing.assert_allclose(robot.foot_positions(), expected, atol=1e-7)


def test_world_push_targets_actual_base_com_even_when_base_is_not_first(tensor_robot):
    robot, view, cmg = tensor_robot
    q = np.asarray(cmg["q_reference"], float)
    q[:6] = [0.5, -0.25, 0.7, 0.5, -0.2, 0.3]
    robot.set_state(q, np.zeros(18))
    force = np.array([2., 32., -4.])
    robot.push(force)
    call = view.force_call
    assert call["is_global"] is True
    assert call["torque"] is None
    assert call["indices"].tolist() == [0]
    np.testing.assert_allclose(call["force"].sum(axis=1)[0], force)
    np.testing.assert_allclose(call["force"][0, robot.base_index], force)
    assert np.count_nonzero(np.linalg.norm(call["force"][0], axis=1)) == 1
    poses = view.poses()
    bodies = {body["id"]: body for body in cmg["bodies"]}
    for index, name in enumerate(view.shared_metatype.link_names):
        expected = (poses[name] @ np.r_[bodies[name]["com_m"], 1.])[:3]
        np.testing.assert_allclose(call["positions"][0, index], expected, atol=6e-8)
    # Applying at the actor origin would inject an unintended moment when COM
    # is offset. The authored base has a real nonzero COM, so test that case.
    base = poses["base"]
    assert np.linalg.norm(call["positions"][0, robot.base_index] - base[:3, 3]) > 0.005
