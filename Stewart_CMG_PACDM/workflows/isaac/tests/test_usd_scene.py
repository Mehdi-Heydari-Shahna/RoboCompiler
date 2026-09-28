"""Independent USD read-back tests. These do not claim to run PhysX."""
import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation
import pytest
pytest.importorskip("pxr", reason="OpenUSD is an optional external engine dependency")
from pxr import Usd, UsdGeom, UsdPhysics

from isaac_validation.usd_scene import build_scene, audit_scene
from stewart.model import make_cmg, inverse_seed
from vendor.pacdm_original import PointGraph


def quat_rotation(q):
    return Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()


def frame(position, orientation):
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = quat_rotation(orientation)
    return T


class USDSceneTests(unittest.TestCase):
    def test_topology_and_physics_settings(self):
        cmg = make_cmg()
        q = inverse_seed(cmg, cmg["geometry"]["nominal_pose"])
        result = build_scene(cmg, q)
        audit = result["audit"]
        json.dumps({k: v for k, v in result.items() if k != "stage"})
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["counts"], dict(rigid_bodies=19, revolute=12, prismatic=6, spherical=6))
        self.assertAlmostEqual(audit["usd_total_mass_kg"], 29.9, places=6)
        self.assertEqual(audit["articulation_roots"], 0)
        self.assertEqual(audit["physical_drives"], 0)
        self.assertEqual(audit["collision_shapes"], 0)
        for path in result["body_paths"].values():
            prim = result["stage"].GetPrimAtPath(path)
            for suffix in ("linearDamping", "angularDamping", "sleepThreshold", "stabilizationThreshold"):
                self.assertEqual(prim.GetAttribute("physxRigidBody:" + suffix).Get(), 0)
            self.assertTrue(prim.GetAttribute("physxRigidBody:enableGyroscopicForces").Get())
            self.assertFalse(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())

    def test_all_local_joint_frames_reproduce_tilted_cmg_coordinates(self):
        cmg = make_cmg()
        q = inverse_seed(cmg, [.017, -.011, .63, .06, -.035, .025])
        result = build_scene(cmg, q)
        cache = UsdGeom.XformCache()
        body_pose = {n: np.asarray(cache.GetLocalToWorldTransform(result["stage"].GetPrimAtPath(p))).T
                     for n, p in result["body_paths"].items()}
        body_pose["world"] = np.eye(4)
        for record in cmg["joints"]:
            if not record["id"].startswith("leg_"):
                continue
            joint = UsdPhysics.Joint(result["stage"].GetPrimAtPath(result["joint_paths"][record["id"]]))
            f0 = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            f1 = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            delta = np.linalg.inv(body_pose[record["base_body"]] @ f0) @ body_pose[record["follower_body"]] @ f1
            coordinate = q[cmg["coordinate_ids"].index(record["id"])]
            axis = np.asarray(record["axis"])
            if record["type"] == "revolute":
                np.testing.assert_allclose(delta[:3, 3], 0, atol=3e-8)
                np.testing.assert_allclose(delta[:3, :3], Rotation.from_rotvec(axis * coordinate).as_matrix(), atol=8e-8)
                typed = UsdPhysics.RevoluteJoint(joint.GetPrim())
                factor = 180 / np.pi
            else:
                np.testing.assert_allclose(delta[:3, 3], coordinate * axis, atol=3e-8)
                np.testing.assert_allclose(delta[:3, :3], np.eye(3), atol=8e-8)
                typed = UsdPhysics.PrismaticJoint(joint.GetPrim())
                factor = 1
            self.assertAlmostEqual(typed.GetLowerLimitAttr().Get(), factor * record["limits"]["lower"], places=5)
            self.assertAlmostEqual(typed.GetUpperLimitAttr().Get(), factor * record["limits"]["upper"], places=5)
            self.assertEqual(typed.GetAxisAttr().Get(), "XYZ"[int(np.argmax(axis))])
        self.assertLess(result["audit"]["maximum_initial_closure_error_m"], 5e-8)

    def test_payload_aggregation_conserves_kinetic_and_gravitational_energy(self):
        cmg = make_cmg(13)
        mount = next(j for j in cmg["joints"] if j["id"] == "payload_mount")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .3]).as_matrix()
        transform[:3, 3] = [.03, -.02, .055]
        mount["T_BJ"] = transform.tolist()
        payload = next(b for b in cmg["bodies"] if b["id"] == "payload")
        payload["inertia_kg_m2"] = [[.3, .015, .009], [.015, .24, -.012], [.009, -.012, .2]]
        q = inverse_seed(cmg, [.02, -.01, .64, .04, -.05, .07])
        result = build_scene(cmg, q)
        graph = PointGraph(cmg, q)
        poses, _ = graph.poses(graph.augment(q))
        props = result["audit"]["expected_body_properties"]["platform"]
        v_origin, omega = np.array([.31, -.17, .29]), np.array([.2, .4, -.3])
        R, origin = poses["platform"][:3, :3], poses["platform"][:3, 3]
        energy, potential = 0., 0.
        for name in ("platform", "payload"):
            body = next(b for b in cmg["bodies"] if b["id"] == name)
            T = poses[name]
            com = T[:3, :3] @ body["com_m"] + T[:3, 3]
            v = v_origin + np.cross(omega, com - origin)
            I_world = T[:3, :3] @ np.asarray(body["inertia_kg_m2"]) @ T[:3, :3].T
            energy += .5 * body["mass_kg"] * np.dot(v, v) + .5 * omega @ I_world @ omega
            potential -= body["mass_kg"] * np.dot(cmg["gravity_m_s2"], com)
        com = R @ props["com_m"] + origin
        v = v_origin + np.cross(omega, com - origin)
        combined = .5 * props["mass_kg"] * np.dot(v, v) + .5 * omega @ R @ props["inertia_kg_m2"] @ R.T @ omega
        self.assertAlmostEqual(energy, combined, places=12)
        self.assertAlmostEqual(potential, -props["mass_kg"] * np.dot(cmg["gravity_m_s2"], com), places=12)

    def test_saved_stage_roundtrip_retains_mass_and_physx_parameters(self):
        cmg = make_cmg(13)
        q = inverse_seed(cmg, cmg["geometry"]["nominal_pose"])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "plant.usda"
            result = build_scene(cmg, q, path, dt=.001)
            loaded = Usd.Stage.Open(str(path))
            audit = audit_scene(cmg, q, loaded, result["body_paths"], result["joint_paths"])
            self.assertEqual(audit["status"], "PASS")
            text = path.read_text()
            self.assertIn("PhysxRigidBodyAPI", text)
            self.assertNotIn("PhysicsArticulationRootAPI", text)
            scene = loaded.GetPrimAtPath(result["scene_path"])
            self.assertEqual(scene.GetAttribute("physxScene:timeStepsPerSecond").Get(), 1000)
            self.assertEqual(scene.GetAttribute("physxScene:solverType").Get(), "TGS")

    def test_bad_initial_closure_and_hidden_chart_mass_are_rejected(self):
        cmg = make_cmg()
        q = inverse_seed(cmg, cmg["geometry"]["nominal_pose"])
        bad = q.copy()
        bad[8] += .001
        with self.assertRaisesRegex(ValueError, "not physically closed"):
            build_scene(cmg, bad)
        altered = copy.deepcopy(cmg)
        altered["bodies"][1]["mass_kg"] = .1
        with self.assertRaisesRegex(ValueError, "Cannot discard"):
            build_scene(altered, q)


if __name__ == "__main__":
    unittest.main()
