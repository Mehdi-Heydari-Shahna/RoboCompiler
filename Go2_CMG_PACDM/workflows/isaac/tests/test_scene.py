"""Offline scene evidence, not an Isaac Sim/PhysX dynamics validation."""
from pathlib import Path
import json
import math

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

from isaac_validation.scene import build_scene

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scene():
    stage = Usd.Stage.CreateInMemory()
    metadata = build_scene(stage, ROOT)
    return stage, metadata


def _transform(prim):
    # USD matrices use row vectors; the independent test uses column vectors.
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T


def _rotation_from_quat(q):
    return Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()


def test_physical_topology_no_drives_and_contact_reports(scene):
    stage, metadata = scene
    assert metadata["source_counts"] == dict(
        rigid_bodies=13, revolute_joints=12, robot_colliders=23,
        visual_meshes=33, unique_obj_meshes=16, foot_colliders=4,
        environment_colliders=6)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(bodies) == 13
    assert [str(p.GetPath()) for p in roots] == ["/World/Go2/base"]
    assert not any(p.GetTypeName() == "PhysicsFixedJoint" for p in stage.Traverse())
    for body in bodies:
        assert body.GetAttribute("physxContactReport:threshold").Get() == 0.0
        assert body.GetAttribute("physxRigidBody:linearDamping").Get() == 0.0
        assert body.GetAttribute("physxRigidBody:angularDamping").Get() == 0.0
        assert body.GetAttribute("physxRigidBody:enableGyroscopicForces").Get()
    assert roots[0].GetAttribute("physxArticulation:enabledSelfCollisions").Get()
    assert all(not p.HasAPI(UsdPhysics.DriveAPI, "angular") for p in stage.Traverse())
    assert set(metadata["foot_collider_paths"]) == {"FL", "FR", "RL", "RR"}


def test_joint_frames_limits_and_armature_match_cmg(scene):
    stage, metadata = scene
    cmg = json.loads((ROOT / "data/go2_cmg.json").read_text())
    for record in (j for j in cmg["joints"] if j.get("actuated")):
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(metadata["joint_paths"][record["id"]]))
        assert joint.GetBody0Rel().GetTargets() == [metadata["body_paths"][record["base_body"]]]
        assert joint.GetBody1Rel().GetTargets() == [metadata["body_paths"][record["follower_body"]]]
        for side, name in ((0, "T_BJ"), (1, "T_FJ")):
            authored = np.eye(4)
            authored[:3, 3] = getattr(joint, f"GetLocalPos{side}Attr")().Get()
            authored[:3, :3] = _rotation_from_quat(getattr(joint, f"GetLocalRot{side}Attr")().Get())
            np.testing.assert_allclose(authored, record[name], atol=2e-8)
        np.testing.assert_allclose(
            np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]),
            [record["limits"]["lower"], record["limits"]["upper"]], atol=2e-7)
        assert not joint.GetCollisionEnabledAttr().Get()
        assert joint.GetPrim().GetAttribute("physxJoint:armature").Get() == pytest.approx(record["armature"])
        assert joint.GetPrim().GetAttribute("physxJoint:jointFriction").Get() == 0.0
        assert joint.GetAxisAttr().Get() == "XYZ"[int(np.argmax(record["axis"]))]


def test_mass_com_inertia_against_independent_mujoco_compiler(scene):
    mujoco = pytest.importorskip("mujoco")
    stage, metadata = scene
    model = mujoco.MjModel.from_xml_path(str(ROOT / "upstream/unitree_go2/go2.xml"))
    total = 0.0
    for name, path in metadata["body_paths"].items():
        bid = model.body(name).id
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        assert mass.GetMassAttr().Get() == pytest.approx(model.body_mass[bid], rel=8e-8)
        np.testing.assert_allclose(mass.GetCenterOfMassAttr().Get(), model.body_ipos[bid], atol=1e-8)
        usd_rotation = _rotation_from_quat(mass.GetPrincipalAxesAttr().Get())
        usd_tensor = usd_rotation @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ usd_rotation.T
        source_rotation = Rotation.from_quat(model.body_iquat[bid][[1, 2, 3, 0]]).as_matrix()
        source_tensor = source_rotation @ np.diag(model.body_inertia[bid]) @ source_rotation.T
        np.testing.assert_allclose(usd_tensor, source_tensor, atol=8e-9, rtol=5e-7)
        total += mass.GetMassAttr().Get()
    assert total == pytest.approx(model.body_mass.sum(), rel=8e-8)
    assert total == pytest.approx(metadata["total_mass_kg"], rel=8e-8)


def test_zero_pose_and_every_collider_match_independent_mujoco_fk(scene):
    mujoco = pytest.importorskip("mujoco")
    stage, metadata = scene
    model = mujoco.MjModel.from_xml_path(str(ROOT / "upstream/unitree_go2/go2.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    data.qpos[:3] = metadata["initial_base_position_m"]
    data.qpos[3] = 1
    mujoco.mj_forward(model, data)
    for name, path in metadata["body_paths"].items():
        bid = model.body(name).id
        usd = _transform(stage.GetPrimAtPath(path))
        np.testing.assert_allclose(usd[:3, 3], data.xpos[bid], atol=1e-10)
        np.testing.assert_allclose(usd[:3, :3], data.xmat[bid].reshape(3, 3), atol=1e-7)
    for record in metadata["geometry_records"]:
        if record["visual"]:
            continue
        bid = model.body(record["body"]).id
        gid = model.body_geomadr[bid] + record["source_geom_index"]
        prim = stage.GetPrimAtPath(record["path"])
        transform = _transform(prim)
        np.testing.assert_allclose(transform[:3, 3], data.geom_xpos[gid], atol=1e-9)
        if record["kind"] == "box":
            scale = np.linalg.norm(transform[:3, :3], axis=0)
            rotation = transform[:3, :3] / scale
            np.testing.assert_allclose(scale, model.geom_size[gid], atol=2e-9)
            assert UsdGeom.Cube(prim).GetSizeAttr().Get() == 2.0
        else:
            rotation = transform[:3, :3]
            assert prim.GetAttribute("radius").Get() == pytest.approx(model.geom_size[gid][0])
            if record["kind"] == "cylinder":
                assert prim.GetAttribute("height").Get() == pytest.approx(2 * model.geom_size[gid][1])
                assert prim.GetAttribute("axis").Get() == "Z"
        np.testing.assert_allclose(rotation, data.geom_xmat[gid].reshape(3, 3), atol=2e-7)
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert prim.GetAttribute("visibility").Get() == "invisible"


def test_original_obj_vertices_faces_and_colors_preserved(scene):
    stage, metadata = scene
    import xml.etree.ElementTree as ET
    xml = ET.parse(ROOT / "upstream/unitree_go2/go2.xml").getroot()
    assets = {Path(x.attrib["file"]).stem: x.attrib["file"] for x in xml.findall("asset/mesh")}
    colors = {x.attrib["name"]: [float(c) for c in x.attrib["rgba"].split()[:3]]
              for x in xml.findall("asset/material")}
    cache = {}
    for record in metadata["geometry_records"]:
        if not record["visual"]:
            continue
        attrs = record["source_attributes"]
        asset = attrs["mesh"]
        if asset not in cache:
            vertices, faces = [], []
            for line in (ROOT / "upstream/unitree_go2/assets" / assets[asset]).read_text().splitlines():
                if line.startswith("v "):
                    vertices.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    faces.append(len(line.split()) - 1)
            cache[asset] = np.asarray(vertices), faces
        vertices, faces = cache[asset]
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(record["path"]))
        np.testing.assert_allclose(mesh.GetPointsAttr().Get(), vertices, rtol=8e-8, atol=1e-9)
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == faces
        np.testing.assert_allclose(mesh.GetDisplayColorAttr().Get()[0], colors[attrs["material"]], atol=3e-8)
        assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert mesh.GetNormalsInterpolation() == "faceVarying"


def test_course_geometry_material_and_visual_only_marks(scene):
    stage, metadata = scene
    assert stage.GetPrimAtPath(metadata["environment_paths"]["floor"]).GetTypeName() in {"Plane", "PhysicsPlane"}
    visual_floor = stage.GetPrimAtPath("/World/Course/floor_visual")
    assert visual_floor.GetTypeName() == "Mesh"
    assert not visual_floor.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(visual_floor).ComputeVisibility() == "inherited"
    np.testing.assert_allclose(UsdGeom.Mesh(visual_floor).GetPointsAttr().Get(),
                               [[-4, -3, 0], [4, -3, 0], [4, 3, 0], [-4, 3, 0]])
    expected = {
        "rail_0": ([.48, 0, .0125], [.025, .6, .0125]),
        "rail_1": ([.86, 0, .0175], [.025, .6, .0175]),
        "low_gate": ([1.65, 0, .38], [.025, .575, .02]),
        "gate_post_negative": ([1.65, -.55, .18], [.025, .025, .18]),
        "gate_post_positive": ([1.65, .55, .18], [.025, .025, .18]),
    }
    for name, (position, half_extents) in expected.items():
        transform = _transform(stage.GetPrimAtPath(metadata["environment_paths"][name]))
        np.testing.assert_allclose(transform[:3, 3], position, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(transform[:3, :3], axis=0), half_extents, atol=3e-8)
    for prim in stage.Traverse():
        if prim.GetName() == "dock" or prim.GetName().startswith("path_mark_"):
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
    contact_material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/World/Looks/FootContact"))
    assert contact_material.GetDynamicFrictionAttr().Get() == pytest.approx(.8)
    for path in metadata["foot_collider_paths"].values():
        bound, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial("physics")
        assert str(bound.GetPath()) == "/World/Looks/FootContact"


def test_payload_friction_and_no_terrain_are_explicit():
    stage = Usd.Stage.CreateInMemory()
    metadata = build_scene(stage, ROOT, payload=2.0, friction=.4, terrain=False)
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(metadata["base_path"]))
    assert mass.GetMassAttr().Get() == pytest.approx(8.921)
    np.testing.assert_allclose(mass.GetCenterOfMassAttr().Get(), [.021112, 0, -.005366], atol=1e-9)
    np.testing.assert_allclose(mass.GetDiagonalInertiaAttr().Get(), [.107027, .0980771, .0244531], atol=6e-9)
    assert metadata["environment_paths"] == {"floor": "/World/Course/floor"}
    assert metadata["source_counts"]["environment_colliders"] == 1
    material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/World/Looks/FootContact"))
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(.4)
    assert material.GetDynamicFrictionAttr().Get() == pytest.approx(.4)
    with pytest.raises(ValueError, match="already exists"):
        build_scene(stage, ROOT)


@pytest.mark.parametrize("kwargs", [{"payload": -1}, {"friction": -1}, {"friction": math.nan}])
def test_invalid_physical_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        build_scene(Usd.Stage.CreateInMemory(), ROOT, **kwargs)
