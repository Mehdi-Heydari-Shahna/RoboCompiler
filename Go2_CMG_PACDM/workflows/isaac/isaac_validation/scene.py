"""Direct, auditable MJCF/CMG to USD translation for the pinned Go2 asset.

The physical tree has thirteen rigid bodies and twelve revolute joints.  The
six massless CMG coordinate-chart stages are deliberately not physical bodies.
All links are authored at q=0, with the base lifted to z=2 m.  The runtime must
initialize the free-base pose and twelve joint positions before simulation.

This module can author and inspect USD without Isaac Sim installed.  If the
PhysxSchema Python bindings are absent, it writes the documented schema tokens
and attributes directly; only an Isaac/PhysX runtime can validate their effect.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

try:
    from pxr import PhysxSchema
except ImportError:  # USD-only authoring/tests, never evidence of simulation.
    PhysxSchema = None

from go2.model import _defaults
from go2.task import HURDLES


LIMITATIONS = [
    "PhysX contacts are independent of MuJoCo: MJCF solref/solimp, condim, "
    "and rolling/torsional friction are not equivalent PhysX parameters.",
    "The original cylinder primitives and dimensions are retained; the PhysX "
    "release/settings determine whether cylinders use convex approximation.",
    "Non-foot robot contacts use Coulomb friction in PhysX, whereas the "
    "original non-foot MJCF geoms used condim=1; any non-foot contact is "
    "an invalid benchmark event regardless of its friction response.",
    "Joint damping and dry friction are applied explicitly by the runner; "
    "no joint drives, PhysX joint friction, or rigid-body damping are enabled.",
    "A payload is a point mass at the original base COM, with unchanged "
    "base inertia, matching the supplied MuJoCo benchmark definition.",
    "Authored USD mass properties and geometry are checked offline; PhysX "
    "runtime behavior requires an actual Isaac Sim execution.",
]


def _vec(text, fallback):
    return np.asarray(fallback if text is None else text.split(), dtype=float)


def _quat(value, double=False):
    q = np.asarray(value, dtype=float)
    q /= np.linalg.norm(q)
    cls, vec = (Gf.Quatd, Gf.Vec3d) if double else (Gf.Quatf, Gf.Vec3f)
    return cls(float(q[0]), vec(*map(float, q[1:])))


def _transform(attrs):
    transform = np.eye(4)
    q = _vec(attrs.get("quat"), [1, 0, 0, 0])
    transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = _vec(attrs.get("pos"), [0, 0, 0])
    return transform


def _set_pose(prim, position, quaternion=(1, 0, 0, 0), scale=None):
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, position)))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_quat(quaternion, double=True))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def _physx(prim, api_name, attributes):
    """Apply registered PhysX APIs, or preserve their schema on USD-only hosts."""
    if PhysxSchema is not None:
        schema = getattr(PhysxSchema, api_name, None)
        if schema is None:
            raise RuntimeError(f"Installed Isaac Sim has no required {api_name}")
        schema.Apply(prim)
    else:
        prim.AddAppliedSchema(api_name)
    for name, value_type, value in attributes:
        prim.CreateAttribute(name, value_type, custom=False).Set(value)


def _display_material(stage, path, rgba):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _physics_material(stage, path, friction):
    material = UsdShade.Material.Define(stage, path)
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(float(friction))
    physical.CreateDynamicFrictionAttr(float(friction))
    physical.CreateRestitutionAttr(0.0)
    _physx(material.GetPrim(), "PhysxMaterialAPI", [
        ("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, "max"),
        ("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token, "min"),
    ])
    return material


def _bind(prim, material, purpose=""):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)


def _collision(prim, material):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    _physx(prim, "PhysxCollisionAPI", [
        ("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, 0.001),
        ("physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0),
    ])
    _bind(prim, material, "physics")


@lru_cache(maxsize=32)
def _read_obj(path_string):
    """Read the packaged OBJ vertices/faces/normals, including OBJ negative IDs."""
    points, normals, counts, indices, face_normals = [], [], [], [], []
    all_normals = True
    with Path(path_string).open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                points.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "vn":
                normals.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "f":
                counts.append(len(parts) - 1)
                for corner in parts[1:]:
                    fields = corner.split("/")
                    vertex = int(fields[0])
                    indices.append(vertex - 1 if vertex > 0 else len(points) + vertex)
                    if len(fields) > 2 and fields[2]:
                        normal = int(fields[2])
                        face_normals.append(normals[normal - 1 if normal > 0 else len(normals) + normal])
                    else:
                        all_normals = False
    if not points or not counts or min(indices) < 0 or max(indices) >= len(points):
        raise ValueError(f"Invalid or empty OBJ asset: {path_string}")
    return (Vt.Vec3fArray(points), Vt.IntArray(counts), Vt.IntArray(indices),
            Vt.Vec3fArray(face_normals) if all_normals else None)


def _visual_mesh(stage, path, asset):
    mesh = UsdGeom.Mesh.Define(stage, path)
    points, counts, indices, normals = _read_obj(str(asset.resolve()))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    if normals is not None:
        mesh.CreateNormalsAttr(normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    values = np.asarray(points)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, values.min(axis=0))),
                           Gf.Vec3f(*map(float, values.max(axis=0)))])
    return mesh.GetPrim()


def _primitive(stage, path, kind, size, position, quaternion=(1, 0, 0, 0)):
    size = np.asarray(size, dtype=float)
    scale = None
    if kind == "box":
        geometry = UsdGeom.Cube.Define(stage, path)
        geometry.CreateSizeAttr(2.0)  # MJCF boxes store half-extents.
        scale = size
    elif kind == "sphere":
        geometry = UsdGeom.Sphere.Define(stage, path)
        geometry.CreateRadiusAttr(float(size[0]))
    elif kind == "cylinder":
        geometry = UsdGeom.Cylinder.Define(stage, path)
        geometry.CreateRadiusAttr(float(size[0]))
        geometry.CreateHeightAttr(float(2 * size[1]))
        geometry.CreateAxisAttr(UsdGeom.Tokens.z)
    else:
        raise ValueError(f"Unsupported source collider {kind!r}; refusing approximation")
    _set_pose(geometry.GetPrim(), position, quaternion, scale)
    return geometry.GetPrim()


def build_scene(stage, root: Path, friction=0.8, payload=0.0, terrain=True):
    """Build the original Go2 physical asset and benchmark course on ``stage``.

    Returns paths and source provenance; no physics scene, camera, lights, or
    simulation are created here.  Call only once on a fresh stage and initialize
    the articulation before the first physics step (q=0 violates knee limits).
    """
    root = Path(root).resolve()
    if not math.isfinite(friction) or friction < 0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(payload) or payload < 0:
        raise ValueError("payload must be a finite, nonnegative point mass")
    if stage.GetPrimAtPath("/World/Go2"):
        raise ValueError("/World/Go2 already exists; build on a fresh stage")
    xml_path = root / "upstream/unitree_go2/go2.xml"
    xml = ET.parse(xml_path).getroot()
    cmg = json.loads((root / "data/go2_cmg.json").read_text(encoding="utf-8"))
    xml_hash = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    if cmg["source"]["sha256"] != xml_hash:
        raise ValueError("Source MJCF hash does not match the packaged CMG")
    defaults = _defaults(xml)
    body_records = {x["id"]: x for x in cmg["bodies"] if x["mass_kg"] > 0}
    joint_records = {x["id"]: x for x in cmg["joints"] if x.get("actuated")}
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    if not stage.GetDefaultPrim():
        stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Go2")
    UsdGeom.Scope.Define(stage, "/World/Go2/joints")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Xform.Define(stage, "/World/Course")
    materials = {}
    rgba_by_name = {}
    for element in xml.findall("asset/material"):
        name = element.attrib["name"]
        rgba = _vec(element.get("rgba"), [0.5, 0.5, 0.5, 1.0])
        rgba_by_name[name] = rgba
        materials[name] = _display_material(stage, "/World/Looks/" + name, rgba)
    foot_material = _physics_material(stage, "/World/Looks/FootContact", friction)
    body_material = _physics_material(stage, "/World/Looks/BodyContact", 0.6)
    gate_material = _physics_material(stage, "/World/Looks/GateContact", 1.0)
    mesh_files = {m.get("name", Path(m.attrib["file"]).stem):
                  xml_path.parent / xml.find("compiler").get("meshdir", "") / m.attrib["file"]
                  for m in xml.findall("asset/mesh")}
    body_paths, joint_paths, foot_paths, geometry_records = {}, {}, {}, []

    def visit(element, parent_transform, inherited_class="main", base=False):
        name = element.attrib["name"]
        childclass = element.get("childclass", inherited_class)
        transform = np.eye(4) if base else parent_transform @ _transform(element.attrib)
        if base:
            transform[2, 3] = 2.0  # Keep initialization/warm-up clear of the floor.
        path = "/World/Go2/" + name
        body_paths[name] = path
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        _set_pose(body, transform[:3, 3], quat_xyzw[[3, 0, 1, 2]])
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        _physx(body, "PhysxRigidBodyAPI", [
            ("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, 0.0),
            ("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float, 0.0),
            ("physxRigidBody:enableGyroscopicForces", Sdf.ValueTypeNames.Bool, True),
        ])
        _physx(body, "PhysxContactReportAPI", [
            ("physxContactReport:threshold", Sdf.ValueTypeNames.Float, 0.0),
        ])
        record = body_records[name]
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(float(record["mass_kg"] + (payload if base else 0)))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*record["com_m"]))
        # Preserve the source principal axes rather than recomputing an arbitrary
        # eigenvector orientation.  Verify them against the independent CMG tensor.
        inertial = record["source_inertial_attributes"]
        principal = _vec(inertial.get("quat"), [1, 0, 0, 0])
        diagonal = _vec(inertial.get("diaginertia"), [])
        rotation = Rotation.from_quat(principal[[1, 2, 3, 0]]).as_matrix()
        if diagonal.shape != (3,) or not np.allclose(
                rotation @ np.diag(diagonal) @ rotation.T, record["inertia_kg_m2"], atol=1e-12):
            raise ValueError(f"CMG/source inertia mismatch at {name}")
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
        mass.CreatePrincipalAxesAttr(_quat(principal))
        UsdGeom.Scope.Define(stage, path + "/visuals")
        UsdGeom.Scope.Define(stage, path + "/collisions")
        for index, geom in enumerate(element.findall("geom")):
            resolved = dict(defaults[geom.get("class", childclass)].get("geom", {}))
            resolved.update(geom.attrib)
            visual = resolved.get("contype", "1") == "0" and resolved.get("conaffinity", "1") == "0"
            geom_name = geom.get("name", f"geom_{index:02d}")
            geom_path = path + ("/visuals/" if visual else "/collisions/") + geom_name
            position = _vec(resolved.get("pos"), [0, 0, 0])
            quaternion = _vec(resolved.get("quat"), [1, 0, 0, 0])
            kind = resolved.get("type", "sphere")
            if visual:
                if kind != "mesh":
                    raise ValueError("Pinned Go2 visual geometry is expected to be OBJ meshes")
                prim = _visual_mesh(stage, geom_path, mesh_files[resolved["mesh"]])
                _set_pose(prim, position, quaternion)
                _bind(prim, materials[resolved["material"]])
                UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgba_by_name[resolved["material"]][:3])])
            else:
                prim = _primitive(stage, geom_path, kind, _vec(resolved.get("size"), []), position, quaternion)
                _collision(prim, foot_material if geom.get("class") == "foot" else body_material)
                # Collision shapes are not rendered over the supplied robot mesh.
                UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                if geom.get("class") == "foot":
                    foot_paths[geom_name] = geom_path
            geometry_records.append(dict(body=name, source_geom_index=index, path=geom_path,
                                         visual=visual, kind=kind, source_attributes=resolved))
        for child in element.findall("body"):
            visit(child, transform, childclass)

    roots = xml.findall("worldbody/body")
    if len(roots) != 1 or roots[0].get("name") != "base" or roots[0].find("freejoint") is None:
        raise ValueError("Expected the pinned Go2 floating-base physical tree")
    visit(roots[0], np.eye(4), base=True)
    base_prim = stage.GetPrimAtPath(body_paths["base"])
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    _physx(base_prim, "PhysxArticulationAPI", [
        ("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, True),
        ("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32),
        ("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 16),
        ("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float, 0.0),
        ("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float, 0.0),
    ])
    for name, record in joint_records.items():
        path = "/World/Go2/joints/" + name
        joint_paths[name] = path
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([body_paths[record["base_body"]]])
        joint.CreateBody1Rel().SetTargets([body_paths[record["follower_body"]]])
        axis = np.asarray(record["axis"])
        axis_index = int(np.argmax(np.abs(axis)))
        if not np.allclose(axis, np.eye(3)[axis_index], atol=1e-12):
            raise ValueError(f"Noncanonical joint axis at {name}; no silent axis approximation")
        joint.CreateAxisAttr("XYZ"[axis_index])
        for side, matrix in enumerate((record["T_BJ"], record["T_FJ"])):
            matrix = np.asarray(matrix)
            quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
            getattr(joint, f"CreateLocalPos{side}Attr")(Gf.Vec3f(*matrix[:3, 3]))
            getattr(joint, f"CreateLocalRot{side}Attr")(_quat(quat))
        joint.CreateLowerLimitAttr(math.degrees(record["limits"]["lower"]))
        joint.CreateUpperLimitAttr(math.degrees(record["limits"]["upper"]))
        joint.CreateCollisionEnabledAttr(False)  # Same parent/child exclusions.
        _physx(joint.GetPrim(), "PhysxJointAPI", [
            ("physxJoint:armature", Sdf.ValueTypeNames.Float, float(record["armature"])),
            ("physxJoint:jointFriction", Sdf.ValueTypeNames.Float, 0.0),
        ])

    environment = {}
    # Plane wrappers moved between USD distributions. Use an installed wrapper
    # when available; older Omniverse releases also recognize the Plane token.
    # Keep rendering separate, as official physicsUtils.add_ground_plane does:
    # an infinite collision plane need not have a renderable Hydra primitive.
    plane_schema = getattr(UsdPhysics, "Plane", None) or getattr(UsdGeom, "Plane", None)
    floor = (plane_schema.Define(stage, "/World/Course/floor").GetPrim()
             if plane_schema is not None else stage.DefinePrim("/World/Course/floor", "Plane"))
    floor.CreateAttribute("axis", Sdf.ValueTypeNames.Token, custom=False).Set("Z")
    floor.CreateAttribute("width", Sdf.ValueTypeNames.Double, custom=False).Set(8.0)
    floor.CreateAttribute("length", Sdf.ValueTypeNames.Double, custom=False).Set(6.0)
    _set_pose(floor, [0, 0, 0])
    _collision(floor, foot_material)
    floor_look = _display_material(stage, "/World/Looks/Floor", [0.10, 0.14, 0.20, 1])
    UsdGeom.Imageable(floor).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    floor_visual = UsdGeom.Mesh.Define(stage, "/World/Course/floor_visual")
    floor_visual.CreatePointsAttr([(-4, -3, 0), (4, -3, 0), (4, 3, 0), (-4, 3, 0)])
    floor_visual.CreateFaceVertexCountsAttr([4])
    floor_visual.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor_visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    floor_visual.CreateDoubleSidedAttr(True)
    floor_visual.CreateNormalsAttr([(0, 0, 1)])
    floor_visual.SetNormalsInterpolation(UsdGeom.Tokens.constant)
    floor_visual.CreateExtentAttr([(-4, -3, 0), (4, 3, 0)])
    _bind(floor_visual.GetPrim(), floor_look)
    environment["floor"] = str(floor.GetPath())

    def course_box(name, size, position, rgba, physical=True, gate=False, yaw=0):
        path = "/World/Course/" + name
        prim = _primitive(stage, path, "box", size, position, [math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
        _bind(prim, _display_material(stage, "/World/Looks/" + name, rgba))
        if physical:
            _collision(prim, gate_material if gate else foot_material)
            environment[name] = path

    if terrain:
        for index, (x, height) in enumerate(HURDLES):
            course_box(f"rail_{index}", [.025, .6, height/2], [x, 0, height/2], [.95, .56, .10, 1])
        course_box("low_gate", [.025, .575, .02], [1.65, 0, .38], [.20, .62, .70, 1], gate=True)
        for suffix, y in (("negative", -.55), ("positive", .55)):
            course_box("gate_post_" + suffix, [.025, .025, .18], [1.65, y, .18], [.20, .62, .70, 1], gate=True)
    course_box("dock", [.34, .28, .0003], [2.20, .30, .0003], [.08, .35, .38, 1], physical=False, yaw=.35)
    for index, x in enumerate(np.arange(-.2, 2.6, .15)):
        course_box(f"path_mark_{index:02d}", [.04, .007, .0005], [x, -.64, .0005], [.25, .6, .7, 1], physical=False)
    counts = dict(rigid_bodies=len(body_paths), revolute_joints=len(joint_paths),
                  robot_colliders=sum(not x["visual"] for x in geometry_records),
                  visual_meshes=sum(x["visual"] for x in geometry_records),
                  unique_obj_meshes=len(mesh_files), foot_colliders=len(foot_paths),
                  environment_colliders=len(environment))
    if counts["rigid_bodies"] != 13 or counts["revolute_joints"] != 12 or counts["foot_colliders"] != 4:
        raise ValueError(f"Unexpected pinned Go2 source topology: {counts}")
    return dict(robot_root="/World/Go2", base_path=body_paths["base"], body_paths=body_paths,
                joint_paths=joint_paths, foot_collider_paths=foot_paths, environment_paths=environment,
                source_counts=counts, geometry_records=geometry_records, source_sha256=xml_hash,
                source_commit=cmg["source"]["commit"], total_mass_kg=cmg["total_mass_kg"] + payload,
                point_payload_kg=payload, friction=friction, terrain=bool(terrain),
                initial_base_position_m=[0.0, 0.0, 2.0], initial_joint_positions_rad=[0.0] * 12,
                physx_schema_bindings_available=PhysxSchema is not None,
                limitations=list(LIMITATIONS))
