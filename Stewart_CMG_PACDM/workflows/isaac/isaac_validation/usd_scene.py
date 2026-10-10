"""Compile the six-UPS CMG into an independent USD/PhysX plant.

The platform pose chart is mathematical bookkeeping, not six physical joints.
It becomes one free rigid body; its fixed payload is combined by the parallel
axis theorem.  Each physical UPS leg is three ordinary joints and its point
cut is a spherical joint.  There is deliberately no ArticulationRootAPI,
joint drive, collider, trajectory animation, or simulation state setter here.

This module needs OpenUSD, NumPy and SciPy, but does not import Isaac Sim.
PhysX API schemas are applied by name when the optional NVIDIA schema plugin
is unavailable, allowing exactly the same stage to be audited off-line.
"""
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from vendor.pacdm_original import PointGraph
from .inertial_audit import principal_properties


ROOT = "/World/Stewart"
SCENE = "/World/PhysicsScene"


def _vec(values):
    return Gf.Vec3f(*[float(v) for v in values])


def _quat(matrix, double=False):
    x, y, z, w = Rotation.from_matrix(np.asarray(matrix)).as_quat()
    return (Gf.Quatd if double else Gf.Quatf)(float(w), float(x), float(y), float(z))


def _rotation(quaternion):
    return Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()


def _transform(prim, transform):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*[float(v) for v in transform[:3, 3]]))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_quat(transform[:3, :3], True))


def _physx(prim, schema, values):
    """Author real PhysX schema attributes, also usable with stock usd-core."""
    try:
        from pxr import PhysxSchema
    except ImportError:
        prim.AddAppliedSchema(schema)
    else:
        getattr(PhysxSchema, schema).Apply(prim)
    for name, type_name, value in values:
        prim.CreateAttribute(name, type_name, custom=False).Set(value)


def _mass_properties(cmg, poses):
    """Return physical body properties; preserve the fixed payload exactly."""
    source = {body["id"]: body for body in cmg["bodies"]}
    if cmg.get("schema") != "cmg.stewart.point-closures/1.0":
        raise ValueError("This compiler accepts the Stewart CMG schema only")
    mounts = [j for j in cmg["joints"] if j["id"] == "payload_mount"]
    if len(mounts) != 1 or mounts[0]["type"] != "fixed":
        raise ValueError("The payload must be fixed to the platform")
    if (mounts[0]["base_body"], mounts[0]["follower_body"]) != ("platform", "payload"):
        raise ValueError("Unexpected fixed payload topology")
    physical = {"platform", "payload"} | {
        f"leg_{i}_{part}" for i in range(6) for part in ("yoke", "barrel", "rod")}
    for name, body in source.items():
        if name not in physical and (body["mass_kg"] != 0 or
                                    np.any(np.asarray(body["inertia_kg_m2"]) != 0)):
            raise ValueError(f"Cannot discard physical inertia on chart frame {name}")
    properties = {}
    for name in sorted(physical - {"payload"}):
        body = source[name]
        properties[name] = dict(mass_kg=float(body["mass_kg"]),
                                com_m=np.asarray(body["com_m"], dtype=float),
                                inertia_kg_m2=np.asarray(body["inertia_kg_m2"], dtype=float))
    pieces = []
    for name in ("platform", "payload"):
        body = source[name]
        local = np.linalg.inv(poses["platform"]) @ poses[name]
        pieces.append((float(body["mass_kg"]),
                       local[:3, :3] @ np.asarray(body["com_m"]) + local[:3, 3],
                       local[:3, :3] @ np.asarray(body["inertia_kg_m2"]) @ local[:3, :3].T))
    mass = sum(m for m, _, _ in pieces)
    com = sum(m * c for m, c, _ in pieces) / mass
    inertia = sum(I + m * (np.dot(c - com, c - com) * np.eye(3) -
                          np.outer(c - com, c - com)) for m, c, I in pieces)
    properties["platform"] = dict(mass_kg=mass, com_m=com, inertia_kg_m2=inertia)
    for name, props in properties.items():
        I = props["inertia_kg_m2"]
        if not np.allclose(I, I.T, atol=1e-12) or not np.all(np.isfinite(I)):
            raise ValueError(f"Invalid inertia for {name}")
        if props["mass_kg"] <= 0 or np.min(np.linalg.eigvalsh(I)) <= 0:
            raise ValueError(f"Non-positive physical mass/inertia for {name}")
    return properties


def _rigid_body(stage, name, pose, properties):
    path = f"{ROOT}/Bodies/{name}"
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    _transform(prim, pose)
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(False)
    body.CreateVelocityAttr(Gf.Vec3f(0))
    body.CreateAngularVelocityAttr(Gf.Vec3f(0))
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(properties["mass_kg"])
    mass.CreateCenterOfMassAttr(_vec(properties["com_m"]))
    diagonal, axes = principal_properties(properties["inertia_kg_m2"])
    mass.CreateDiagonalInertiaAttr(_vec(diagonal))
    mass.CreatePrincipalAxesAttr(_quat(axes))
    _physx(prim, "PhysxRigidBodyAPI", [
        ("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, 0.0),
        ("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float, 0.0),
        ("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float, 0.0),
        ("physxRigidBody:stabilizationThreshold", Sdf.ValueTypeNames.Float, 0.0),
        ("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 128),
        ("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 32),
        ("physxRigidBody:enableGyroscopicForces", Sdf.ValueTypeNames.Bool, True),
        ("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, False),
    ])
    return path


def _joint_frames(joint, base_path, follower_path, T_BJ, T_FJ):
    if base_path is not None:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
    else:
        joint.CreateBody0Rel().SetTargets([])  # Empty body0 means the static world.
    joint.CreateBody1Rel().SetTargets([Sdf.Path(follower_path)])
    joint.CreateLocalPos0Attr(_vec(T_BJ[:3, 3]))
    joint.CreateLocalRot0Attr(_quat(T_BJ[:3, :3]))
    joint.CreateLocalPos1Attr(_vec(T_FJ[:3, 3]))
    joint.CreateLocalRot1Attr(_quat(T_FJ[:3, :3]))
    joint.CreateJointEnabledAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    # This is a regular constraint, even if a caller later adds an articulation.
    joint.CreateExcludeFromArticulationAttr(True)
    _physx(joint.GetPrim(), "PhysxJointAPI", [
        ("physxJoint:jointFriction", Sdf.ValueTypeNames.Float, 0.0),
        ("physxJoint:armature", Sdf.ValueTypeNames.Float, 0.0),
        ("physxJoint:enableProjection", Sdf.ValueTypeNames.Bool, False),
    ])


def _material(stage, name, color, metallic=0., roughness=.4, emission=None):
    material = UsdShade.Material.Define(stage, f"{ROOT}/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_vec(color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if emission is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(_vec(emission))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _visual(stage, path, kind, material, position=(0, 0, 0), scale=None,
            radius=None, height=None):
    shape = getattr(UsdGeom, kind).Define(stage, path)
    if radius is not None:
        shape.CreateRadiusAttr(float(radius))
    if height is not None:
        shape.CreateHeightAttr(float(height))
        shape.CreateAxisAttr("Z")
    if kind == "Cube":
        shape.CreateSizeAttr(1.)
    xform = UsdGeom.Xformable(shape)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    if scale is not None:
        xform.AddScaleOp().Set(_vec(scale))
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    shape.CreateDisplayColorAttr([material.GetPrim().GetStage().GetPrimAtPath(
        material.GetPath().AppendChild("Shader")).GetAttribute("inputs:diffuseColor").Get()])
    # Visuals have no CollisionAPI and no independent mass properties.
    return shape


def _visuals(stage, cmg, bodies, poses):
    colors = {
        "frame": ((.08, .15, .20), .7, .3), "teal": ((.03, .34, .42), .6, .3),
        "steel": ((.7, .78, .82), .9, .18), "gold": ((.85, .52, .10), .7, .3),
        "white": ((.82, .88, .92), .25, .3), "floor": ((.025, .045, .075), .0, .5),
    }
    mats = {n: _material(stage, n, *v) for n, v in colors.items()}
    mats["cyan"] = _material(stage, "cyan", (.03, .75, .85), .1, .3, (.03, .55, .65))
    base = ROOT + "/Decor"
    _visual(stage, base + "/Floor", "Cube", mats["floor"], (0, 0, -.14), scale=(5, 5, .04))
    _visual(stage, base + "/Base", "Cylinder", mats["frame"], (0, 0, -.065), radius=.64, height=.09)
    _visual(stage, base + "/BaseInset", "Cylinder", mats["teal"], (0, 0, -.017), radius=.59, height=.004)
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, 36, endpoint=False)):
        _visual(stage, base + f"/Ring_{i}", "Sphere", mats["cyan"],
                (.607 * math.cos(angle), .607 * math.sin(angle), -.016), radius=.006)
    platform = bodies["platform"]
    _visual(stage, platform + "/Plate", "Cylinder", mats["frame"], radius=.39, height=.05)
    _visual(stage, platform + "/PlateInset", "Cylinder", mats["teal"], (0, 0, .028), radius=.355, height=.006)
    for i, anchor in enumerate(cmg["geometry"]["platform_anchors_m"]):
        _visual(stage, platform + f"/Anchor_{i}", "Sphere", mats["gold"], anchor, radius=.025)
    payload_frame = UsdGeom.Xform.Define(stage, platform + "/PayloadVisual")
    _transform(payload_frame.GetPrim(), np.linalg.inv(poses["platform"]) @ poses["payload"])
    payload = str(payload_frame.GetPath())
    _visual(stage, payload + "/Housing", "Cube", mats["white"], (0, 0, .10), scale=(.23, .21, .20))
    _visual(stage, payload + "/Stripe", "Cube", mats["frame"], (0, 0, .10), scale=(.234, .05, .206))
    _visual(stage, payload + "/Probe", "Cylinder", mats["frame"], (0, 0, .25), radius=.047, height=.13)
    _visual(stage, payload + "/Tip", "Sphere", mats["cyan"], (0, 0, .328), radius=.018)
    for i in range(6):
        _visual(stage, bodies[f"leg_{i}_yoke"] + "/Housing", "Sphere", mats["steel"], radius=.037)
        barrel = bodies[f"leg_{i}_barrel"]
        _visual(stage, barrel + "/Tube", "Cylinder", mats["teal"], (0, 0, .245), radius=.025, height=.49)
        _visual(stage, barrel + "/Collar", "Cylinder", mats["gold"], (0, 0, .485), radius=.029, height=.04)
        rod = bodies[f"leg_{i}_rod"]
        _visual(stage, rod + "/Shaft", "Cylinder", mats["steel"], (0, 0, -.225), radius=.013, height=.45)
        _visual(stage, rod + "/Head", "Sphere", mats["gold"], radius=.024)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Fill")
    dome.CreateIntensityAttr(450.)
    dome.CreateColorAttr(_vec((.68, .79, 1.)))
    light = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    light.CreateIntensityAttr(2500.)
    light.CreateAngleAttr(12.)
    light.CreateColorAttr(_vec((1., .90, .78)))
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(_vec((-30., -25., -35.)))


def build_scene(cmg, q, output_path=None, stage=None, dt=.002):
    """Build an initial, unforced plant; return paths and a serializable audit.

    ``q`` contains the 24 CMG coordinates (an augmented PACDM vector is also
    accepted). It initializes transforms once; normal PhysX stepping alone
    updates the scene thereafter. ``stage`` must not already contain the
    Stewart root. An existing /World/PhysicsScene may be configured in place.
    """
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(round(1 / dt), 1 / dt):
        raise ValueError("dt must be positive and have an integer reciprocal in Hz")
    q = np.asarray(q, dtype=float)
    nt = len(cmg["coordinate_ids"])
    if q.ndim != 1 or q.size not in (nt, nt + 3 * len(cmg["closures"])) or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite CMG coordinate vector")
    q = q[:nt]
    graph = PointGraph(cmg, q)
    if np.any(q < graph.lower[:nt]) or np.any(q > graph.upper[:nt]):
        raise ValueError("Initial coordinates exceed the CMG joint limits")
    poses, _ = graph.poses(graph.augment(q))
    properties = _mass_properties(cmg, poses)
    residual = max(np.linalg.norm(poses[c["body1"]][:3, :3] @ c["point1_m"] +
                                  poses[c["body1"]][:3, 3] -
                                  poses[c["body2"]][:3, :3] @ c["point2_m"] -
                                  poses[c["body2"]][:3, 3]) for c in cmg["closures"])
    if residual > 1e-7:
        raise ValueError(f"Initial state is not physically closed: {residual:.6g} m")
    if stage is None:
        stage = Usd.Stage.CreateInMemory()
    if stage.GetPrimAtPath(ROOT):
        raise ValueError(f"Stage already contains {ROOT}; use a fresh stage")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    stage.SetTimeCodesPerSecond(1 / dt)
    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Scope.Define(stage, ROOT + "/Bodies")
    UsdGeom.Scope.Define(stage, ROOT + "/Joints")
    gravity = np.asarray(cmg["gravity_m_s2"], dtype=float)
    magnitude = float(np.linalg.norm(gravity))
    scene = UsdPhysics.Scene.Define(stage, SCENE)
    scene.CreateGravityDirectionAttr(_vec(gravity / magnitude if magnitude else (0, 0, -1)))
    scene.CreateGravityMagnitudeAttr(magnitude)
    _physx(scene.GetPrim(), "PhysxSceneAPI", [
        ("physxScene:solverType", Sdf.ValueTypeNames.Token, "TGS"),
        ("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool, False),
        ("physxScene:broadphaseType", Sdf.ValueTypeNames.Token, "MBP"),
        ("physxScene:enableStabilization", Sdf.ValueTypeNames.Bool, False),
        ("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt, round(1 / dt)),
    ])
    body_paths = {name: _rigid_body(stage, name, poses[name], props)
                  for name, props in properties.items()}
    joint_paths = {}
    for record in cmg["joints"]:
        if not record["id"].startswith("leg_"):
            continue
        parent, child = record["base_body"], record["follower_body"]
        path = ROOT + "/Joints/" + record["id"]
        klass = {"revolute": UsdPhysics.RevoluteJoint,
                 "prismatic": UsdPhysics.PrismaticJoint}.get(record["type"])
        if klass is None:
            raise ValueError(f"Unsupported leg joint {record['id']}")
        joint = klass.Define(stage, path)
        axis = np.asarray(record["axis"])
        matching = [i for i in range(3) if np.array_equal(axis, np.eye(3)[i])]
        if len(matching) != 1:
            raise ValueError("This Stewart compiler requires positive cardinal joint axes")
        joint.CreateAxisAttr("XYZ"[matching[0]])
        factor = 180 / np.pi if record["type"] == "revolute" else 1.
        joint.CreateLowerLimitAttr(float(record["limits"]["lower"] * factor))
        joint.CreateUpperLimitAttr(float(record["limits"]["upper"] * factor))
        _joint_frames(joint, None if parent == cmg["root_body"] else body_paths[parent],
                      body_paths[child], np.asarray(record["T_BJ"]), np.asarray(record["T_FJ"]))
        joint_paths[record["id"]] = path
    for cut in cmg["closures"]:
        path = ROOT + "/Joints/" + cut["id"]
        joint = UsdPhysics.SphericalJoint.Define(stage, path)
        frame0, frame1 = np.eye(4), np.eye(4)
        frame0[:3, 3], frame1[:3, 3] = cut["point1_m"], cut["point2_m"]
        _joint_frames(joint, body_paths[cut["body1"]], body_paths[cut["body2"]], frame0, frame1)
        # Negative cone angles disable spherical limits: this is a point cut.
        joint.CreateConeAngle0LimitAttr(-1.)
        joint.CreateConeAngle1LimitAttr(-1.)
        joint_paths[cut["id"]] = path
    if len(body_paths) != 19 or len(joint_paths) != 24:
        raise ValueError("Expected 19 physical bodies and 24 physical joints")
    _visuals(stage, cmg, body_paths, poses)
    audit = audit_scene(cmg, q, stage, body_paths, joint_paths)
    result = dict(stage=stage, body_paths=body_paths, joint_paths=joint_paths,
                  platform_path=body_paths["platform"], scene_path=SCENE, audit=audit)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not stage.GetRootLayer().Export(str(output_path)):
            raise OSError(f"Unable to export USD stage to {output_path}")
    return result


def audit_scene(cmg, q, stage, body_paths, joint_paths):
    """Read back authored USD properties and fail on a physical mismatch.

    This is a model compilation audit, not evidence of an Isaac Sim run.
    Float tolerances account for USD Physics's float32 mass/joint attributes.
    """
    q = np.asarray(q)[:len(cmg["coordinate_ids"])]
    graph = PointGraph(cmg, q)
    poses, _ = graph.poses(graph.augment(q))
    expected = _mass_properties(cmg, poses)
    body_audit = {}
    cache = UsdGeom.XformCache()
    matrices = {}
    for name, path in body_paths.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"{name} is not a rigid body")
        api = UsdPhysics.MassAPI(prim)
        mass = float(api.GetMassAttr().Get())
        com = np.asarray(api.GetCenterOfMassAttr().Get())
        axes = _rotation(api.GetPrincipalAxesAttr().Get())
        inertia = axes @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ axes.T
        matrices[name] = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        props = expected[name]
        errors = dict(mass_kg=abs(mass - props["mass_kg"]),
                      com_m=float(np.max(abs(com - props["com_m"]))),
                      inertia_kg_m2=float(np.max(abs(inertia - props["inertia_kg_m2"]))),
                      transform=float(np.max(abs(matrices[name] - poses[name]))))
        if not (np.isclose(mass, props["mass_kg"], rtol=1e-6) and
                np.allclose(com, props["com_m"], atol=2e-8) and
                np.allclose(inertia, props["inertia_kg_m2"], rtol=1e-6, atol=2e-8) and
                errors["transform"] < 1e-10):
            raise ValueError(f"Physical property/initial-pose mismatch for {name}: {errors}")
        body_audit[name] = dict(mass_kg=mass, com_m=com.tolist(), inertia_kg_m2=inertia.tolist(),
                                initial_transform=matrices[name].tolist(), errors=errors)
    closure_errors = {}
    for cut in cmg["closures"]:
        joint = UsdPhysics.SphericalJoint(stage.GetPrimAtPath(joint_paths[cut["id"]]))
        a, b = matrices[cut["body1"]], matrices[cut["body2"]]
        pa = a[:3, :3] @ joint.GetLocalPos0Attr().Get() + a[:3, 3]
        pb = b[:3, :3] @ joint.GetLocalPos1Attr().Get() + b[:3, 3]
        closure_errors[cut["id"]] = float(np.linalg.norm(pa - pb))
    forbidden = []
    counts = dict(rigid_bodies=0, revolute=0, prismatic=0, spherical=0)
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(Sdf.Path(ROOT)):
            continue
        counts["rigid_bodies"] += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        for label, klass in (("revolute", UsdPhysics.RevoluteJoint),
                             ("prismatic", UsdPhysics.PrismaticJoint),
                             ("spherical", UsdPhysics.SphericalJoint)):
            counts[label] += int(prim.IsA(klass))
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(UsdPhysics.CollisionAPI) or
                any(s.startswith("PhysicsDriveAPI") for s in prim.GetAppliedSchemas())):
            forbidden.append(str(prim.GetPath()))
    if forbidden or counts != dict(rigid_bodies=19, revolute=12, prismatic=6, spherical=6):
        raise ValueError(f"Unexpected plant topology: {counts}, forbidden prims={forbidden}")
    if max(closure_errors.values()) > 5e-8:
        raise ValueError(f"USD initial closure mismatch: {closure_errors}")
    source_mass = float(sum(b["mass_kg"] for b in cmg["bodies"]))
    usd_mass = sum(b["mass_kg"] for b in body_audit.values())
    if not np.isclose(source_mass, usd_mass, rtol=1e-6):
        raise ValueError("Total rigid-body mass does not match CMG")
    return dict(status="PASS", evidence="OpenUSD compilation audit; no PhysX execution implied",
                counts=counts, source_total_mass_kg=source_mass, usd_total_mass_kg=usd_mass,
                payload_aggregation="Exact mass, COM and inertia via parallel-axis theorem",
                body_properties=body_audit, initial_closure_error_m=closure_errors,
                expected_body_properties={n: dict(mass_kg=p["mass_kg"],
                                                   com_m=p["com_m"].tolist(),
                                                   inertia_kg_m2=p["inertia_kg_m2"].tolist())
                                          for n, p in expected.items()},
                maximum_initial_closure_error_m=max(closure_errors.values()),
                cmg_to_rigid_body={**{n: n for n in body_paths}, "payload": "platform"},
                gravity_m_s2=list(cmg["gravity_m_s2"]), physical_drives=0,
                collision_shapes=0, articulation_roots=0, initial_velocities="zero")
