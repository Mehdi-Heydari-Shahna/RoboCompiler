#!/usr/bin/env python3
"""Native PhysX runner; execute with Isaac Sim's python.bat/python.sh.

This is a measurement runner, not a certificate of backend equivalence. It never
steps MuJoCo and never writes robot poses/velocities after physics initialization.
The external controller bridge owns PACDM and the original hydraulic dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import socket
import shutil
import sys
import time
import traceback

import numpy as np

from measurement_store import CheckpointStore, atomic_json
from rolling_resistance import GrainRollingResistance


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def json_write(path, data):
    atomic_json(path, data)


def rotations(q):
    """Batch rotation matrices, scalar-first unit quaternions."""
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=-1)
    if np.any(norms < 1e-12) or not np.all(np.isfinite(q)):
        raise ValueError("Invalid native quaternion")
    q = q / norms[..., None]
    w, x, y, z = np.moveaxis(q, -1, 0)
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, :] = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)), -1)
    r[..., 1, :] = np.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), -1)
    r[..., 2, :] = np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
    return r


def rotate(r, v):
    return np.einsum("...ij,...j->...i", r, v)


class Bridge:
    """JSON lines on loopback only; arbitrary Python objects are never decoded."""
    def __init__(self, port, timeout):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.file = self.sock.makefile("rwb")

    def request(self, request):
        self.file.write(json.dumps(request, allow_nan=False, separators=(",", ":")).encode() + b"\n")
        self.file.flush()
        line = self.file.readline(16 * 1024 * 1024 + 1)
        if not line or not line.endswith(b"\n") or len(line) > 16 * 1024 * 1024:
            raise RuntimeError("Controller bridge disconnected or response exceeds 16 MB")
        result = json.loads(line)
        if not isinstance(result, dict):
            raise RuntimeError("Controller bridge returned a non-object")
        if result.get("error") or result.get("ok") is False:
            raise RuntimeError("Controller bridge: " + str(result.get("error", result)))
        payload = result.get("result", result)
        if not isinstance(payload, dict):
            raise RuntimeError("Controller bridge result is not an object")
        return payload

    def close(self):
        try:
            self.file.close()
        finally:
            self.sock.close()


class NativeMeasurements:
    def __init__(self, manifest):
        self.m = manifest
        self.bodies = manifest["bodies"]
        self.index = {b["id"]: k for k, b in enumerate(self.bodies)}
        self.com_local = np.array([b["com_local"] for b in self.bodies])
        self.mass = np.array([b["mass"] for b in self.bodies])
        ri = rotations([b["inertia_quat_wxyz"] for b in self.bodies])
        diag = np.array([b["inertia_diagonal"] for b in self.bodies])
        self.inertia = np.einsum("nik,nk,njk->nij", ri, diag, ri)
        self.prev_angles = {}
        self.unwrapped = {}

    def kinematics(self, transforms, velocities):
        p = transforms[:, :3].astype(float, copy=True)
        q = transforms[:, [6, 3, 4, 5]].astype(float, copy=True)
        r = rotations(q)
        offset = rotate(r, self.com_local)
        vc = velocities[:, :3].astype(float, copy=True)
        w = velocities[:, 3:6].astype(float, copy=True)
        v = vc - np.cross(w, offset)
        for a in (p, q, vc, w):
            if not np.all(np.isfinite(a)):
                raise RuntimeError("Nonfinite PhysX state")
        return p, q, v, w, p + offset, vc, r

    def energy(self, state):
        p, q, v, w, com, vc, r = state
        wb = rotate(np.swapaxes(r, -1, -2), w)
        kinetic = .5*np.sum(self.mass*np.sum(vc*vc, axis=1))
        kinetic += .5*np.einsum("ni,nij,nj->", wb, self.inertia, wb)
        potential = -np.sum(self.mass*(com @ np.asarray(self.m["gravity"])))
        return float(kinetic), float(potential)

    def closure_gaps(self, state):
        p, _, _, _, _, _, r = state
        def point(bid, local):
            if bid == 0:
                return np.asarray(local)
            k = self.index[bid]
            return p[k] + r[k] @ np.asarray(local)
        return np.array([np.linalg.norm(point(c["body0_id"], c["pos0"])-
                        point(c["body1_id"], c["pos1"])) for c in self.m["closures"]])

    def joint_coordinates(self, state):
        p, _, _, _, _, _, r = state
        result = {}
        for j in self.m["joints"]:
            if j["type"] not in ("hinge", "slide"):
                continue
            i = self.index[j["body_id"]]
            par = self.index.get(j["parent_id"])
            rp = np.eye(3) if par is None else r[par]
            pp = np.zeros(3) if par is None else p[par]
            a = rp @ rotations(j["frame_quat0_wxyz"])
            b = r[i] @ rotations(j["frame_quat1_wxyz"])
            rel = a.T @ b
            if j["type"] == "hinge":
                # Exported USD hinge axis is X. Each per-step increment is
                # unwrapped; therefore fast explicit rotors can make full turns.
                angle = math.atan2(rel[2, 1]-rel[1, 2], rel[1, 1]+rel[2, 2])
                prev = self.prev_angles.get(j["id"], angle)
                change = (angle-prev+math.pi)%(2*math.pi)-math.pi
                self.unwrapped[j["id"]] = self.unwrapped.get(j["id"], angle)+change
                self.prev_angles[j["id"]] = angle
                result[j["id"]] = self.unwrapped[j["id"]] + j["q_initial"]
            else:
                c0 = pp + rp @ np.asarray(j["frame_pos0"])
                c1 = p[i] + r[i] @ np.asarray(j["frame_pos1"])
                result[j["id"]] = float(np.dot(a[:, 0], c1-c0)) + j["q_initial"]
        return result


def schema_check(stage, manifest, UsdPhysics, PhysxSchema):
    errors = []
    if not hasattr(PhysxSchema, "PhysxMimicJointAPI"):
        errors.append("Installed PhysxSchema has no PhysxMimicJointAPI")
    for b in manifest["bodies"]:
        prim = stage.GetPrimAtPath(b["path"])
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            errors.append("Missing rigid body: " + b["path"])
        elif UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
            errors.append("Kinematic body is forbidden: " + b["path"])
    body_paths = {b["id"]:b["path"] for b in manifest["bodies"]}
    joint_classes = {"hinge":UsdPhysics.RevoluteJoint, "slide":UsdPhysics.PrismaticJoint, "fixed":UsdPhysics.FixedJoint}
    for j in manifest["joints"]:
        prim = stage.GetPrimAtPath(j["path"])
        cls = joint_classes[j["type"]]
        if not prim or not prim.IsA(cls):
            errors.append("Missing native joint: " + j["path"])
            continue
        joint = cls(prim)
        parent = joint.GetBody0Rel().GetTargets()
        child = joint.GetBody1Rel().GetTargets()
        expected_parent = [] if j["parent_id"] == 0 else [body_paths[j["parent_id"]]]
        if list(map(str, parent)) != expected_parent or list(map(str, child)) != [body_paths[j["body_id"]]]:
            errors.append("Native joint body relation differs: " + j["path"])
        if joint.GetJointEnabledAttr().Get() is False:
            errors.append("Disabled source joint: " + j["path"])
        if j["type"] in ("hinge", "slide"):
            if str(joint.GetAxisAttr().Get()) != "X":
                errors.append("Native joint axis was not exported as X: " + j["path"])
            if j.get("limits") is not None:
                limits = np.asarray(j["limits"])-j["q_initial"]
                if j["type"] == "hinge":
                    limits = np.rad2deg(limits)
                actual = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
                if not np.allclose(actual, limits, rtol=2e-6, atol=2e-6):
                    errors.append("Native joint limits differ: " + j["path"])
    for c in manifest["closures"]:
        prim = stage.GetPrimAtPath(c["path"])
        if not prim or not prim.IsA(UsdPhysics.SphericalJoint):
            errors.append("Missing spherical closure: " + c["path"])
        elif not UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
            errors.append("Closure was not excluded from tree articulation: " + c["path"])
    for g in manifest["gears"]:
        prim = stage.GetPrimAtPath(g["path"])
        instances = [str(a) for a in prim.GetAppliedSchemas() if str(a).startswith("PhysxMimicJointAPI:")] if prim else []
        if not instances:
            errors.append("Native articulation mimic missing: " + g["path"])
        elif hasattr(PhysxSchema, "PhysxMimicJointAPI"):
            api = PhysxSchema.PhysxMimicJointAPI(prim, instances[0].split(":", 1)[1])
            if not math.isclose(float(api.GetGearingAttr().Get()), float(g["mimic_gearing"]), abs_tol=1e-7):
                errors.append("Mimic gearing mismatch: " + g["path"])
            expected_reference = next(j["path"] for j in manifest["joints"] if j["id"] == g["sprocket_joint_id"])
            if list(map(str, api.GetReferenceJointRel().GetTargets())) != [expected_reference]:
                errors.append("Mimic reference joint differs: " + g["path"])
            if not math.isclose(float(api.GetOffsetAttr().Get()), float(g["mimic_offset_degrees"]), abs_tol=1e-6):
                errors.append("Mimic offset differs: " + g["path"])
    if errors:
        raise RuntimeError("Native USD/PhysX schema gate failed: " + "; ".join(errors))
    return {"rigid_bodies":len(manifest["bodies"]), "closures":len(manifest["closures"]),
            "native_mimic_gears":len(manifest["gears"]), "schema_gate":"PASS"}


def articulation_probe(stage, sim_view, manifest, UsdPhysics):
    """Compare authored root with every runtime body/joint classification."""
    root_path = manifest["articulation_root"]
    root = stage.GetPrimAtPath(root_path)
    scene_paths = []
    root_paths = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene_paths.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root_paths.append(str(prim.GetPath()))
    root_id = next(b["id"] for b in manifest["bodies"] if b["path"] == root_path)
    first_children = [j["body_id"] for j in manifest["joints"] if j["parent_id"] == root_id][:5]
    by_id = {b["id"]: b["path"] for b in manifest["bodies"]}
    paths = list(dict.fromkeys([root_path] + [by_id[i] for i in first_children] +
                               [manifest["joints"][0]["path"], manifest["physics_scene_path"]]))
    owner = root.GetRelationship("physics:simulationOwner") if root else None
    result = {
        "root_path": root_path,
        "usd_root_exists": bool(root),
        "usd_root_schemas": [str(s) for s in root.GetAppliedSchemas()] if root else [],
        "usd_root_has_articulation_root_api": bool(root and root.HasAPI(UsdPhysics.ArticulationRootAPI)),
        "usd_articulation_root_paths": root_paths,
        "usd_physics_scene_paths": scene_paths,
        "usd_root_simulation_owner_targets": [str(p) for p in owner.GetTargets()] if owner else [],
        "sampled_runtime_object_types": {},
    }
    query = getattr(sim_view, "get_object_type", None)
    result["object_type_query_available"] = callable(query)
    if query is not None and callable(query):
        for path in paths:
            try:
                result["sampled_runtime_object_types"][path] = str(query(path))
            except Exception as exc:
                result["sampled_runtime_object_types"][path] = "QUERY_ERROR: %s: %s" % (type(exc).__name__, exc)
        inventory = {"body_paths_by_type": {}, "joint_paths_by_type": {},
                     "ancestor_object_types": {}, "root_link_paths": [],
                     "query_errors": {}}
        for path in ("/World", "/World/Robot"):
            try:
                inventory["ancestor_object_types"][path] = str(query(path))
            except Exception as exc:
                inventory["query_errors"][path] = "%s: %s" % (type(exc).__name__, exc)
        for category, items in (("body_paths_by_type", manifest["bodies"]),
                                ("joint_paths_by_type", manifest["joints"])):
            for item in items:
                path = item["path"]
                try:
                    obj_type = str(query(path))
                    inventory[category].setdefault(obj_type, []).append(path)
                    if obj_type.endswith(".ArticulationRootLink"):
                        inventory["root_link_paths"].append(path)
                except Exception as exc:
                    inventory["query_errors"][path] = "%s: %s" % (type(exc).__name__, exc)
        result["runtime_object_inventory"] = inventory
    return result


def version_info(app):
    info = {"python":sys.version, "platform":platform.platform(), "runtime_execution":"Isaac Sim"}
    for package in ("isaacsim", "isaacsim-core", "numpy"):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = "not available as package metadata"
    try:
        import omni.kit.app
        ka = omni.kit.app.get_app()
        info["kit_version"] = ka.get_build_version()
        em = ka.get_extension_manager()
        info["physx_extension"] = em.get_enabled_extension_id("omni.physx")
        info["tensors_extension"] = em.get_enabled_extension_id("omni.physics.tensors")
    except Exception as e:
        info["extension_version_error"] = str(e)
    return info


def resolve_source(args, manifest):
    base = Path(args.manifest).resolve().parent
    def resolve(value, explicit=False):
        path = Path(value)
        return path.resolve() if explicit or path.is_absolute() else (base/path).resolve()
    source_root = resolve(args.source_root or manifest.get("source_root", "source"), bool(args.source_root))
    scene_value = args.scene_xml or manifest.get("source_scene") or manifest.get("source_scene_xml")
    if not scene_value:
        raise RuntimeError("Manifest does not identify original source scene")
    scene_xml = resolve(scene_value, bool(args.scene_xml))
    if not source_root.is_dir() or not scene_xml.is_file():
        raise FileNotFoundError("Supply --source-root and --scene-xml paths on this machine")
    if sha256(scene_xml) != manifest["source_scene_sha256"]:
        raise RuntimeError("Controller scene XML hash differs from exported USD source")
    return source_root, scene_xml


def run(args, status):
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "isaac_excavator_graph_v1":
        raise RuntimeError("Unknown manifest schema")
    scene_path = Path(args.scene).resolve() if args.scene else manifest_path.with_name("scene.usda")
    if manifest.get("runtime_blockers"):
        raise RuntimeError("Exporter reports runtime blockers: " + str(manifest["runtime_blockers"]))
    dt = args.dt if args.dt is not None else float(manifest["dt"])
    if not 0 < dt <= .002:
        raise ValueError("Choose 0 < --dt <= 0.002 s for this mechanism")
    steps = int(round(args.duration/dt))
    if not args.preflight and (steps < 1 or abs(steps*dt-args.duration) > 1e-9):
        raise ValueError("Duration must be a positive integral number of timesteps")
    shutil.copy2(manifest_path, Path(args.output)/"manifest.json")
    status.update({"manifest_sha256":sha256(manifest_path), "scene_sha256":sha256(scene_path),
                   "dt":dt, "requested_duration":args.duration, "mode":args.mode,
                   "preflight_only":args.preflight, "native_steps_completed":0, "native_runtime_executed":False, "native_physics_initialized":False,
                   "initial_rigid_state_validated_at_t0":False, "first_controlled_step_articulation_confirmed":False,
                   "prestep_articulation_confirmed":False, "physx_start_simulation_called":False,
                   "physx_flush_changes_called":False,
                   "uncontrolled_warmup_steps":0,
                   "state_writes_after_initialization":0, "mujoco_integration_steps":0,
                   "validation_passed":False, "full_force_energy_validation":"NOT_RUN", "physics_backend_validation_complete":False,
                   "validation_scope":"native-run evidence only",
                   "stop_when_mission_done":bool(args.stop_when_done)})
    # SimulationApp must precede every omni/pxr import in standalone programs.
    try:
        from isaacsim import SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp
        except ImportError as e:
            raise RuntimeError("Isaac Sim runtime unavailable. Use its python.bat or python.sh; ordinary conda Python cannot execute PhysX.") from e
    # No camera/Replicator calls in the live dynamics process. A requested
    # movie is rendered from saved states by a separate process after shutdown.
    # Headless alone still creates an updating viewport on Isaac Sim.
    app_config = {"headless": args.headless, "disable_viewport_updates": args.headless,
                  "renderer": "RaytracedLighting", "width": 640, "height": 360, "multi_gpu": False,
                  "max_gpu_count": 1, "anti_aliasing": 0, "denoiser": False,
                  "extra_args": ["--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=0.15"]}
    status["render_policy"] = "deferred_state_replay_no_camera_in_physics_loop"
    status["simulation_app_config"] = app_config
    status["video_requested"] = bool(args.video)
    json_write(Path(args.output)/"status.json", status)
    app = SimulationApp(app_config)
    bridge = None
    physics = None
    rows = []
    controls = []
    contact_rows = []
    errors = []
    error_subscription = None
    error_stream = None
    diag_file = None
    store = None
    row = None
    rolling_report = None
    try:
        import omni.kit.app
        import carb.settings
        import omni.usd
        import omni.timeline
        import omni.physx
        from pxr import UsdPhysics, UsdGeom, PhysxSchema, UsdUtils
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.physics.tensors", True)
        import omni.physics.tensors as tensors
        status["versions"] = version_info(app)
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        usd_context = omni.usd.get_context()
        if not usd_context.open_stage(str(scene_path)):
            raise RuntimeError("Isaac could not open scene.usda")
        stage = usd_context.get_stage()
        status["schema"] = schema_check(stage, manifest, UsdPhysics, PhysxSchema)
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, abs_tol=1e-12):
            raise RuntimeError("Stage units must be metres")
        scene_api = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath("/World/PhysicsScene"))
        if not scene_api:
            raise RuntimeError("PhysX scene API missing")
        scene_api.CreateEnableGPUDynamicsAttr().Set(False)
        scene_api.CreateBroadphaseTypeAttr().Set("MBP")
        scene_api.CreateTimeStepsPerSecondAttr().Set(round(1/dt))
        native_settings = carb.settings.get_settings()
        cylinder_key = "/physics/collisionApproximateCylinders"
        native_settings.set(cylinder_key, False)
        status["native_cylinder_approximation_setting"] = native_settings.get(cylinder_key)
        physics = omni.physx.get_physx_interface()
        for required in ("update_transformations",):
            if not callable(getattr(physics, required, None)):
                raise RuntimeError("Installed PhysX API lacks " + required)
        if not callable(getattr(physics, "get_error_event_stream", None)):
            raise RuntimeError("Installed PhysX API lacks the error event stream required to verify force application")
        error_stream = physics.get_error_event_stream()
        if error_stream is None or not callable(getattr(error_stream, "pump", None)):
            raise RuntimeError("PhysX error event stream cannot be pumped")
        error_subscription = error_stream.create_subscription_to_pop(
            lambda event: errors.append({"type":str(event.type), "payload":str(event.payload)}))
        # The timeline stays stopped. Explicit stage attachment parses the USD
        # scene into PhysX without performing an integration step.
        simulation = omni.physx.get_physx_simulation_interface()
        for required in ("attach_stage", "get_attached_stage", "simulate", "fetch_results"):
            if not callable(getattr(simulation, required, None)):
                raise RuntimeError("Installed PhysX simulation API lacks " + required)
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if stage_id <= 0:
            raise RuntimeError("Opened USD scene has no valid stage cache ID")
        attached_before = int(simulation.get_attached_stage())
        status["physx_stage_attachment"] = {"usd_stage_id":stage_id,
                                              "attached_stage_id_before":attached_before}
        if attached_before not in (0, stage_id):
            raise RuntimeError("PhysX is attached to a different USD stage (%d)" % attached_before)
        status["initialization_checkpoint"] = "before_explicit_physx_attach"
        print("ISAAC INIT: attach USD stage %d to PhysX (previous %d)" % (stage_id, attached_before), flush=True)
        if attached_before == 0:
            attach_result = simulation.attach_stage(stage_id)
            status["physx_stage_attachment"]["attach_result"] = attach_result if isinstance(attach_result, (bool, type(None))) else str(attach_result)
        attached_id = int(simulation.get_attached_stage())
        status["physx_stage_attachment"]["attached_stage_id"] = attached_id
        status["initialization_checkpoint"] = "after_explicit_physx_attach"
        if attached_id != stage_id:
            raise RuntimeError("PhysX did not attach the opened USD stage (USD %d, PhysX %d)" % (stage_id, attached_id))
        print("ISAAC INIT: PhysX attached USD stage %d" % attached_id, flush=True)
        error_stream.pump()
        if errors:
            raise RuntimeError("PhysX reported stage attachment errors: " + str(errors[-3:]))
        # Explicitly start the already attached native scene without integrating
        # time. A tensor view alone does not establish that links are in a scene.
        start_simulation = getattr(physics, "start_simulation", None)
        if not callable(start_simulation):
            raise RuntimeError("Installed PhysX API lacks start_simulation for zero-step scene initialization")
        status["initialization_checkpoint"] = "before_physx_start_simulation"
        status["physx_start_simulation_called"] = True
        start_result = start_simulation()
        status["physx_start_simulation_result"] = (start_result if isinstance(start_result, (bool, type(None))) else str(start_result))
        if start_result is False:
            raise RuntimeError("PhysX rejected the explicit simulation start")
        attached_after_start = int(simulation.get_attached_stage())
        status["physx_stage_attachment"]["attached_stage_id_after_start"] = attached_after_start
        if attached_after_start != stage_id:
            raise RuntimeError("PhysX changed the attached USD stage while starting simulation")
        error_stream.pump()
        if errors:
            raise RuntimeError("PhysX reported scene-start errors: " + str(errors[-3:]))
        status["initialization_checkpoint"] = "after_physx_start_simulation"
        print("ISAAC INIT: PhysX scene started without a native integration step", flush=True)
        # Some USD changes are buffered until the next simulation step. Ask
        # PhysX to process them now, while the original t=0 state is still
        # available for the independent rigid-body checks below.
        flush_changes = getattr(simulation, "flush_changes", None)
        status["physx_flush_changes_available"] = callable(flush_changes)
        if callable(flush_changes):
            status["initialization_checkpoint"] = "before_physx_flush_changes"
            flush_result = flush_changes()
            status["physx_flush_changes_called"] = True
            status["physx_flush_changes_result"] = (
                flush_result if isinstance(flush_result, (bool, type(None))) else str(flush_result))
            if flush_result is False:
                raise RuntimeError("PhysX rejected flushing buffered USD physics changes")
            if int(simulation.get_attached_stage()) != stage_id:
                raise RuntimeError("PhysX changed its attached USD stage while flushing changes")
            error_stream.pump()
            if errors:
                raise RuntimeError("PhysX reported errors while flushing changes: " + str(errors[-3:]))
            status["initialization_checkpoint"] = "after_physx_flush_changes"
            print("ISAAC INIT: PhysX buffered changes flushed", flush=True)
        status["initialization_checkpoint"] = "before_numpy_tensor_view"
        print("ISAAC INIT: create NumPy tensor view", flush=True)
        sim_view = tensors.create_simulation_view("numpy", stage_id=stage_id)
        status["initialization_checkpoint"] = "after_numpy_tensor_view"
        sim_view.set_subspace_roots("/")
        status["articulation_probe"] = articulation_probe(stage, sim_view, manifest, UsdPhysics)
        rb = sim_view.create_rigid_body_view("/World/Robot/*")
        paths = list(rb.prim_paths)
        expected_paths = [b["path"] for b in manifest["bodies"]]
        status["articulation_probe"]["rigid_body_view"] = {
            "observed_count": len(paths), "expected_count": len(expected_paths),
            "root_in_view": manifest["articulation_root"] in paths,
        }
        if len(paths) != len(expected_paths) or set(paths) != set(expected_paths):
            raise RuntimeError("Native rigid-body inventory differs from source manifest")
        root_id = next(b["id"] for b in manifest["bodies"] if b["path"] == manifest["articulation_root"])
        art_ids = {root_id}
        for _ in range(len(manifest["bodies"])):
            before = len(art_ids)
            art_ids.update(j["body_id"] for j in manifest["joints"] if j["parent_id"] in art_ids)
            if len(art_ids) == before:
                break
        expected_dofs = sum(j["type"] in ("hinge", "slide") and j["body_id"] in art_ids for j in manifest["joints"])
        status["articulation_probe"]["expected_articulation"] = {"count": 1, "links": len(art_ids), "dofs": expected_dofs}
        # Validate the untouched rigid bodies and the initialized articulation
        # before contacting the bridge or applying a single body wrench.
        order = np.array([paths.index(p) for p in expected_paths], dtype=np.int64)
        reverse = np.argsort(order)
        indices = np.arange(len(paths), dtype=np.int32)
        measures = NativeMeasurements(manifest)
        def state():
            transforms = np.asarray(rb.get_transforms()).reshape(-1, 7)[order].copy()
            velocities = np.asarray(rb.get_velocities()).reshape(-1, 6)[order].copy()
            return measures.kinematics(transforms, velocities)
        s = state()
        p0 = np.array([b["initial_pos"] for b in manifest["bodies"]])
        q0 = np.array([b["initial_quat_wxyz"] for b in manifest["bodies"]])
        position_error = float(np.max(np.linalg.norm(s[0]-p0, axis=1)))
        quaternion_error = float(np.max(1-np.abs(np.sum(s[1]*q0, axis=1))))
        if position_error > 2e-5 or quaternion_error > 2e-6:
            raise RuntimeError("PhysX initialized away from source state (possible discarded joint/implicit warmup): " + str((position_error, quaternion_error)))
        native_mass = np.asarray(rb.get_masses()).reshape(-1)[order]
        mass_error = float(np.max(np.abs(native_mass-measures.mass)/np.maximum(measures.mass, 1e-12)))
        native_com = np.asarray(rb.get_coms()).reshape(-1, 7)[order, :3]
        com_error = float(np.max(np.linalg.norm(native_com-measures.com_local, axis=1)))
        native_inertia = np.asarray(rb.get_inertias()).reshape(-1, 3, 3)[order].transpose(0, 2, 1)
        inertia_error = float(np.max(np.linalg.norm(native_inertia-measures.inertia, axis=(1,2)) /
                            np.maximum(np.linalg.norm(measures.inertia, axis=(1,2)), 1e-12)))
        if inertia_error > 5e-5:
            raise RuntimeError("Native full COM inertia tensors differ from source manifest")
        if mass_error > 2e-5 or com_error > 2e-5:
            raise RuntimeError("Native mass/COM differs from physical manifest")
        if np.max(np.abs(manifest.get("initial_qvel", [0.]))) > 0:
            raise RuntimeError("This controller run currently requires zero initial source generalized velocity")
        if np.max(np.abs(s[5])) > 1e-7 or np.max(np.abs(s[3])) > 1e-7:
            raise RuntimeError("Native initial velocities are nonzero: hidden warmup or mismatched initial state")
        status["initial_checks"] = {"max_position_error_m":position_error, "max_quaternion_dot_defect":quaternion_error,
                                    "max_mass_relative_error":mass_error, "max_com_error_m":com_error, "max_inertia_relative_frobenius_error":inertia_error}
        error_stream.pump()
        if errors:
            raise RuntimeError("PhysX reported initialization errors: " + str(errors[-3:]))
        status["initial_rigid_state_validated_at_t0"] = True
        root_path = manifest["articulation_root"]
        runtime_type = str(sim_view.get_object_type(root_path))
        status["articulation_probe"]["prestep_root_object_type"] = runtime_type
        # In the observed 110.3 runtime the authored root was reported as an
        # ArticulationLink before the first step. Record the classification,
        # then test the articulation view itself. A view cannot certify force
        # acceptance; the PhysX error stream is checked before AND after step 1.
        try:
            articulation = sim_view.create_articulation_view(root_path)
            if articulation is None:
                raise RuntimeError("Tensor API returned no articulation view")
            observed = {"count": int(articulation.count), "links": int(articulation.max_links),
                        "dofs": int(articulation.max_dofs)}
            status["articulation_probe"]["observed_articulation"] = observed
            if observed != status["articulation_probe"]["expected_articulation"]:
                raise RuntimeError("Native articulation link/DOF inventory differs from source tree")
        except Exception as exc:
            status["articulation_probe"]["view_failure"] = "%s: %s" % (type(exc).__name__, exc)
            raise
        error_stream.pump()
        if errors:
            raise RuntimeError("PhysX reported articulation initialization errors: " + str(errors[-3:]))
        status["prestep_articulation_confirmed"] = True
        status["native_physics_initialized"] = True
        status["native_articulation"] = observed
        if args.preflight:
            status["status"] = "PREFLIGHT_COMPLETED_NO_DYNAMIC_VALIDATION"
            status["preflight_passed"] = True
            return
        if args.video:
            status["video_capture"] = {"png_complete": False, "captured_frames": 0,
                                       "mode": "deferred_state_replay",
                                       "mp4_status": "DEFERRED_UNTIL_PHYSICS_EXITS"}
        source_root, scene_xml = resolve_source(args, manifest)
        bridge = Bridge(args.port, args.bridge_timeout)
        init = bridge.request({"op":"initialize", "source_root":str(source_root), "scene_xml":str(scene_xml),
                               "initial_qpos":manifest["initial_qpos"], "dt":dt, "case":args.case, "mode":args.mode})
        names = [b["name"] for b in manifest["bodies"]]
        if init.get("body_names") != names:
            raise RuntimeError("Controller bridge rigid-body order differs from manifest")
        status["bridge_initialization"] = init
        from wrenches import joint_wrenches, mechanical_observables
        contact_view = None
        contact_order = None
        def make_contact_view(current_sim_view):
            view = current_sim_view.create_rigid_contact_view("/World/Robot/*")
            cp = list(view.sensor_paths)
            if len(cp) != len(expected_paths) or set(cp) != set(expected_paths):
                raise RuntimeError("Contact sensor inventory differs from rigid-body inventory")
            return view, np.array([cp.index(p) for p in expected_paths])
        try:
            contact_view, contact_order = make_contact_view(sim_view)
            status["contact_measurements"] = "native reported net contact force; no contact moment/work inference"
        except Exception as e:
            status["contact_measurements"] = "UNAVAILABLE: " + str(e)
            if args.require_contact_forces:
                raise RuntimeError("Required native contact force readback unavailable") from e
            contact_view = None
        def read_contacts():
            cf = np.asarray(contact_view.get_net_contact_forces(dt)).reshape(-1, 3)[contact_order].copy()
            if not np.all(np.isfinite(cf)):
                raise RuntimeError("Nonfinite native contact-force measurement")
            return cf
        # PhysX has no rolling friction: soil grains receive explicit Coulomb
        # rolling-resistance torques from their reported net contact force.
        rolling = GrainRollingResistance(manifest, dt, args.grain_rolling_friction)
        rolling_note = None
        if rolling.enabled and contact_view is None:
            rolling.enabled = False
            rolling_note = "disabled: native contact-force readback unavailable"
        def rolling_report():
            report = rolling.report()
            if rolling_note:
                report["note"] = rolling_note
            return report
        status["soil_rolling_resistance"] = rolling_report()
        log_stride = max(1, int(round(args.log_dt/dt)))
        diag_file = open(Path(args.output)/"bridge_diagnostics.jsonl", "w", encoding="utf-8", buffering=1)
        store = CheckpointStore(args.output, names, dt)
        checkpoint_stride = max(1, int(round(args.checkpoint_dt/dt)))
        total_work = 0.0
        initial_energy = sum(measures.energy(s))
        max_closure = 0.0
        closure_peaks = np.zeros(len(manifest["closures"]), dtype=float)
        max_gear = 0.0
        max_coupling_power_difference = 0.0
        integrated_coupling_power_difference = 0.0
        wall_start = time.monotonic()
        def collect(t, ss):
            nonlocal max_closure, max_gear
            kin, pot = measures.energy(ss)
            gaps = measures.closure_gaps(ss)
            np.maximum(closure_peaks, gaps, out=closure_peaks)
            coords = measures.joint_coordinates(ss)
            gear_errors = [coords[g["rotor_joint_id"]]-g["ratio"]*coords[g["sprocket_joint_id"]] for g in manifest["gears"]]
            max_closure = max(max_closure, float(np.max(gaps, initial=0)))
            max_gear = max(max_gear, float(np.max(np.abs(gear_errors), initial=0)))
            return (t, ss[0].copy(), ss[1].copy(), ss[2].copy(), ss[3].copy(), kin, pot, total_work,
                    gaps, np.asarray(gear_errors))
        def checkpoint():
            status.update({"wall_seconds": time.monotonic()-wall_start,
                "max_sampled_closure_gap_m": max_closure,
                "max_arm_closure_gap_m": max((float(g) for c,g in zip(manifest["closures"], closure_peaks)
                                              if not c["name"].startswith("track_")), default=0.),
                "max_track_closure_gap_m": max((float(g) for c,g in zip(manifest["closures"], closure_peaks)
                                                if c["name"].startswith("track_")), default=0.),
                "per_closure_peak_gap_m": {c["name"]:float(g) for c,g in zip(manifest["closures"], closure_peaks)},
                "physx_error_events": list(errors), "soil_rolling_resistance": rolling_report()})
            diag_file.flush()
            store.flush(rows, controls, contact_rows, status)
        rows.append(collect(0, s))
        status["simulated_duration"] = 0.0
        status["status"] = "RUNNING"
        checkpoint()
        terminal_step = 0
        for step in range(steps):
            t = step*dt
            if not app.is_running():
                raise RuntimeError("Isaac application closed before requested duration")
            if timeline.is_playing():
                raise RuntimeError("Do not press Play: this runner owns exact native timestep advancement")
            reply = bridge.request({"op":"tick", "t":t, "positions":s[0].tolist(),
                                    "quaternions_wxyz":s[1].tolist(), "linear_velocities":s[2].tolist(),
                                    "angular_velocities":s[3].tolist()})
            efforts = np.asarray(reply["efforts"], dtype=float)
            if efforts.shape != (manifest["nu"],) or not np.all(np.isfinite(efforts)):
                raise RuntimeError("Controller effort vector has invalid size or nonfinite values")
            force, torque = joint_wrenches(manifest, s[0], s[1], s[2], s[3], efforts)
            force, torque = np.asarray(force), np.asarray(torque)
            if force.shape != (len(paths), 3) or torque.shape != force.shape or not np.all(np.isfinite(force)) or not np.all(np.isfinite(torque)):
                raise RuntimeError("Invalid actuator/passive body wrenches")
            passive_force, passive_torque = joint_wrenches(manifest, s[0], s[1], s[2], s[3], np.zeros_like(efforts))
            actual_actuator_power = float(np.sum((force-passive_force)*s[5]) + np.sum((torque-passive_torque)*s[3]))
            hydraulic = reply.get("hydraulics", {})
            projected_hydraulic_power = float(np.sum(hydraulic.get("arm_power_W", {}).get("mechanical", 0.)))
            projected_hydraulic_power += float(np.sum(hydraulic.get("travel_power_W", {}).get("mechanical", 0.)))
            coupling_power_difference = actual_actuator_power-projected_hydraulic_power
            max_coupling_power_difference = max(max_coupling_power_difference, abs(coupling_power_difference))
            integrated_coupling_power_difference += coupling_power_difference*dt
            coupling = {"native_actuator_power_W":actual_actuator_power, "bridge_hydraulic_mechanical_power_W":projected_hydraulic_power,
                        "native_minus_bridge_power_W":coupling_power_difference}
            # Added after the actuator coupling power: these torques are soil
            # contact resistance, not actuator work. They are applied and recorded.
            grain_torque = rolling.torques(s[3]) if rolling.enabled else None
            if grain_torque is not None:
                torque = torque + grain_torque
            if step == 0:
                status["first_controlled_step"] = {"t_start_s":0.0, "dt_s":dt,
                                                   "controller_tick_completed":True,
                                                   "tensor_wrench_call_returned":False,
                                                   "physx_error_free_before_step":False,
                                                   "commanded_efforts":efforts.tolist(),
                                                   "commanded_body_force_l2_N":float(np.linalg.norm(force)),
                                                   "commanded_body_torque_l2_Nm":float(np.linalg.norm(torque))}
            rb.apply_forces_and_torques_at_position(np.ascontiguousarray(force[reverse], dtype=np.float32),
                                                    np.ascontiguousarray(torque[reverse], dtype=np.float32),
                                                    np.ascontiguousarray(s[4][reverse], dtype=np.float32), indices, True)
            if step == 0:
                status["first_controlled_step"]["tensor_wrench_call_returned"] = True
            error_stream.pump()
            if errors:
                raise RuntimeError("PhysX error while applying body wrenches before native step: " + str(errors[-3:]))
            if step == 0:
                status["first_controlled_step"]["physx_error_free_before_step"] = True
            simulation.simulate(dt, t)
            # Wait for the native step and update tensor readback.
            simulation.fetch_results()
            if step == 0:
                # This is the first requested integration step, driven by the
                # controller at t=0 after a zero-step native scene start.
                status["native_steps_completed"] = 1
                status["native_runtime_executed"] = True
                status["first_controlled_step"]["native_step_fetched"] = True
                sim_view = tensors.create_simulation_view("numpy", stage_id=stage_id)
                sim_view.set_subspace_roots("/")
                after = articulation_probe(stage, sim_view, manifest, UsdPhysics)
                status["articulation_probe"]["after_first_controlled_step"] = after
                try:
                    articulation = sim_view.create_articulation_view(manifest["articulation_root"])
                    if articulation is None:
                        raise RuntimeError("Tensor API returned no articulation view")
                    observed_articulation = {"count":int(articulation.count),
                                            "links":int(articulation.max_links),
                                            "dofs":int(articulation.max_dofs)}
                except Exception as exc:
                    status["articulation_probe"]["first_controlled_step_view_failure"] = (
                        "%s: %s" % (type(exc).__name__, exc))
                    raise RuntimeError("PhysX did not expose the expected articulation after the first controlled step") from exc
                status["articulation_probe"]["first_controlled_step_observed_articulation"] = observed_articulation
                if observed_articulation != status["articulation_probe"]["expected_articulation"]:
                    raise RuntimeError("Native articulation link/DOF inventory differs from source tree after the first controlled step")
                rb = sim_view.create_rigid_body_view("/World/Robot/*")
                paths = list(rb.prim_paths)
                status["articulation_probe"]["rigid_body_view_after_first_controlled_step"] = {
                    "observed_count":len(paths), "expected_count":len(expected_paths),
                    "root_in_view":manifest["articulation_root"] in paths}
                if len(paths) != len(expected_paths) or set(paths) != set(expected_paths):
                    raise RuntimeError("Native rigid-body inventory changed during the first controlled step")
                order = np.array([paths.index(p) for p in expected_paths], dtype=np.int64)
                reverse = np.argsort(order)
                indices = np.arange(len(paths), dtype=np.int32)
                status["articulation_probe"]["observed_articulation"] = observed_articulation
                status["native_articulation"] = observed_articulation
                status["first_controlled_step_articulation_confirmed"] = True
                status["native_physics_initialized"] = True
                # The fresh tensor simulation view owns subsequent reads and
                # writes. Rebind its contact view and source-body ordering too.
                try:
                    contact_view, contact_order = make_contact_view(sim_view)
                    status["contact_measurements"] = "native reported net contact force; no contact moment/work inference"
                except Exception as exc:
                    status["contact_measurements"] = "UNAVAILABLE AFTER FIRST STEP: " + str(exc)
                    if args.require_contact_forces:
                        raise RuntimeError("Required native contact-force readback unavailable after the first controlled step") from exc
                    contact_view = None
                    contact_order = None
            sn = state()
            logging_step = (step == 0 or (step+1)%log_stride == 0 or step+1 == steps or
                            (step+1)%checkpoint_stride == 0 or
                            (args.stop_when_done and reply.get("metrics", {}).get("done") is True))
            contact_now = None
            if rolling.enabled and contact_view is None:
                # Same policy as at start-up: never abort a run that does not
                # require contact forces; report that the surrogate stopped.
                rolling.enabled = False
                rolling_note = "disabled after native step %d: contact-force readback unavailable" % (step+1)
            if rolling.enabled:
                contact_now = read_contacts()
                rolling.update_contacts(contact_now)
                rolling.account(grain_torque, s[3], sn[3])
            power0 = float(np.sum(force*s[5]) + np.sum(torque*s[3]))
            power1 = float(np.sum(force*sn[5]) + np.sum(torque*sn[3]))
            total_work += .5*dt*(power0+power1)
            row = collect((step+1)*dt, sn)  # compute gear unwrapping every step
            controls.append((t, efforts.copy(), force.copy(), torque.copy()))
            if logging_step:
                rows.append(row)
                diag_file.write(json.dumps({"t":t, "state_time":(step+1)*dt, "bridge_response":reply,
                    "coupling":coupling, "mechanical_observables_at_step_start":mechanical_observables(manifest, s[0], s[1], s[2], s[3], force, torque)}, allow_nan=False)+"\n")
                if contact_view is not None:
                    cf = contact_now if contact_now is not None else read_contacts()
                    contact_rows.append(((step+1)*dt, cf))
            s = sn
            status["native_steps_completed"] = step+1
            status["native_runtime_executed"] = True
            terminal_step = step+1
            status["simulated_duration"] = terminal_step*dt
            status["last_observed_mission_metrics"] = reply.get("metrics", {})
            if step == 0 or terminal_step%checkpoint_stride == 0:
                checkpoint()
            if error_stream is not None:
                error_stream.pump()
            if errors:
                raise RuntimeError("PhysX error event during simulation: " + str(errors[-3:]))
            if not args.headless and (step+1)%max(1, round(1/(30*dt))) == 0:
                physics.update_transformations(False, True, True, False)
                app.update()
            if (step+1)%max(1, round(1/dt)) == 0:
                print("PhysX %.3f / %.3f s | closure %.3g m | gear %.3g rad" % ((step+1)*dt, args.duration, max_closure, max_gear), flush=True)
            # A guarded mission finishes on a controller tick. That tick still
            # owns one physics interval, so complete it before finalization.
            if args.stop_when_done and reply.get("metrics", {}).get("done") is True:
                status["mission_done_observed_on_tick_s"] = t
                break
        terminal_time = terminal_step*dt
        last_saved_time = rows[-1][0] if rows else store.last_state_time
        if terminal_step and (last_saved_time is None or last_saved_time < terminal_time-1e-10):
            # The guarded task may end between the regular 10-ms samples.
            # Always retain the exact terminal native state for assessment.
            rows.append(row)
        final_response = bridge.request({"op":"finalize", "t":terminal_time, "positions":s[0].tolist(),
                    "quaternions_wxyz":s[1].tolist(), "linear_velocities":s[2].tolist(),
                    "angular_velocities":s[3].tolist()})
        diag_file.write(json.dumps({"t":terminal_time, "terminal":True, "bridge_response":final_response,
                "mechanical_observables":mechanical_observables(manifest, s[0], s[1], s[2], s[3])}, allow_nan=False)+"\n")
        status["bridge_final_response"] = final_response
        status["bridge_summary"] = bridge.request({"op":"shutdown"})
        last_metrics = final_response.get("last_metrics", {})
        status["mission_terminal_metrics"] = last_metrics
        run_status = "NATIVE_RUN_COMPLETED_UNASSESSED"
        if args.stop_when_done:
            if not last_metrics.get("done"):
                run_status = "NATIVE_MISSION_INCOMPLETE"
            elif last_metrics.get("failures"):
                run_status = "NATIVE_MISSION_FAILED"
        arm_closure_peaks = [float(gap) for c, gap in zip(manifest["closures"], closure_peaks)
                             if not c["name"].startswith("track_")]
        track_closure_peaks = [float(gap) for c, gap in zip(manifest["closures"], closure_peaks)
                               if c["name"].startswith("track_")]
        status.update({"status":run_status, "simulated_duration":terminal_time,
                       "wall_seconds":time.monotonic()-wall_start, "max_sampled_closure_gap_m":max_closure,
                       "max_arm_closure_gap_m":max(arm_closure_peaks, default=0.),
                       "max_track_closure_gap_m":max(track_closure_peaks, default=0.),
                       "per_closure_peak_gap_m":{c["name"]:float(gap) for c, gap in zip(manifest["closures"], closure_peaks)},
                       "max_sampled_gear_coordinate_error_rad":max_gear, "mechanical_energy_change_J":sum(measures.energy(s))-initial_energy,
                       "applied_wrench_midpoint_work_J":total_work,
                       "max_abs_native_minus_bridge_actuator_power_W":max_coupling_power_difference,
                       "integrated_native_minus_bridge_actuator_work_J":integrated_coupling_power_difference,
                       "coupling_note":"Measures physical body wrench power minus hydraulic projected-port power, including off-manifold and effort-limiting discrepancies.",
                       "unaccounted_work_J":sum(measures.energy(s))-initial_energy-total_work,
                       "soil_rolling_resistance":rolling_report(),
                       "energy_residual_interpretation":"Includes unmeasured contact and constraint work/dissipation; NOT a conservation-error validation.",
                       "validation_passed":False,
                       "remaining_assessment":"Review model equivalence, native errors, closure/gear convergence, task metrics and paired backend comparisons."})
    finally:
        try:
            if diag_file is not None:
                diag_file.close()
            if store is not None:
                # A Python exception may occur between regular samples. Retain
                # the latest successfully collected state as a partial endpoint.
                last_saved_time = rows[-1][0] if rows else store.last_state_time
                if row is not None and (last_saved_time is None or row[0] > last_saved_time + 1e-10):
                    rows.append(row)
                store.flush(rows, controls, contact_rows, status)
            if bridge:
                bridge.close()
        finally:
            # Kit shutdown may end the process before main() runs its finalizer.
            # Persist the native outcome while the application is still alive.
            exc_type, exc, exc_tb = sys.exc_info()
            if rolling_report is not None:
                try:
                    status["soil_rolling_resistance"] = rolling_report()
                except Exception as report_error:  # never block the terminal status write
                    status["soil_rolling_resistance_report_error"] = str(report_error)
            if exc is not None:
                status.update({"status":"FAILED", "validation_passed":False,
                               "error":str(exc), "traceback":"".join(traceback.format_exception(exc_type, exc, exc_tb))})
            elif status.get("status") in ("STARTED", "RUNNING"):
                status.update({"status":"FAILED", "validation_passed":False,
                               "error":"Native runner ended without completing preflight or requested simulation"})
            status["physx_error_events"] = errors
            status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                json_write(Path(args.output)/"status.json", status)
            finally:
                # Do not reset the physics scene: preserve the final native state.
                app.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="build/manifest.json")
    ap.add_argument("--scene", help="Defaults to scene.usda beside manifest")
    ap.add_argument("--source-root", help="Original Soil v01 package folder in the controller environment")
    ap.add_argument("--scene-xml", help="Original MuJoCo scene.xml used only by the controller bridge")
    ap.add_argument("--case", default="soil_final")
    ap.add_argument("--preflight", action="store_true", help="Check native initialization and articulation without stepping")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--stop-when-done", action="store_true", help="Stop after the final controlled interval when the guarded soil mission is done")
    ap.add_argument("--video", action="store_true", help="Record deferred-video intent; use launch.py to render after physics")
    ap.add_argument("--checkpoint-dt", type=float, default=.25, help="Simulation seconds per durable measurement chunk")
    ap.add_argument("--capture-stride", type=int, default=100, help="Native step interval between video frames (100 steps = 2x video at 20 fps)")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--log-dt", type=float, default=.01)
    ap.add_argument("--mode", choices=("mission", "zero_effort"), default="mission")
    ap.add_argument("--port", type=int, default=47653)
    ap.add_argument("--bridge-timeout", type=float, default=120.0)
    ap.add_argument("--require-contact-forces", action="store_true")
    ap.add_argument("--grain-rolling-friction", type=float, default=None,
                    help="Soil-grain rolling-resistance coefficient in metres (MuJoCo convention), applied as "
                         "explicit torques; default derives 0.008 m from the manifest, 0 disables it")
    ap.add_argument("--output", default="results/isaac_smoke")
    args = ap.parse_args()
    if not math.isfinite(args.checkpoint_dt) or args.checkpoint_dt <= 0:
        ap.error("--checkpoint-dt must be positive and finite")
    if args.capture_stride < 1:
        ap.error("--capture-stride must be positive")
    if args.grain_rolling_friction is not None and (not math.isfinite(args.grain_rolling_friction) or
                                                    args.grain_rolling_friction < 0):
        ap.error("--grain-rolling-friction must be finite and nonnegative")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"status.json").exists():
        ap.error("Output folder already contains status.json; choose a new --output to preserve prior evidence")
    status = {"status":"STARTED", "validation_passed":False, "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rc = 0
    try:
        run(args, status)
    except Exception as exc:
        status.update({"status":"FAILED", "validation_passed":False, "error":str(exc), "traceback":traceback.format_exc()})
        print("ISAAC RUN FAILED: " + str(exc), file=sys.stderr)
        rc = 1
    finally:
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_write(out/"status.json", status)
    print(str(out/"status.json"), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
