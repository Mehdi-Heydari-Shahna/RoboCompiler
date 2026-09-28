"""Evidence-driven acceptance and an offline report for real Isaac Sim runs.

This module imports no Isaac, USD, MuJoCo or Pinocchio package. It never treats
a missing rollout, a controller prediction, or an old MuJoCo log as an Isaac
result. The acceptance limits are declared here before running the campaign.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .evidence import case_process_evidence

CASES = ("nominal", "fine", "low_friction", "payload", "strong_push",
         "no_actuation", "PD_ablation")
POSITIVE_CASES = CASES[:5]
CASE_PARAMETERS = {
    name: dict(dt=0.0005 if name == "fine" else 0.001,
               friction=0.55 if name == "low_friction" else 0.8,
               payload=1.5 if name == "payload" else 0.0,
               push=1.25 if name == "strong_push" else 1.0,
               feedforward=name != "PD_ablation", actuation=name != "no_actuation")
    for name in CASES
}
LIMITS = {
    "duration_s": 26.0,
    "duration_error_s": 0.0011,
    "final_position_error_m": 0.025,
    "rms_body_error_m": 0.035,
    "peak_body_error_m": 0.10,
    "final_yaw_error_rad": 0.08,
    "min_base_height_m": 0.18,
    "max_tilt_rad": 0.25,
    "min_joint_margin_rad": -1e-5,
    "peak_torque_limit_fraction": 1.000001,
    "max_qp_violation": 0.005,
    "final_speed_m_s": 0.03,
    "unexpected_contact_instances": 0,
    "max_foot_fk_error_m": 2e-5,
    "refinement_max_position_m": 0.025,
    "refinement_final_position_m": 0.010,
    "refinement_max_orientation_rad": 0.10,
}
MOTOR_LIMITS = np.tile([23.7, 23.7, 45.43], 4)
_SHAPES = {
    "time": (), "q": (18,), "v": (18,), "q_ref": (18,),
    "torque": (12,), "passive_torque": (12,), "feet": (4, 3),
    "foot_ref": (4, 3), "cmg_feet": (4, 3), "stance": (4,),
    "predicted_force": (4, 3), "qp_violation": (), "push": (3,),
    "measured_support_force": (4,),
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(value, limit=True, relation="==", unit=""):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        value = None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        passed = value is not None and {
            "==": lambda: value == limit,
            "<=": lambda: value <= limit,
            ">=": lambda: value >= limit,
        }[relation]()
    except (TypeError, ValueError):
        passed = False
    return dict(passed=bool(passed), value=value, limit=limit, relation=relation, unit=unit)


def _timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result if result.tzinfo is not None else None
    except ValueError:
        return None


def _finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)


def _load_trajectory(path):
    try:
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
    except (OSError, ValueError, EOFError) as error:
        return {}, [f"trajectory.npz unreadable: {type(error).__name__}: {error}"]
    problems = []
    time = arrays.get("time", np.empty(0))
    count = len(time) if time.ndim else 0
    if count < 2:
        problems.append("At least two recorded physics states are required.")
    for name, shape in _SHAPES.items():
        if name not in arrays:
            problems.append(f"Missing trajectory array: {name}")
        elif arrays[name].shape != (count,) + shape:
            problems.append(f"Wrong shape for {name}: {arrays[name].shape}, expected {(count,) + shape}")
        elif arrays[name].dtype.kind not in "bifu" or not np.all(np.isfinite(arrays[name])):
            problems.append(f"Nonfinite or nonnumeric trajectory array: {name}")
    if not problems:
        if not np.all(np.diff(time) > 0) or abs(float(time[0])) > 0.0011:
            problems.append("Time must increase strictly and begin at simulation time zero.")
        if not np.all(np.isin(arrays["stance"], [0, 1])):
            problems.append("Stance must contain only 0/1 values.")
        if np.any(arrays["measured_support_force"] < 0):
            problems.append("Measured support is a nonnegative normal-force magnitude.")
    return arrays, problems


def _video_evidence(outdir, metadata, process_evidence=None):
    record = metadata.get("video", {})
    if not isinstance(record, dict):
        record = {}
    relative = record.get("path") or record.get("file") or record.get("filename")
    path = Path(outdir) / relative if isinstance(relative, str) else Path(outdir) / "Go2_IsaacSim.mp4"
    try:
        inside = path.resolve().is_relative_to(Path(outdir).resolve())
    except (OSError, ValueError):
        inside = False
    exists = inside and path.is_file() and path.stat().st_size > 0
    actual_hash = _hash(path) if exists else None
    frames = record.get("frames", record.get("frame_count", 0))
    fps, duration = record.get("fps"), record.get("duration_s")
    expected_duration = metadata.get("simulated_s")
    provenance = record.get("provenance") or record.get("source") or record.get("capture_mode")
    process_evidence = process_evidence or case_process_evidence(Path(outdir).parent, Path(outdir).name, metadata)
    checks = {
        "capture_process_exited_cleanly": _gate(process_evidence["passed"]),
        "nonempty_local_video": _gate(exists),
        "capture_and_encode_valid": _gate(record.get("valid") is True and record.get("encoded") is True),
        "recorded_frames": _gate(frames if _finite_number(frames) else None, 2, ">=", "frames"),
        "distinct_rendered_frames": _gate(record.get("unique_frames") if _finite_number(record.get("unique_frames")) else None, 2, ">=", "frames"),
        "capture_provenance": _gate(bool(provenance)),
        "positive_frame_rate": _gate(_finite_number(fps) and fps > 0),
    }
    duration_ok = (_finite_number(duration) and _finite_number(expected_duration)
                   and _finite_number(fps) and fps > 0 and abs(duration - expected_duration) <= 1.0 / fps + 1e-6)
    checks["recorded_duration_covers_run"] = _gate(duration_ok)
    frames_ok = (_finite_number(frames) and _finite_number(duration)
                 and _finite_number(fps) and fps > 0 and abs(frames / fps - duration) <= 1e-6)
    checks["frame_count_matches_duration"] = _gate(frames_ok)
    if record.get("sha256"):
        checks["recorded_hash_matches_video"] = _gate(record["sha256"] == actual_hash)
    if record.get("run_id") is not None:
        checks["same_run_id"] = _gate(record["run_id"] == metadata.get("run_id"))
    if metadata.get("schema_version", 1) == 2 and metadata.get("video_requested") is True:
        audit_meta = metadata.get("capture_audit", {})
        if not isinstance(audit_meta, dict):
            audit_meta = {}
        audit_path = Path(outdir)/"capture_timing.json"
        audit = _json(audit_path)
        frames_audit = audit.get("frames", [])
        warmup = audit.get("warmup", {})
        audit_ok = (audit.get("physics_unchanged") is True and audit_meta.get("physics_unchanged") is True
                    and isinstance(frames_audit, list) and bool(frames_audit)
                    and audit.get("frame_count") == len(frames_audit) == frames == audit_meta.get("frames")
                    and audit.get("run_id") == metadata.get("run_id")
                    and audit_meta.get("sha256") == _hash(audit_path)
                    and audit_meta.get("mode") == audit.get("mode") == record.get("capture_mode"))
        states = [warmup, *frames_audit] if isinstance(frames_audit, list) else []
        for state in states:
            audit_ok = audit_ok and isinstance(state, dict) and state.get("passed") is True
            if not audit_ok:
                break
            for key, tolerance in (("max_abs_dq", 1e-7), ("max_abs_dv", 1e-7), ("abs_dt_s", 1e-9)):
                value = state.get(key)
                audit_ok = audit_ok and _finite_number(value) and 0 <= value <= tolerance
        if audit_ok:
            times = [state.get("time_s") for state in frames_audit]
            audit_ok = all(_finite_number(t) for t in times) and _finite_number(fps) and fps > 0
            if audit_ok:
                audit_ok = bool(np.allclose(times, np.arange(len(times))/fps, rtol=0, atol=1e-8))
        checks["zero_physics_capture_audit"] = _gate(audit_ok)
        if (metadata.get("capture_audit_schema_version") == 2 or metadata.get("code_revision") == 3
                or audit.get("schema_version") == 2):
            readiness_ok = (audit_ok and audit.get("schema_version") == 2
                            and audit_meta.get("schema_version") == 2
                            and audit.get("camera_ready") is True
                            and audit_meta.get("camera_ready") is True
                            and audit.get("failed") is False
                            and audit.get("capture_on_play") is (audit.get("mode") == "render")
                            and audit.get("annotator_device") == "cpu")
            resolution = audit.get("resolution")
            readiness_ok = readiness_ok and (isinstance(resolution, list) and len(resolution) == 2
                            and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in resolution))
            if readiness_ok:
                for state in states:
                    samples = state.get("samples")
                    passes, maximum = state.get("render_passes"), state.get("max_render_passes")
                    readiness_ok = (state.get("rgb_ready") is True and isinstance(samples, list) and bool(samples)
                                    and isinstance(passes, int) and isinstance(maximum, int)
                                    and 1 <= passes <= maximum)
                    if not readiness_ok:
                        break
                    sample = samples[-1]
                    expected_shapes = ([resolution[1], resolution[0], 3], [resolution[1], resolution[0], 4])
                    readiness_ok = (isinstance(sample, dict) and sample.get("empty") is False
                                    and sample.get("dtype") == state.get("last_rgb_dtype") == "uint8"
                                    and sample.get("shape") == state.get("last_rgb_shape")
                                    and sample.get("shape") in expected_shapes
                                    and sample.get("render_pass") == passes)
                    if not readiness_ok:
                        break
            checks["camera_rgb_readiness"] = _gate(readiness_ok)
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks,
                path=str(path) if exists else None, sha256=actual_hash,
                metadata=record, scope="Video supports visual review; it does not replace numerical acceptance.")


def summarize_case(outdir):
    """Compute one case's acceptance from its actual saved PhysX evidence.

    Writes ``outdir/summary.json``. ``mission_passed`` is evaluated even for
    controls; ``expected_outcome_passed`` applies control-specific semantics.
    Missing evidence is never an expected negative-control success.
    """
    outdir = Path(outdir)
    metadata = _json(outdir / "case.json")
    case = metadata.get("name", metadata.get("case", outdir.name))
    arrays, problems = _load_trajectory(outdir / "trajectory.npz")
    checks = {
        "case_metadata_exists": _gate(bool(metadata)),
        "trajectory_schema_finite": _gate(not problems),
        "recognized_case": _gate(case in CASES),
        "directory_matches_case": _gate(case == outdir.name),
        "Isaac_PhysX_engine": _gate("isaac" in str(metadata.get("engine", "")).lower()
                                    and "physx" in str(metadata.get("engine", "")).lower()),
        "run_identity": _gate(bool(metadata.get("run_id"))),
        "source_hashes_recorded": _gate(isinstance(metadata.get("source_sha256"), dict)
                                       and bool(metadata.get("source_sha256"))),
        "contact_report_available": _gate(metadata.get("contact_report_available") is True),
        "support_measurement_definition": _gate(bool(metadata.get("support_force_definition"))),
        "valid_physical_termination": _gate(
            (metadata.get("termination_reason") == "duration" and metadata.get("completed") is True and not metadata.get("failure"))
            or (metadata.get("termination_reason") == "fall" and metadata.get("completed") is False)),
    }
    process = case_process_evidence(outdir.parent, outdir.name, metadata)
    checks["native_process_and_log_clean"] = _gate(process["passed"])
    start, finish = (_timestamp(metadata.get(key)) for key in ("started_utc", "finished_utc"))
    checks["run_timestamps"] = _gate(start is not None and finish is not None and finish >= start)
    parameters = metadata.get("parameters", metadata.get("params", {}))
    if not isinstance(parameters, dict):
        parameters = {}
    checks["declared_case_parameters"] = _gate(parameters == CASE_PARAMETERS.get(case))
    scene = _json(outdir / "scene_metadata.json")
    checks["scene_metadata_present"] = _gate(bool(scene))
    checks["scene_friction"] = _gate(scene.get("friction"), CASE_PARAMETERS.get(case, {}).get("friction"))
    checks["scene_payload"] = _gate(scene.get("point_payload_kg"), CASE_PARAMETERS.get(case, {}).get("payload"))
    checks["physical_course_present"] = _gate(scene.get("terrain") is True)
    source_counts = scene.get("source_counts", {})
    for name, expected in (("rigid_bodies", 13), ("revolute_joints", 12), ("foot_colliders", 4)):
        checks["physical_" + name] = _gate(source_counts.get(name), expected, "==", "count")
    metrics = {}
    if not problems:
        q, reference, t = arrays["q"], arrays["q_ref"], arrays["time"]
        error = np.linalg.norm(q[:, :3] - reference[:, :3], axis=1)
        moving = t >= 2.0
        yaw = float(abs(np.arctan2(np.sin(q[-1, 3] - reference[-1, 3]),
                                   np.cos(q[-1, 3] - reference[-1, 3]))))
        metrics.update(
            samples=len(t), recorded_start_s=float(t[0]), recorded_end_s=float(t[-1]),
            final_position_error_m=float(error[-1]),
            rms_body_error_m=float(np.sqrt(np.mean(error[moving] ** 2))) if moving.any() else None,
            rms_body_error_after_2s_samples=int(np.count_nonzero(moving)),
            rms_body_error_all_samples_m=float(np.sqrt(np.mean(error ** 2))),
            peak_body_error_m=float(error.max()), final_yaw_error_rad=yaw,
            final_speed_m_s=float(np.linalg.norm(arrays["v"][-1, :3])),
            sampled_min_base_height_m=float(q[:, 2].min()),
            sampled_max_tilt_rad=float(np.linalg.norm(q[:, 4:6], axis=1).max()),
            sampled_peak_torque_limit_fraction=float(np.max(np.abs(arrays["torque"]) / MOTOR_LIMITS)),
            sampled_max_qp_violation=float(arrays["qp_violation"].max()),
            max_foot_fk_error_m=float(np.linalg.norm(arrays["feet"] - arrays["cmg_feet"], axis=2).max()),
            rms_foot_tracking_error_m=float(np.sqrt(np.mean(np.sum((arrays["feet"] - arrays["foot_ref"]) ** 2, axis=2)))),
            max_motor_effort_Nm=float(np.abs(arrays["torque"]).max()),
            final_forward_progress_m=float(q[-1, 0] - q[0, 0]),
            measured_support_modes_after_2s=sorted(set(np.sum(arrays["measured_support_force"][moving] > 2.0, axis=1).astype(int).tolist())),
        )
        required_metadata = (
            "simulated_s", "duration_s", "max_qp_violation", "peak_torque_limit_fraction",
            "min_joint_margin_rad", "min_base_height_m", "max_tilt_rad", "unexpected_contact_instances",
        )
        for name in required_metadata:
            metrics[name] = metadata.get(name) if _finite_number(metadata.get(name)) else None
            checks["recorded_" + name] = _gate(metrics[name] is not None)
        # Whole-step extrema must be at least as conservative as saved samples.
        for name, sampled, relation in (
            ("min_base_height_m", "sampled_min_base_height_m", "<="),
            ("max_tilt_rad", "sampled_max_tilt_rad", ">="),
            ("peak_torque_limit_fraction", "sampled_peak_torque_limit_fraction", ">="),
            ("max_qp_violation", "sampled_max_qp_violation", ">="),
        ):
            value = metrics[name]
            sample = metrics[sampled]
            consistent = value is not None and (value <= sample + 1e-8 if relation == "<=" else value >= sample - 1e-8)
            checks["per_step_consistency_" + name] = _gate(consistent)
        simulated = metrics["simulated_s"]
        checks["recorded_final_state"] = _gate(abs(float(t[-1]) - simulated) if simulated is not None else None,
                                                  0.011, "<=", "s")
        checks["full_mission_requested"] = _gate(metrics["duration_s"], LIMITS["duration_s"], "==", "s")
        checks["sampled_motor_limits"] = _gate(metrics["sampled_peak_torque_limit_fraction"], LIMITS["peak_torque_limit_fraction"], "<=", "ratio")
        checks["independent_PhysX_CMG_foot_FK"] = _gate(metrics["max_foot_fk_error_m"], LIMITS["max_foot_fk_error_m"], "<=", "m")
        scale = CASE_PARAMETERS.get(case, {}).get("push", 1.0)
        expected_push = np.zeros_like(arrays["push"])
        first_pulse = (t >= 1.2 - 1e-10) & (t < 1.35 - 1e-10)
        second_pulse = (t >= 24.3 - 1e-10) & (t < 24.5 - 1e-10)
        expected_push[first_pulse, 1] = 32.0 * scale
        expected_push[second_pulse, 0] = -25.0 * scale
        checks["logged_external_push_profile"] = _gate(float(np.max(abs(arrays["push"] - expected_push))), 1e-8, "<=", "N")
        for key, expected in (
            ("external_push_impulse_Ns", [0., 32. * scale * max(0., min(simulated or 0., 1.35) - 1.2), 0.]),
            ("recovery_push_impulse_Ns", [-25. * scale * max(0., min(simulated or 0., 24.5) - 24.3), 0., 0.]),
        ):
            try:
                observed = np.asarray(metadata.get(key), dtype=float)
                difference = float(np.max(abs(observed - expected))) if observed.shape == (3,) and np.isfinite(observed).all() else None
            except (TypeError, ValueError):
                difference = None
            checks[key] = _gate(difference, 1e-6, "<=", "N s")
    evidence_passed = all(value["passed"] for value in checks.values())
    acceptance = {
        "completed": _gate(metadata.get("completed") is True),
        "duration_error": _gate(abs(metrics["simulated_s"] - 26.0) if metrics.get("simulated_s") is not None else None,
                                LIMITS["duration_error_s"], "<=", "s"),
        "no_execution_failure": _gate(not metadata.get("failure")),
    }
    for name, unit in (("final_position_error_m", "m"), ("rms_body_error_m", "m"),
                       ("peak_body_error_m", "m"), ("final_yaw_error_rad", "rad"),
                       ("min_base_height_m", "m"), ("max_tilt_rad", "rad"),
                       ("min_joint_margin_rad", "rad"), ("peak_torque_limit_fraction", "ratio"),
                       ("max_qp_violation", "mixed constraint SI"), ("final_speed_m_s", "m/s"),
                       ("unexpected_contact_instances", "contact instances")):
        relation = ">=" if name.startswith("min_") else "==" if name == "unexpected_contact_instances" else "<="
        acceptance[name] = _gate(metrics.get(name), LIMITS[name], relation, unit)
    modes = metrics.get("measured_support_modes_after_2s", [])
    acceptance["observed_two_foot_support"] = _gate(2 in modes)
    acceptance["observed_four_foot_support"] = _gate(4 in modes)
    mission_passed = evidence_passed and all(c["passed"] for c in acceptance.values())
    expected_checks = {}
    if case == "no_actuation":
        physical_failure = (
            metrics.get("sampled_min_base_height_m", 1) < 0.18
            or metrics.get("sampled_max_tilt_rad", 0) > 0.65
            or (metadata.get("completed") is True and metrics.get("final_position_error_m", 0) > 0.025)
        )
        expected_checks = {
            "evidence_complete": _gate(evidence_passed),
            "motor_effort_is_zero": _gate(metrics.get("max_motor_effort_Nm"), 1e-9, "<=", "Nm"),
            "mission_did_not_pass": _gate(not mission_passed),
            "observed_physical_failure": _gate(physical_failure),
            "physical_fall_termination": _gate(metadata.get("termination_reason") == "fall"),
            "course_not_completed": _gate(metadata.get("completed") is False),
            "missed_course": _gate(metrics.get("final_forward_progress_m", 1) < 0.3 or metrics.get("final_position_error_m", 0) > 0.3),
        }
        expected_passed = all(check["passed"] for check in expected_checks.values())
        interpretation = "Negative control: valid zero-actuation execution must physically fail the course. Missing evidence is a failure of the test."
    elif case == "PD_ablation":
        expected_checks = {"evidence_complete": _gate(evidence_passed)}
        expected_passed = evidence_passed
        interpretation = "Descriptive ablation: successful completion is allowed and does not invalidate the full controller. This case alone establishes no causal PACDM benefit."
    else:
        expected_passed = mission_passed
        interpretation = "Positive rollout: every declared mission and evidence gate must pass."
    result = dict(
        name=case, evaluated_utc=_utc(), run_id=metadata.get("run_id"),
        engine=metadata.get("engine"), source_sha256=metadata.get("source_sha256"),
        executed=not problems and bool(metadata), evidence_passed=evidence_passed,
        mission_passed=mission_passed,
        mission_evaluated=metrics.get("duration_s") == LIMITS["duration_s"],
        mission_status=("PASS" if mission_passed else "FAIL") if metrics.get("duration_s") == LIMITS["duration_s"] else "NOT_EVALUATED",
        expected_outcome_passed=expected_passed,
        passed=expected_passed, status="PASS" if expected_passed else "FAIL" if metadata or arrays else "UNEXECUTED",
        completed=metadata.get("completed") is True, metrics=metrics,
        evidence_checks=checks, mission_checks=acceptance, expected_outcome_checks=expected_checks,
        interpretation=interpretation, problems=problems,
        failure=metadata.get("failure"), termination_reason=metadata.get("termination_reason"),
        file_sha256={name: _hash(outdir / name) for name in ("case.json", "trajectory.npz", "scene_metadata.json", "scene.usda")},
        process_evidence=process,
        video=_video_evidence(outdir, metadata, process), parameters=parameters,
        dt_s=metadata.get("dt_s", parameters.get("dt")),
        support_force_definition=metadata.get("support_force_definition"),
    )
    if outdir.is_dir():
        _write(outdir / "summary.json", result)
    return result


def _rotation(q):
    """ZYX Euler chart to rotation matrices; no extra scientific dependency."""
    y, p, r = q[:, 3], q[:, 4], q[:, 5]
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    matrix = np.empty((len(q), 3, 3))
    matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2] = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2] = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2] = -sp, cp * sr, cp * cr
    return matrix


def _refinement(root, summaries):
    checks = {"both_missions_passed": _gate(all(summaries.get(name, {}).get("mission_passed") is True for name in ("nominal", "fine")))}
    metrics = {}
    coarse, errors_a = _load_trajectory(root / "nominal" / "trajectory.npz")
    fine, errors_b = _load_trajectory(root / "fine" / "trajectory.npz")
    aligned = (not errors_a and not errors_b and coarse["time"].shape == fine["time"].shape
               and np.allclose(coarse["time"], fine["time"], rtol=0, atol=1e-8))
    checks["common_recording_times"] = _gate(aligned)
    checks["nominal_timestep"] = _gate(summaries.get("nominal", {}).get("dt_s"), 0.001, "==", "s")
    checks["fine_timestep"] = _gate(summaries.get("fine", {}).get("dt_s"), 0.0005, "==", "s")
    if aligned:
        delta = np.linalg.norm(coarse["q"][:, :3] - fine["q"][:, :3], axis=1)
        rot_a, rot_b = _rotation(coarse["q"]), _rotation(fine["q"])
        trace = np.einsum("nij,nij->n", rot_a, rot_b)
        angles = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        metrics = dict(refinement_max_position_m=float(delta.max()), refinement_final_position_m=float(delta[-1]),
                       refinement_max_orientation_rad=float(angles.max()))
    for name in ("refinement_max_position_m", "refinement_final_position_m", "refinement_max_orientation_rad"):
        checks[name] = _gate(metrics.get(name), LIMITS[name], "<=", "rad" if name.endswith("rad") else "m")
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks, metrics=metrics,
                scope="One refinement pair (1 ms and 0.5 ms), using common physical timestamps. This is a finite resolution check, not a proof of continuum convergence.")


def _preflight(root, name, run_id, source_hashes):
    path = root / name
    record = _json(path)
    raw = record.get("checks", {})
    entries = list(raw.values()) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    checks = {
        "file_present": _gate(bool(record)),
        "declared_passed": _gate(record.get("passed") is True),
        "nonempty_passing_checks": _gate(bool(entries) and all(isinstance(check, dict) and check.get("passed") is True for check in entries)),
        "same_run_identity": _gate(bool(run_id) and record.get("run_id") == run_id),
        "same_source_hashes": _gate(bool(source_hashes) and record.get("source_sha256") == source_hashes),
    }
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks,
                file_sha256=_hash(path), record=record)


def _baseline_comparison(root, summaries):
    baseline_root = Path(__file__).resolve().parents[1] / "baseline_mujoco"
    native = _json(baseline_root / "nominal.json")
    result = dict(available=bool(native), required_for_acceptance=False,
                  scope="The supplied MuJoCo result is historical baseline evidence, never an Isaac run. Different contact solvers need not produce identical trajectories.")
    if native:
        keys = ("final_position_error_m", "rms_body_error_m", "min_base_height_m", "max_tilt_rad", "peak_torque_limit_fraction")
        result["metrics"] = {key: {"mujoco": native.get(key), "isaac": summaries.get("nominal", {}).get("metrics", {}).get(key)} for key in keys}
        result["baseline_sha256"] = _hash(baseline_root / "nominal.json")
    return result


def _plots(root, summaries):
    if not summaries.get("nominal", {}).get("executed"):
        return [], None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arrays, problems = _load_trajectory(root / "nominal" / "trajectory.npz")
        if problems:
            return [], "Nominal trajectory failed schema checks; plots omitted."
        t, q, qr = arrays["time"], arrays["q"], arrays["q_ref"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
        axes[0, 0].plot(qr[:, 0], qr[:, 1], "--", label="PACDM reference", color="#d79428")
        axes[0, 0].plot(q[:, 0], q[:, 1], label="PhysX measured", color="#058d93")
        axes[0, 0].set(xlabel="World x [m]", ylabel="World y [m]", title="Base path")
        axes[0, 0].legend()
        axes[0, 1].plot(t, np.linalg.norm(q[:, :3] - qr[:, :3], axis=1) * 1000, color="#058d93")
        axes[0, 1].axhline(1000 * LIMITS["peak_body_error_m"], ls="--", color="#b44343")
        axes[0, 1].set(xlabel="Time [s]", ylabel="3D error [mm]", title="Actual body tracking")
        axes[1, 0].plot(t, np.abs(arrays["torque"]) / MOTOR_LIMITS, lw=0.7)
        axes[1, 0].axhline(1.0, ls="--", color="#b44343")
        axes[1, 0].set(xlabel="Time [s]", ylabel="|Motor torque| / limit", title="Twelve commanded motor efforts")
        axes[1, 1].plot(t, arrays["measured_support_force"], lw=0.8)
        axes[1, 1].set(xlabel="Time [s]", ylabel="Normal force magnitude [N]", title="Measured PhysX foot contact")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        fig.suptitle("Go2 CMG / PACDM — Isaac Sim evidence")
        output = root / "nominal_metrics.png"
        fig.savefig(output, dpi=150, facecolor="white")
        plt.close(fig)
        return [output], None
    except Exception as error:
        return [], f"Optional plot generation failed: {type(error).__name__}: {error}"


def _html_report(root, report, plots):
    escape = html.escape
    rows = []
    for name, case in report["cases"].items():
        metrics = case["metrics"]
        number = lambda key: "not available" if metrics.get(key) is None else f"{metrics[key]:.5g}"
        rms_key = "rms_body_error_m" if case.get("mission_evaluated") else "rms_body_error_all_samples_m"
        values = [name, case["status"], case.get("mission_status", "NOT_EVALUATED"),
                  "PASS" if case["process_evidence"]["passed"] else "FAIL",
                  str(case["completed"]), number("final_position_error_m"),
                  number(rms_key), number("max_tilt_rad"), number("max_foot_fk_error_m")]
        rows.append("<tr>" + "".join("<td>" + escape(value) + "</td>" for value in values) + "</tr>")
    images = "".join('<img alt="Measured Isaac trajectory, tracking, motor effort and foot contact" src="data:image/png;base64,'
                     + base64.b64encode(path.read_bytes()).decode("ascii") + '">' for path in plots)
    gates = "".join("<tr><td>" + escape(name) + "</td><td>" + ("PASS" if check["passed"] else "FAIL") + "</td></tr>"
                    for name, check in report["checks"].items())
    limitations = "".join("<li>" + escape(item) + "</li>" for item in report["scope_notes"])
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2 CMG / PACDM — Isaac Sim validation</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#183348;background:#f2f6f8;margin:0}main{max-width:1150px;margin:auto;padding:38px 24px}
h1{font-size:36px;line-height:1.14}h2{font-size:23px;margin-top:30px}.status{padding:18px;background:#fff3d9;border-left:5px solid #b37a17;font-weight:700}
section{background:white;padding:24px;margin:24px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px;text-align:left;border-bottom:1px solid #d9e4ec}th{color:#48677d}.scroll{overflow:auto}img{width:100%;height:auto}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#eef3f6;padding:14px}code{font-size:13px}small{color:#48677d}
</style><main><small>FLOATING-BASE ROBOT · INDEPENDENT PHYSX EXECUTION</small>
<h1>Go2 CMG / PACDM<br>Isaac Sim validation</h1><p class="status">__STATUS__</p>
<p>__CONCLUSION__</p><p><small>Generated __TIME__ · Run __RUN__</small></p>
<section><h2>Recorded cases</h2><p>Case PASS refers to its declared expected outcome. Smoke status is diagnostic only; its body RMS covers all recorded samples, not the full-mission window. The zero-actuation control must fail the mission physically; the PD ablation is descriptive.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Requested check</th><th>26 s mission</th><th>Process/log</th><th>Physical duration complete</th><th>Final error [m]</th><th>Body RMS [m]</th><th>Tilt [rad]</th><th>Foot FK error [m]</th></tr></thead><tbody>__CASES__</tbody></table></div></section>
__IMAGES____SMOKE__<section><h2>Full-campaign gates (not a smoke-test checklist)</h2><table><tbody>__GATES__</tbody></table></section>
<section><h2>Interpretation and limits</h2><ul>__NOTES__</ul></section>
<section><h2>Exact acceptance evidence</h2><p>The JSON below contains every threshold, individual gate, source fingerprint and result-file hash. Reproduce the campaign in a new output directory to keep evidence from different runs separate.</p>
<details><summary>Machine-readable validation record</summary><pre>__JSON__</pre></details></section></main></html>"""
    smoke_section = ""
    if "smoke_checks" in report:
        smoke_rows = "".join("<tr><td>" + escape(name) + "</td><td>" + ("PASS" if check["passed"] else "FAIL") + "</td></tr>"
                             for name, check in report["smoke_checks"].items())
        smoke_section = "<section><h2>Requested smoke-check gates</h2><table>" + smoke_rows + "</table></section>"
    replacements = {"__SMOKE__": smoke_section, "__STATUS__": escape(report["status"]), "__CONCLUSION__": escape(report["conclusion"]),
                    "__TIME__": escape(report["evaluated_utc"]), "__RUN__": escape(str(report["run_id"] or "unexecuted")),
                    "__CASES__": "".join(rows), "__IMAGES__": images, "__GATES__": gates,
                    "__NOTES__": limitations, "__JSON__": escape(json.dumps(report, indent=2, allow_nan=False))}
    for key, value in replacements.items():
        page = page.replace(key, value)
    (root / "report.html").write_text(page, encoding="utf-8")


def report_suite(results_root, requested_cases):
    """Write ``validation.json`` and standalone ``report.html``; return dict.

    Only the complete seven-case campaign can receive ``full_validation``.
    Partial runs expose ``scoped_passed`` while keeping ``passed=False``.
    """
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(requested_cases))
    if any(name not in CASES for name in requested):
        raise ValueError("Unknown cases: " + ", ".join(name for name in requested if name not in CASES))
    summaries = {name: summarize_case(root / name) for name in requested}
    identities = [value.get("run_id") for value in summaries.values()]
    run_id = identities[0] if identities and identities[0] and all(value == identities[0] for value in identities) else None
    hashes = [value.get("source_sha256") for value in summaries.values()]
    source_hashes = hashes[0] if hashes and hashes[0] and all(value == hashes[0] for value in hashes) else None
    mechanics = _preflight(root, "mechanics.json", run_id, source_hashes)
    reference = _preflight(root, "reference_validation.json", run_id, source_hashes)
    convergence = _refinement(root, summaries)
    checks = {
        "all_seven_cases_requested": _gate(set(requested) == set(CASES)),
        "same_run_identity": _gate(run_id is not None),
        "same_source_hashes": _gate(source_hashes is not None),
        "requested_processes_clean": _gate(bool(summaries) and all(value["process_evidence"]["passed"] for value in summaries.values())),
        "mechanics_preflight": _gate(mechanics["passed"]),
        "PACDM_reference_preflight": _gate(reference["passed"]),
        "positive_missions": _gate(all(summaries.get(name, {}).get("mission_passed") is True for name in POSITIVE_CASES)),
        "timestep_refinement": _gate(convergence["passed"]),
        "zero_actuation_negative_control": _gate(summaries.get("no_actuation", {}).get("expected_outcome_passed") is True),
        "PD_ablation_recorded": _gate(summaries.get("PD_ablation", {}).get("evidence_passed") is True),
        "nominal_video_evidence": _gate(summaries.get("nominal", {}).get("video", {}).get("passed") is True),
    }
    full = all(value["passed"] for value in checks.values())
    executed = any(value["executed"] for value in summaries.values())
    scoped = bool(requested) and all(value["expected_outcome_passed"] for value in summaries.values()) and mechanics["passed"] and reference["passed"] and run_id is not None and source_hashes is not None
    manifest = _json(root/"suite_manifest.json")
    video_requested = manifest.get("video_requested")
    scoped = scoped and type(video_requested) is bool and (video_requested is False or summaries.get("nominal", {}).get("video", {}).get("passed") is True)
    if set(("nominal", "fine")).issubset(requested):
        scoped = scoped and convergence["passed"]
    status = "PASS" if full else "UNEXECUTED" if not executed else "PARTIAL" if set(requested) != set(CASES) else "FAIL"
    if full:
        conclusion = "The complete declared Isaac Sim / PhysX campaign passed its numerical, control, refinement and video evidence gates. This result is limited to the recorded model, course and test cases."
    elif not executed:
        conclusion = "Isaac Sim validation has not been executed or usable local Isaac evidence is absent. The package and the historical MuJoCo results do not establish Isaac validation."
    elif status == "PARTIAL":
        conclusion = "Only a partial campaign was requested. Its scoped checks " + ("passed" if scoped else "did not pass") + "; full Isaac validation is not established."
    else:
        conclusion = "The full requested campaign did not pass every acceptance gate. Review the failed checks; a video or a completed nominal trajectory alone does not establish validation."
    notes = [
        "Acceptance thresholds were declared before Isaac execution. Source MuJoCo mission limits are retained, with an explicit 1e-5 rad numerical allowance below the zero joint-limit margin.",
        "The 26 s mission and source motor caps are required. RMS body error is evaluated at t >= 2 s. Independent foot FK compares actual PhysX link-derived points with CMG kinematics; it does not compare two controller predictions.",
        "Physical contact forces come from PhysX contact reports. Support-mode counts use the reported normal impulse magnitude divided by physics dt (>2 N). The original MuJoCo gate used upward world-Z reaction, so the support definitions differ for sloped or side contacts.",
        "Predicted contact forces are controller outputs and are not measurements. Numerical contact-force equality between MuJoCo and PhysX is not an acceptance requirement.",
        "The full campaign includes nominal, finer timestep, lower friction, payload, stronger push, zero actuation and PD ablation. The negative control must have valid zero-torque physical failure evidence; the ablation is descriptive and need not fail.",
        "Refinement uses the original 0.025 m maximum path, 0.010 m final path and 0.10 rad maximum orientation differences. One timestep pair is not a proof of asymptotic convergence.",
        "The delivered Go2 is a tree mechanism with changing robot–environment contact constraints, not a permanent mechanically closed-loop leg robot. Passing this benchmark does not validate every closed-chain robot or physical hardware.",
        "A saved case is written before native shutdown. Acceptance also requires a matching parent manifest with an observed zero process exit, successful cleanup, and a native log without errors. Warnings are retained separately.",
        "The all-samples body RMS is reported separately. For a 2 s smoke check, the original t >= 2 s mission RMS contains only one sample and is not a full-run RMS.",
        "Source hashes, run IDs, timestamps and artifact hashes associate evidence with a run; they are reproducibility records, not cryptographic attestations of an untampered simulator.",
        "MuJoCo baseline results remain separately labelled historical evidence. Isaac remains unexecuted until local recorded PhysX evidence passes the gates above.",
    ]
    report = dict(schema_version=2, evaluated_utc=_utc(), status=status, passed=full,
                  full_validation=full, scoped_passed=bool(scoped), run_id=run_id,
                  conclusion=conclusion, requested_cases=requested, required_cases=list(CASES),
                  source_sha256=source_hashes, thresholds=LIMITS.copy(), checks=checks,
                  cases=summaries, mechanics=mechanics, reference=reference,
                  refinement=convergence, mujoco_baseline=_baseline_comparison(root, summaries),
                  scope_notes=notes)
    plots, plot_problem = _plots(root, summaries)
    report["plot_problem"] = plot_problem
    report["plots"] = [path.name for path in plots]
    _write(root / "validation.json", report)
    _html_report(root, report, plots)
    return report


def main(argv=None):
    """CLI used by launch.py; smoke success never implies mission acceptance."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "nominal", "full"), default="full")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    requested = list(CASES) if args.suite == "full" else ["nominal"]
    report = report_suite(args.output, requested)
    report["suite"] = args.suite
    identity_ok = report["run_id"] == args.run_id
    report["checks"]["requested_run_identity"] = _gate(identity_ok)
    if not identity_ok:
        report.update(passed=False, full_validation=False, scoped_passed=False,
                      conclusion="Evidence does not belong to the requested run ID; validation is not established.")
        if report["status"] != "UNEXECUTED":
            report["status"] = "FAIL"
    if args.suite == "smoke":
        nominal = report["cases"]["nominal"]
        metrics = nominal["metrics"]
        diagnostic_evidence = {name: record for name, record in nominal["evidence_checks"].items()
                               if name not in {"full_mission_requested", "native_process_and_log_clean"}}
        manifest = _json(args.output / "suite_manifest.json")
        video_requested = manifest.get("video_requested")
        smoke_checks = {
            "parent_process_and_native_log": _gate(nominal.get("process_evidence", {}).get("passed") is True),
            "video_request_recorded": _gate(type(video_requested) is bool),
            "requested_video_evidence": _gate(video_requested is False or nominal.get("video", {}).get("passed") is True),
            "same_run_identity": _gate(identity_ok),
            "mechanics_preflight": _gate(report["mechanics"]["passed"]),
            "PACDM_reference_preflight": _gate(report["reference"]["passed"]),
            "finite_physics_evidence": _gate(bool(diagnostic_evidence) and all(record["passed"] for record in diagnostic_evidence.values())),
            "completed_without_failure": _gate(nominal["completed"] and not nominal["failure"]),
            "two_second_duration_requested": _gate(metrics.get("duration_s"), 2.0, "==", "s"),
            "two_second_duration": _gate(abs(metrics.get("simulated_s", -1) - 2.0) if metrics.get("simulated_s") is not None else None, 0.0011, "<=", "s"),
        }
        for name in ("min_base_height_m", "max_tilt_rad", "min_joint_margin_rad", "peak_torque_limit_fraction", "max_qp_violation", "unexpected_contact_instances"):
            smoke_checks[name] = nominal["mission_checks"][name]
        physical_checks = {name: value for name, value in smoke_checks.items()
                           if name not in {"parent_process_and_native_log", "video_request_recorded", "requested_video_evidence"}}
        physics_passed = all(check["passed"] for check in physical_checks.values())
        smoke_passed = all(check["passed"] for check in smoke_checks.values())
        report["physics_diagnostic_passed"] = physics_passed
        nominal.update(physics_diagnostic_passed=physics_passed,
                       smoke_passed=smoke_passed, passed=smoke_passed,
                       expected_outcome_passed=smoke_passed,
                       status="SMOKE_PASS" if smoke_passed else "SMOKE_FAIL" if nominal["executed"] else "UNEXECUTED",
                       interpretation="Two-second standing/push diagnostic only. Physical checks, native process exit, and requested video are evaluated separately; no 26-second mission outcome is claimed.")
        if (args.output/"nominal").is_dir():
            _write(args.output/"nominal"/"summary.json", nominal)
        report["smoke_checks"] = smoke_checks
        report["smoke_passed"] = smoke_passed
        report["status"] = "SMOKE_PASS" if smoke_passed else "UNEXECUTED" if not nominal["executed"] else "SMOKE_FAIL"
        report["conclusion"] = ("The two-second startup diagnostic passed. The 26-second mission, perturbations, timestep refinement and negative control remain unvalidated."
                                if smoke_passed else ("The two-second physical diagnostic checks passed, but the process or requested video evidence failed. Isaac mission validation is not established."
                                                     if physics_passed else "The two-second startup diagnostic did not pass. Isaac mission validation is not established."))
        accepted = smoke_passed
    else:
        accepted = report["full_validation"] if args.suite == "full" else report["scoped_passed"]
    _write(args.output / "validation.json", report)
    _html_report(args.output, report, [args.output / name for name in report.get("plots", [])])
    print(json.dumps({key: report.get(key) for key in ("status", "full_validation", "scoped_passed", "smoke_passed", "conclusion")}, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
