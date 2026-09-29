"""Fail-closed scoring of measured Isaac Sim trajectories.

No simulator is imported here. Saved video is supporting evidence only; the
physics decision is based on complete time series and predeclared tolerances.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .protocol import ZERO_ACTUATION_DURATION, ZERO_ACTUATION_TIMESTEP, ZERO_MIN_DISPLACEMENT_M

MISSION_DURATION = 22.0
REQUIRED_CASES = ("nominal", "fine", "heavy_payload", "no_feedforward")
CASE_TIMESTEPS = {"nominal": .002, "fine": .001, "heavy_payload": .002, "no_feedforward": .002}
LIMITS = {
    "peak_position_m": .010,
    "peak_orientation_deg": .5,
    "docking_position_m": .0005,
    "heavy_docking_position_m": .002,
    "docking_orientation_deg": .1,
    "closure_m": .0001,
    "joint_linear_m": .0001,
    "joint_angular_deg": .05,
    "refinement_position_m": .0005,
    "refinement_orientation_deg": .03,
    "cross_engine_position_m": .002,
    "cross_engine_orientation_deg": .15,
}
_SHAPES = {
    "engine_time": (), "q": (24,), "velocity": (24,), "target_pose": (6,), "force": (6,),
    "requested_force": (6,), "wrench": (6,), "pose_error_m": (),
    "angle_error_rad": (), "closure_error_m": (), "joint_error_m": (),
    "joint_error_rad": (), "actuator_error_m": (6,), "power_W": (),
}


def _check(checks, name, passed, value=None, limit=None, unit="", detail=""):
    checks[name] = {"passed": bool(passed), "value": value, "limit": limit,
                    "unit": unit, "detail": detail}


def _bounded(checks, name, value, limit, unit=""):
    finite = value is not None and bool(np.isfinite(value))
    _check(checks, name, finite and value <= limit,
           float(value) if finite else None, float(limit), unit)


def _time_problem(time, timestep=None, duration=MISSION_DURATION):
    if not np.isfinite(duration) or duration <= 0:
        return "Duration must be positive and finite."
    if timestep is not None and (not np.isfinite(timestep) or timestep <= 0):
        return "Timestep must be positive and finite."
    t = np.asarray(time)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
        return "Time must be a finite one-dimensional series with at least two samples."
    if not np.all(np.diff(t) > 0):
        return "Time must increase strictly, with no duplicate or reordered samples."
    tol = max(1e-8, (timestep or .001) * 1e-5)
    if abs(float(t[0])) > tol or abs(float(t[-1]) - duration) > tol:
        return f"Required time coverage is 0 through {duration:g} s; got {t[0]:g} through {t[-1]:g} s."
    if timestep is not None:
        n = round(duration/timestep) + 1
        if len(t) != n or not np.allclose(t, np.arange(n)*timestep, atol=tol, rtol=0):
            return f"Expected all {n} samples at {timestep:g} s intervals; missing or irregular samples found."
    return None


def _pose_differences(q, target):
    position = np.linalg.norm(q[:, :3] - target[:, :3], axis=1)
    actual_r = Rotation.from_euler("ZYX", q[:, 3:6])
    target_r = Rotation.from_euler("ZYX", target[:, 3:6])
    angle = (target_r.inv()*actual_r).magnitude()
    return position, angle


def score_case(data, cmg, name, timestep, duration, feedforward):
    """Return JSON-safe measured metrics and acceptance checks for one case.

    ``data`` is the NPZ mapping described in the runner. Diagnostic cases can
    have shorter durations, but ``eligible_for_mission_acceptance`` remains
    false. No-feedforward is a control ablation and has no docking requirement.
    """
    checks = {}
    result = {"name": name, "timestep_s": float(timestep) if np.isfinite(timestep) else None,
              "duration_s": float(duration) if np.isfinite(duration) else None,
              "feedforward": bool(feedforward), "checks": checks,
              "eligible_for_mission_acceptance": name in REQUIRED_CASES and abs(duration-MISSION_DURATION) < 1e-8,
              "passed": False, "status": "FAIL"}
    errors = []
    try:
        time = np.asarray(data["time"], dtype=float)
    except (KeyError, TypeError, ValueError):
        time = np.array([])
        errors.append("Missing or invalid time array.")
    result["samples"] = int(len(time)) if time.ndim == 1 else 0
    problem = _time_problem(time, timestep, duration)
    _check(checks, "trajectory_complete", problem is None, detail=problem or "All scheduled samples are present.")
    arrays = {"time": time}
    for key, shape in _SHAPES.items():
        try:
            a = np.asarray(data[key], dtype=float)
            if a.shape != (result["samples"],) + shape:
                errors.append(f"{key}: expected {(result['samples'],)+shape}, got {a.shape}.")
            elif not np.all(np.isfinite(a)):
                errors.append(f"{key}: nonfinite values found.")
            else:
                arrays[key] = a
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key}: missing or invalid array.")
    _check(checks, "finite_complete_schema", not errors, detail=" ".join(errors))
    _check(checks, "full_mission", result["eligible_for_mission_acceptance"],
           float(duration) if np.isfinite(duration) else None, MISSION_DURATION, "s", "Short diagnostics cannot establish mission acceptance.")
    if errors or problem:
        result["errors"] = errors + ([problem] if problem else [])
        return result
    _bounded(checks, "engine_clock_matches_samples", float(np.max(abs(arrays["engine_time"]-time))), 2e-7, "s")
    if name in REQUIRED_CASES:
        _check(checks, "scheduled_timestep", abs(timestep-CASE_TIMESTEPS[name]) < 1e-12,
               timestep, CASE_TIMESTEPS[name], "s")
        _check(checks, "feedforward_mode", bool(feedforward) == (name != "no_feedforward"))
    q = arrays["q"]
    pos, angle = _pose_differences(q, arrays["target_pose"])
    _bounded(checks, "logged_position_consistency", float(np.max(abs(pos-arrays["pose_error_m"]))), 1e-8, "m")
    _bounded(checks, "logged_orientation_consistency", float(np.max(abs(angle-arrays["angle_error_rad"]))), 1e-7, "rad")
    result.update({
        "rms_position_error_m": float(np.sqrt(np.mean(pos**2))),
        "max_position_error_m": float(np.max(pos)),
        "max_orientation_error_deg": float(np.rad2deg(np.max(angle))),
        "max_closure_error_m": float(np.max(abs(arrays["closure_error_m"]))),
        "max_joint_error_m": float(np.max(abs(arrays["joint_error_m"]))),
        "max_joint_error_deg": float(np.rad2deg(np.max(abs(arrays["joint_error_rad"])))),
        "max_actuator_force_N": float(np.max(abs(arrays["force"]))),
        "max_requested_force_N": float(np.max(abs(arrays["requested_force"]))),
        "max_actuator_error_m": float(np.max(abs(arrays["actuator_error_m"]))),
        "positive_actuator_work_J": float(np.trapezoid(np.maximum(arrays["power_W"], 0), time)) if hasattr(np, "trapezoid") else float(np.trapz(np.maximum(arrays["power_W"], 0), time)),
    })
    joints = {j["id"]: j for j in cmg["joints"]}
    lower = np.array([joints[k]["limits"]["lower"] for k in cmg["coordinate_ids"]])
    upper = np.array([joints[k]["limits"]["upper"] for k in cmg["coordinate_ids"]])
    result["min_joint_limit_margin"] = float(min(np.min(q-lower), np.min(upper-q)))
    result["payload_mass_kg"] = float(next(b["mass_kg"] for b in cmg["bodies"] if b["id"] == "payload"))
    limit = float(cmg["actuation"]["force_limit_N"])
    saturated = np.any(abs(arrays["requested_force"]) > limit + 1e-8, axis=1)
    clipped = np.any(abs(arrays["force"] - arrays["requested_force"]) > 1e-8, axis=1)
    result["saturation_samples"] = int(np.count_nonzero(saturated | clipped))
    _bounded(checks, "no_saturation", result["saturation_samples"], 0, "samples")
    _bounded(checks, "force_limit", result["max_actuator_force_N"], limit + 1e-8, "N")
    _check(checks, "joint_limits", result["min_joint_limit_margin"] >= -1e-8,
           result["min_joint_limit_margin"], 0, "mixed SI")
    for key, metric, threshold, unit in (
        ("closure", "max_closure_error_m", "closure_m", "m"),
        ("joint_linear", "max_joint_error_m", "joint_linear_m", "m"),
        ("joint_angular", "max_joint_error_deg", "joint_angular_deg", "deg")):
        _bounded(checks, key, result[metric], LIMITS[threshold], unit)
    if name in REQUIRED_CASES:
        _bounded(checks, "peak_position", result["max_position_error_m"], LIMITS["peak_position_m"], "m")
        _bounded(checks, "peak_orientation", result["max_orientation_error_deg"], LIMITS["peak_orientation_deg"], "deg")
        dock = time >= 21.0 - 1e-8
        result["docking_max_position_error_m"] = float(np.max(pos[dock])) if np.any(dock) else None
        result["docking_max_orientation_error_deg"] = float(np.rad2deg(np.max(angle[dock]))) if np.any(dock) else None
        if feedforward:
            dock_limit = LIMITS["heavy_docking_position_m"] if name == "heavy_payload" else LIMITS["docking_position_m"]
            _bounded(checks, "docking_position", result["docking_max_position_error_m"], dock_limit, "m")
            _bounded(checks, "docking_orientation", result["docking_max_orientation_error_deg"], LIMITS["docking_orientation_deg"], "deg")
    if name == "zero_actuation":
        # A diagnostic has its own fixed schedule; it cannot establish a mission.
        checks.pop("full_mission", None)
        _check(checks, "diagnostic_duration", abs(duration-ZERO_ACTUATION_DURATION) < 1e-8,
               duration, ZERO_ACTUATION_DURATION, "s")
        _check(checks, "scheduled_timestep", abs(timestep-ZERO_ACTUATION_TIMESTEP) < 1e-12,
               timestep, ZERO_ACTUATION_TIMESTEP, "s")
        _check(checks, "feedforward_mode", feedforward is False)
        displacement = q[-1, :3]-q[0, :3]
        magnitude = float(np.linalg.norm(displacement))
        gravity = np.asarray(cmg["gravity_m_s2"], dtype=float)
        gravity_norm = float(np.linalg.norm(gravity))
        along_gravity = float(displacement @ gravity / gravity_norm) if gravity_norm > 0 else None
        _check(checks, "gravity_motion", magnitude >= ZERO_MIN_DISPLACEMENT_M,
               magnitude, ZERO_MIN_DISPLACEMENT_M, "m", "Minimum displacement, not a tracking objective.")
        _check(checks, "gravity_direction", along_gravity is not None and along_gravity >= ZERO_MIN_DISPLACEMENT_M,
               along_gravity, ZERO_MIN_DISPLACEMENT_M, "m", "Motion must be along declared gravity, not sideways or upward.")
        for field in ("force", "requested_force", "wrench"):
            _bounded(checks, f"zero_{field}", float(np.max(abs(arrays[field]))), 1e-12)
        result["gravity_displacement_m"] = magnitude
        result["displacement_along_gravity_m"] = along_gravity
    result["passed"] = all(c["passed"] for c in checks.values())
    if name == "zero_actuation":
        result["diagnostic_passed"] = result["passed"]
    result["status"] = "PASS" if result["passed"] else ("DIAGNOSTIC" if name not in REQUIRED_CASES else "FAIL")
    return result


def compare_trajectories(first, second, position_limit, orientation_limit_deg, label="comparison"):
    """Compare full missions, never just the overlap of truncated trajectories."""
    result = {"name": label, "passed": False, "checks": {}}
    checks = result["checks"]
    for key, data in (("first", first), ("second", second)):
        try:
            t = np.asarray(data["time"], dtype=float)
            q = np.asarray(data["q"], dtype=float)
            problem = _time_problem(t)
            valid = problem is None and q.shape == (len(t), 24) and np.all(np.isfinite(q))
            # All states must be measured at a regular interval, not two endpoints.
            regular = len(t) >= 11001 and np.allclose(np.diff(t), np.diff(t)[0], atol=1e-8, rtol=0)
            _check(checks, f"{key}_complete", valid and regular, detail=problem or ("" if regular else "Trajectory is sparse or irregular."))
        except (KeyError, TypeError, ValueError):
            _check(checks, f"{key}_complete", False, detail="Missing or invalid time/state data.")
    if not all(c["passed"] for c in checks.values()):
        return result
    ta = np.asarray(first["time"]); tb = np.asarray(second["time"])
    qa = np.asarray(first["q"]); qb = np.asarray(second["q"])
    # Use the coarser measured time grid; positions interpolate linearly and
    # orientation uses rotations on SO(3), avoiding Euler wrap artefacts.
    if len(ta) > len(tb):
        ta, tb, qa, qb = tb, ta, qb, qa
    pos_b = np.column_stack([np.interp(ta, tb, qb[:, i]) for i in range(3)])
    rot_a = Rotation.from_euler("ZYX", qa[:, 3:6])
    rot_b = Slerp(tb, Rotation.from_euler("ZYX", qb[:, 3:6]))(np.clip(ta, tb[0], tb[-1]))
    position = float(np.max(np.linalg.norm(qa[:, :3]-pos_b, axis=1)))
    angle = float(np.rad2deg(np.max((rot_a.inv()*rot_b).magnitude())))
    result.update(max_position_difference_m=position, max_orientation_difference_deg=angle,
                  samples_compared=int(len(ta)), duration_s=MISSION_DURATION)
    _bounded(checks, "position_difference", position, position_limit, "m")
    _bounded(checks, "orientation_difference", angle, orientation_limit_deg, "deg")
    result["passed"] = all(c["passed"] for c in checks.values())
    return result


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def aggregate(output_dir, baseline_dir, preflight, video_metadata=None):
    """Read current run artifacts and return a full, fail-closed report.

    preflight must include ``passed: true`` and the current ``run_id``. Each
    required case JSON and NPZ must carry that run_id. JSON must also declare
    completion, the native-body mass/COM audit, and no later state assignment.
    ``video_requested`` controls only
    combined run status, never the independent physics acceptance status.
    """
    output = Path(output_dir)
    baseline = Path(baseline_dir)
    checks, cases, datasets, comparisons = {}, {}, {}, {}
    model_audits = {}
    run_id = preflight.get("run_id")
    _check(checks, "preflight", preflight.get("passed") is True)
    _check(checks, "run_identity", isinstance(run_id, str) and bool(run_id))
    cmg_path = Path(__file__).resolve().parents[1]/"data"/"stewart.cmg.json"
    cmg = json.loads(cmg_path.read_text(encoding="utf-8"))
    for name in REQUIRED_CASES + ("zero_actuation",):
        try:
            failure_path=output/f"{name}.failure.json"
            if failure_path.exists():
                failure=json.loads(failure_path.read_text(encoding="utf-8"))
                raise ValueError("Worker failure: "+str(failure.get("error","unspecified")))
            saved = json.loads((output/f"{name}.json").read_text(encoding="utf-8"))
            data = _load_npz(output/f"{name}.npz")
            case_cmg = json.loads(json.dumps(cmg))
            if name == "heavy_payload":
                for body in case_cmg["bodies"]:
                    if body["id"] == "payload":
                        body["mass_kg"] = 14.0
            result = score_case(data, case_cmg, name, saved["timestep_s"], saved["duration_s"], saved["feedforward"])
            current = bool(run_id) and saved.get("run_id") == run_id
            _check(result["checks"], "current_run", current)
            _check(result["checks"], "engine_identity", saved.get("engine") == "Isaac Sim / PhysX")
            array_run_id = np.asarray(data.get("run_id", []))
            array_current = array_run_id.shape == () and array_run_id.dtype.kind == "U" and bool(run_id) and array_run_id.item() == run_id
            _check(result["checks"], "array_current_run", array_current,
                   detail="The NPZ scalar run_id must match this run's JSON and preflight identity.")
            _check(result["checks"], "engine_run_completed", saved.get("completed") is True)
            _check(result["checks"], "no_later_state_assignment", saved.get("physics_state_assignment_after_initialization") is False,
                   detail="Only the initial assembled state may be assigned; subsequent motion must be integrated by PhysX.")
            for field, limit, unit in (("native_mass_max_error_kg", 1e-5, "kg"), ("native_com_max_error_m", 1e-6, "m"), ("native_inertia_max_error_kg_m2",1e-6,"kg m2")):
                value = saved.get(field)
                finite = type(value) in (int, float) and np.isfinite(value)
                _check(result["checks"], field, finite and 0 <= value <= limit,
                       float(value) if finite else None, limit, unit)
                result[field] = float(value) if finite else None
            gravity=np.asarray(saved.get("native_gravity_m_s2",[]),dtype=float)
            _check(result["checks"], "native_gravity", gravity.shape==(3,) and np.all(np.isfinite(gravity))
                   and np.allclose(gravity,case_cmg["gravity_m_s2"],rtol=0,atol=1e-6),
                   detail="Native PhysX gravity must match the declared CMG.")
            body_count = saved.get("native_body_count")
            _check(result["checks"], "native_body_count", type(body_count) is int and body_count == 19,
                   body_count if type(body_count) is int else None, 19, "bodies")
            result["native_body_count"] = body_count if type(body_count) is int else None
            result["completed"] = saved.get("completed") is True
            result["physics_state_assignment_after_initialization"] = saved.get("physics_state_assignment_after_initialization")
            result["passed"] = all(c["passed"] for c in result["checks"].values())
            result["status"] = "PASS" if result["passed"] else "FAIL"
            result["run_id"] = saved.get("run_id")
            cases[name] = result
            datasets[name] = data
            if name in REQUIRED_CASES and result.get("eligible_for_mission_acceptance"):
                from .model_checks import audit_case, physical_case_cmg
                model_cmg=physical_case_cmg(cmg_path.parents[1],name)
                with np.load(baseline/"reference.npz",allow_pickle=False) as ref:
                    seed=ref["q"][0]
                model_audits[name]=audit_case(data,model_cmg,seed)
                (output/f"{name}.model_validation.json").write_text(
                    json.dumps(model_audits[name],indent=2,allow_nan=False)+"\n",encoding="utf-8")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            cases[name] = {"name": name, "passed": False, "status": "UNVERIFIED", "error": str(exc), "checks": {}}
        _check(checks, f"case.{name}", cases[name]["passed"])
    specs = [
        ("timestep_refinement", "fine", LIMITS["refinement_position_m"], LIMITS["refinement_orientation_deg"]),
        ("isaac_vs_mujoco", "baseline_nominal", LIMITS["cross_engine_position_m"], LIMITS["cross_engine_orientation_deg"]),
        ("isaac_vs_pacdm", "baseline_pacdm", LIMITS["cross_engine_position_m"], LIMITS["cross_engine_orientation_deg"]),
    ]
    for key, counterpart, plimit, alimit in specs:
        try:
            other = _load_npz(baseline/("nominal.npz" if counterpart == "baseline_nominal" else "pacdm.npz")) if counterpart.startswith("baseline_") else datasets[counterpart]
            comparisons[key] = compare_trajectories(datasets["nominal"], other, plimit, alimit, key)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            comparisons[key] = {"name": key, "passed": False, "error": str(exc), "checks": {}}
        _check(checks, key, comparisons[key]["passed"])
    nominal_rms = cases["nominal"].get("rms_position_error_m")
    ablation_rms = cases["no_feedforward"].get("rms_position_error_m")
    ratio = nominal_rms/ablation_rms if nominal_rms is not None and ablation_rms is not None and ablation_rms > 0 else None
    # Strictly lower, so identical/no-effect feedforward does not establish benefit.
    _check(checks, "feedforward_ablation", ratio is not None and np.isfinite(ratio) and ratio < 1,
           ratio, 1.0, "nominal/no-feedforward RMS", "Nominal RMS must be strictly lower. This is force-feedforward ablation, not a PACDM solver ablation.")
    physics_passed = all(c["passed"] for c in checks.values())
    model_validation_passed=all(model_audits.get(name,{}).get("passed") is True for name in REQUIRED_CASES)
    simulation_validation_passed=physics_passed and model_validation_passed
    video = dict(video_metadata or {})
    video_requested = bool(preflight.get("video_requested", video.get("requested", video_metadata is not None)))
    video_checks = {}
    if video_requested:
        _check(video_checks, "writer_verified", video.get("passed") is True)
        _check(video_checks, "current_run", bool(run_id) and video.get("run_id") == run_id)
        video_path = Path(str(video.get("path", "")))
        if not video_path.is_absolute():
            video_path = output/video_path
        _check(video_checks, "file_available", video_path.is_file() and video_path.stat().st_size > 0)
        _check(video_checks, "live_physics_source", video.get("source") == "live PhysX physics")
    video_passed = bool(video_checks) and all(c["passed"] for c in video_checks.values())
    evidence_checks = ("engine_identity", "current_run", "array_current_run", "engine_run_completed", "engine_clock_matches_samples")
    new_isaac_evidence = any(all(case.get("checks", {}).get(key, {}).get("passed") for key in evidence_checks) for case in cases.values())
    report = {
        "schema": "stewart.isaac.validation/2.0", "run_id": run_id,
        "model_validation_passed": model_validation_passed,
        "simulation_validation_passed": simulation_validation_passed,
        "model_audits": model_audits,
        "physics_passed": physics_passed,
        "physics_status": "PASS" if physics_passed else ("UNVERIFIED" if not datasets else "FAIL"),
        "video_requested": video_requested, "video_passed": video_passed,
        "passed": simulation_validation_passed and (video_passed or not video_requested),
        "status": "PASS" if simulation_validation_passed and (video_passed or not video_requested) else ("UNVERIFIED" if not datasets else "FAIL"),
        "checks": checks, "cases": cases, "comparisons": comparisons,
        "preflight": preflight, "video": video, "video_checks": video_checks, "thresholds": LIMITS,
        "required_duration_s": MISSION_DURATION,
        "scope": "Scoped simulation verification of the idealized rigid Stewart CMG/PACDM implementation: full-mission physics gates plus sampled fresh assembly, tangent velocity and Newton-Euler force balance. Hardware validation, general CMG compiler validation and global workspace certification are outside this scope.",
        "source": {"status": "ARCHIVED_MUJOCO_AND_PACDM_BASELINES", "new_isaac_evidence": bool(new_isaac_evidence),
                   "note": "Local preflight is a prerequisite only; it cannot prove Isaac Sim physics acceptance."},
        "interpretation": "Graphics do not establish physics acceptance. The recorded MuJoCo and PACDM runs are archived comparison evidence; they are not new Isaac Sim results.",
    }
    return report


def _format(value):
    if value is None:
        return "not available"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return html.escape(str(value))


def _check_table(checks):
    rows = []
    for name, item in checks.items():
        rows.append(f"<tr><td>{html.escape(name)}</td><td class={'ok' if item['passed'] else 'bad'}>{'PASS' if item['passed'] else 'FAIL'}</td><td>{_format(item.get('value'))}</td><td>{_format(item.get('limit'))}</td><td>{html.escape(item.get('unit', ''))}</td><td>{html.escape(item.get('detail', ''))}</td></tr>")
    return "<table><thead><tr><th>Check</th><th>Status</th><th>Measured</th><th>Limit</th><th>Units</th><th>Notes</th></tr></thead><tbody>"+"".join(rows)+"</tbody></table>"


def _trace_svg(data):
    t = np.asarray(data["time"]); y = 1000*np.asarray(data["pose_error_m"])
    if t.ndim != 1 or y.shape != t.shape or len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        return "<p>Trace unavailable: invalid data.</p>"
    indices = np.unique(np.linspace(0, len(t)-1, min(len(t), 1500), dtype=int))
    ymax = max(10.0, float(np.max(y))*1.1)
    points = " ".join(f"{50+900*t[i]/MISSION_DURATION:.2f},{220-180*y[i]/ymax:.2f}" for i in indices)
    threshold_y = 220-180*10/ymax
    return f'<svg viewBox="0 0 980 260" role="img" aria-label="Measured platform position error in millimetres over time"><rect width="980" height="260" fill="#fff"/><path d="M50 20V220H950" fill="none" stroke="#444"/><path d="M50 {threshold_y:g}H950" stroke="#c23" stroke-dasharray="5 5"/><text x="55" y="18">Position error (mm), peak limit 10 mm</text><text x="4" y="45">{ymax:.2g}</text><text x="22" y="224">0</text><polyline points="{points}" stroke="#18718f" stroke-width="2" fill="none"/><text x="50" y="244">0 s</text><text x="906" y="244">22 s</text></svg>'


def build_report(output_dir, report):
    """Write a standalone HTML report with inline traces; return its path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    title = "Stewart CMG/PACDM · Isaac Sim validation"
    pieces = [f"<h1>{title}</h1>",
        f"<p class=lead>Physics: <strong>{html.escape(report.get('physics_status', 'UNVERIFIED'))}</strong> · Run status: <strong>{html.escape(report.get('status', 'UNVERIFIED'))}</strong></p>",
        f"<p>Run: <code>{html.escape(str(report.get('run_id') or 'not executed'))}</code>. Required mission: 22 seconds for each of four cases.</p>",
        f"<p>{html.escape(report.get('scope', ''))}</p>",
        "<p>PASS requires complete engine trajectories, timestep refinement, both archived reference comparisons, force-feedforward benefit, and all physical tolerances. Missing data, short smoke tests and nonfinite values cannot pass. A video demonstrates appearance and motion; it does not validate dynamics.</p>",
        "<h2>Acceptance</h2>", _check_table(report.get("checks", {})),
        f"<h2>CMG/PACDM simulation decision: {'PASS' if report.get('simulation_validation_passed') else 'NOT ESTABLISHED'}</h2>",
        "<p>The physics score alone is insufficient for model acceptance. Model checks are recomputed from raw body states and held actuator forces; both groups must pass.</p>"]
    for failure in report.get("process_failures",[]):
        pieces.append("<p class=bad>Worker failure: "+html.escape(str(failure))+"</p>")
    for name,audit in report.get("model_audits",{}).items():
        pieces.append(f"<h2>{html.escape(name)} model checks: {html.escape(audit.get('status','UNVERIFIED'))}</h2>")
        if audit.get("error"):pieces.append("<p class=bad>"+html.escape(audit["error"])+"</p>")
        pieces.append(_check_table(audit.get("checks",{})))
    for name, case in report.get("cases", {}).items():
        pieces.append(f"<h2>{html.escape(name)} · {html.escape(case.get('status', 'UNVERIFIED'))}</h2>")
        if case.get("error"):
            pieces.append(f"<p class=bad>{html.escape(case['error'])}</p>")
        pieces.append(_check_table(case.get("checks", {})))
        try:
            pieces.append(_trace_svg(_load_npz(output/f"{name}.npz")))
        except (OSError, ValueError, KeyError):
            pass
    for name, comp in report.get("comparisons", {}).items():
        pieces.append(f"<h2>{html.escape(name)}</h2>")
        pieces.append(_check_table(comp.get("checks", {})))
        if comp.get("error"):
            pieces.append(f"<p class=bad>{html.escape(comp['error'])}</p>")
    pieces.append("<h2>Video</h2><p>" + ("Available and verified for this run." if report.get("video_passed") else "No verified video for this run.") + " Physics acceptance is reported independently above.</p>")
    pieces.append(_check_table(report.get("video_checks", {})))
    pieces.append("<details><summary>Full machine-readable report</summary><pre>"+html.escape(json.dumps(report, indent=2, allow_nan=False))+"</pre></details>")
    path = output/"report.html"
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><style>body{font-family:system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;color:#172638;background:#f5f7fa}h1{font-size:30px}h2{margin-top:36px}p{line-height:1.55}.lead{font-size:22px}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{text-align:left;border-bottom:1px solid #dce3ea;padding:9px;vertical-align:top}th{background:#e7edf4}.ok{color:#167442;font-weight:700}.bad{color:#b52131;font-weight:700}svg{width:100%;margin-top:16px}pre{white-space:pre-wrap;background:white;padding:20px;font-size:12px}code{word-break:break-all}</style><body>'+"".join(pieces)+"</body></html>", encoding="utf-8")
    return path
