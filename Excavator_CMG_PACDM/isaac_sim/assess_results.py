#!/usr/bin/env python3
"""Independently assess an Isaac soil_final result against its MuJoCo reference.

This reads evidence only; it never starts physics or changes a result directory.
Use ``python assess_results.py --result PATH [--output comparison.json]``.
An incomplete smoke run is reported as such, not as a digging success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parent
EXPECTED_PHASES = (
    "settle soil", "lower below surface", "draw through soil", "curl bucket",
    "lift clear", "hold load", "slew to receiver", "dump material",
    "wait for deposition", "close bucket", "return",
)
# These are declared by launch.py's full-mission gate; the source mission
# itself requires 1 kg in the bucket at lift and 1 kg deposited.
ARM_CLOSURE_LIMIT_M = 1e-4
TRACK_CLOSURE_LIMIT_M = 1e-3
REQUIRED_PAYLOAD_KG = 1.0


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def read_json(path, issues, *, mandatory=False):
    if not path.is_file():
        if mandatory:
            issues.append("Missing " + path.name)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (OSError, ValueError) as exc:
        issues.append("Unreadable %s: %s" % (path.name, exc))
        return None


def strictly_increasing(values):
    return bool(values.ndim == 1 and values.size > 0 and
                np.all(np.isfinite(values)) and np.all(np.diff(values) > 0))


def inspect_npz(path, fields, issues):
    """Inspect numerical arrays without retaining the large wrench arrays."""
    outcome = {"available": False, "fields": {}}
    if not path.is_file():
        issues.append("Missing " + path.name)
        return outcome, None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            outcome["available"] = True
            for field in fields:
                if field not in arrays:
                    issues.append("%s lacks %s" % (path.name, field))
                    continue
                array = arrays[field]
                numeric = np.issubdtype(array.dtype, np.number)
                finite = bool(np.all(np.isfinite(array))) if numeric else False
                outcome["fields"][field] = {"shape": list(array.shape), "finite": finite}
                if not numeric or not finite:
                    issues.append("%s has nonfinite or nonnumeric %s" % (path.name, field))
            if "time" in arrays:
                times = np.asarray(arrays["time"], dtype=float)
                outcome["time_count"] = int(times.size)
                outcome["first_time_s"] = finite_number(times[0]) if times.size else None
                outcome["last_time_s"] = finite_number(times[-1]) if times.size else None
                outcome["time_strictly_increasing"] = strictly_increasing(times)
                if not outcome["time_strictly_increasing"]:
                    issues.append(path.name + " time is nonfinite, empty or not strictly increasing")
            else:
                times = None
        return outcome, times
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        issues.append("Unreadable %s: %s" % (path.name, exc))
        return outcome, None


def inspect_bridge(path, issues):
    outcome = {"available": False, "sampled_rows": 0, "terminal_record": False}
    samples = []
    if not path.is_file():
        issues.append("Missing bridge_diagnostics.jsonl")
        return outcome, samples
    try:
        with path.open(encoding="utf-8") as lines:
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                record = json.loads(line,
                    parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                if not isinstance(record, dict):
                    raise ValueError("Record %d is not an object" % line_number)
                response = record.get("bridge_response") or {}
                if record.get("terminal") is True:
                    outcome["terminal_record"] = True
                    outcome["terminal_time_s"] = finite_number(record.get("t"))
                    metrics = response.get("last_metrics") or {}
                else:
                    if outcome["terminal_record"]:
                        raise ValueError("Control record after terminal record")
                    outcome["sampled_rows"] += 1
                    metrics = response.get("metrics") or {}
                timestamp = finite_number(metrics.get("time_s"))
                lip = metrics.get("lip_position_m")
                if timestamp is not None and isinstance(lip, list) and len(lip) == 3:
                    lip = [finite_number(x) for x in lip]
                    if all(x is not None for x in lip):
                        samples.append((timestamp, lip))
        outcome["available"] = True
        if not outcome["terminal_record"]:
            issues.append("Bridge diagnostics lacks terminal controller record")
    except (OSError, ValueError, TypeError) as exc:
        issues.append("Invalid bridge diagnostics: " + str(exc))
    outcome["lip_metric_observations"] = len(samples)
    return outcome, samples


def compare_lip(samples, reference_path, issues):
    outcome = {"available": False, "meaning": "Measured Isaac bridge lip position minus time-interpolated MuJoCo reference; distinct physics engines and soil contacts can diverge."}
    if not reference_path.is_file():
        issues.append("Missing reference trajectory")
        return outcome
    try:
        with np.load(reference_path, allow_pickle=False) as reference:
            t_ref = np.asarray(reference["time"], dtype=float)
            lip_ref = np.asarray(reference["lip_position"], dtype=float)
        if not strictly_increasing(t_ref) or lip_ref.shape != (len(t_ref), 3) or not np.all(np.isfinite(lip_ref)):
            raise ValueError("Reference lip trajectory invalid")
        outcome["reference_duration_s"] = float(t_ref[-1])
        # Repeated controller diagnostics carry the last 40 ms soil observation.
        # Compare once per *observation time*, never treat repeats as new data.
        unique = {t: np.asarray(x) for t, x in samples if t_ref[0] - 1e-9 <= t <= t_ref[-1] + 1e-9}
        if not unique:
            outcome["note"] = "No matching measured lip observations"
            return outcome
        ts = np.asarray(sorted(unique), dtype=float)
        if not strictly_increasing(ts) and len(ts) != 1:
            raise ValueError("Lip times are not ordered")
        observations = np.stack([unique[t] for t in ts])
        interpolated = np.stack([np.interp(ts, t_ref, lip_ref[:, axis]) for axis in range(3)], axis=1)
        residual = observations - interpolated
        lengths = np.linalg.norm(residual, axis=1)
        outcome.update({"available": True, "observation_count": len(ts),
            "overlap_time_s": [float(ts[0]), float(ts[-1])],
            "rmse_euclidean_m": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            "rmse_per_axis_m": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
            "max_euclidean_error_m": float(np.max(lengths)),
            "last_observation_time_s": float(ts[-1]),
            "last_observed_lip_m": observations[-1].tolist(),
            "last_reference_lip_m": interpolated[-1].tolist(),
            "last_error_m": residual[-1].tolist()})
        if ts[0] <= 2.0 + 1e-9 and ts[-1] >= 2.0 - 1e-9:
            if np.any(np.isclose(ts, 2.0, atol=1e-9, rtol=0)):
                at2 = observations[int(np.argmin(np.abs(ts - 2.0)))]
                ref2 = np.array([np.interp(2., t_ref, lip_ref[:, axis]) for axis in range(3)])
                outcome["t2_s"] = {"isaac_lip_m": at2.tolist(), "reference_lip_m": ref2.tolist(),
                    "error_vector_m": (at2-ref2).tolist(), "euclidean_error_m": float(np.linalg.norm(at2-ref2))}
            else:
                outcome["t2_s_note"] = "No observation exactly at 2 seconds; result is not inferred"
        return outcome
    except (OSError, ValueError, KeyError, EOFError) as exc:
        issues.append("Cannot compare reference lip trajectory: " + str(exc))
        return outcome


def assess_video(result, status, launcher, issues):
    requested = bool((launcher or {}).get("video_requested") or (status or {}).get("video_start"))
    capture = (status or {}).get("video_capture") or {}
    meta = read_json(result/"video_frames.json", issues) or {}
    encoding = read_json(result/"video_encoding.json", issues) or {}
    movie = result/"isaac_native.mp4"
    out = {"requested": requested, "metadata_available": bool(meta),
           "frame_count_reported": meta.get("frame_count"),
           "capture_timeline_delta_s": finite_number(meta.get("capture_timeline_delta_s")),
           "png_complete_reported": capture.get("png_complete") is True,
           "mp4_reported_created": encoding.get("mp4_created") is True,
           "mp4_in_result": movie.is_file(),
           "mode": meta.get("mode", "live_native_frame_capture"),
           "scope": meta.get("scope", "frames captured during native execution")}
    frames = meta.get("frames")
    if isinstance(frames, list):
        steps = [f.get("native_step") for f in frames if isinstance(f, dict)]
        times = [finite_number(f.get("sim_time_s")) for f in frames if isinstance(f, dict)]
        out["frames_metadata_consistent"] = bool(
            len(steps) == len(frames) >= 2 and all(type(x) is int for x in steps) and
            all(x is not None for x in times) and
            all(b>a for a,b in zip(steps, steps[1:])) and
            all(b>a for a,b in zip(times, times[1:])) and
            meta.get("frame_count") == len(frames) and
            finite_number(meta.get("capture_timeline_delta_s")) == 0.0)
        if steps:
            out["first_native_step"] = steps[0]
            out["last_native_step"] = steps[-1]
    else:
        out["frames_metadata_consistent"] = False
    png_files = sorted((result/"frames").glob("frame_*.png"))
    out["included_png_count"] = len(png_files)
    out["included_pngs_valid_and_vary"] = False
    if (out["frames_metadata_consistent"] and len(png_files) == len(frames) and
        [path.name for path in png_files] ==
            ["frame_%06d.png" % n for n in range(len(frames))]):
        image_digests = []
        for path in png_files:
            with path.open("rb") as stream:
                content = stream.read()
            if len(content) < 33 or content[:8] != b"\x89PNG\r\n\x1a\n":
                break
            image_digests.append(hashlib.sha256(content).hexdigest())
        out["included_pngs_valid_and_vary"] = (
            len(image_digests) == len(frames) and len(set(image_digests)) >= 2)
    if movie.is_file():
        with movie.open("rb") as stream:
            header = stream.read(32)
        out["mp4_header_valid"] = movie.stat().st_size >= 1024 and b"ftyp" in header
        out["mp4_bytes"] = movie.stat().st_size
    else:
        out["mp4_header_valid"] = False
    replay_ok = True
    if meta.get("mode") == "post_run_measured_state_replay":
        replay_ok = False
        try:
            from replay_video import file_hash
            from render_replay_worker import pose_digest
            source_path = result/"states.npz"
            if meta.get("source_states_sha256") != file_hash(source_path):
                raise ValueError("Replay source hash does not match the saved native states")
            with np.load(source_path, allow_pickle=False) as data:
                times = data["time"]
                pos = data["positions"]
                quat = data["quaternions_wxyz"]
            dt = finite_number((status or {}).get("dt"))
            if dt is None or dt <= 0 or not isinstance(frames, list):
                raise ValueError("Replay timing/provenance unavailable")
            for frame in frames:
                index = frame["state_sample_index"]
                if (type(index) is not int or not 0 <= index < len(times) or
                    not math.isclose(float(times[index]), frame["sim_time_s"], abs_tol=1e-9) or
                    round(float(times[index])/dt) != frame["native_step"] or
                    pose_digest(pos[index], quat[index]) != frame.get("pose_sha256")):
                    raise ValueError("Replay frame does not match an actual measured pose sample")
                path = result/"frames"/("frame_%06d.png" % frame["frame_index"])
                if frame.get("png_sha256") != file_hash(path):
                    raise ValueError("Replay frame hash mismatch")
            replay_ok = True
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            issues.append("Post-run measured-state replay provenance failed: "+str(exc))
    out["replay_provenance_verified"] = replay_ok if meta.get("mode") == "post_run_measured_state_replay" else None
    out["video_evidence_available"] = bool(out["frames_metadata_consistent"] and
        out["png_complete_reported"] and out["mp4_reported_created"] and
        out["mp4_header_valid"] and out["included_pngs_valid_and_vary"] and
        out.get("last_native_step") == (status or {}).get("native_steps_completed") and replay_ok)
    if requested and not out["video_evidence_available"]:
        issues.append("Requested native-state video is missing or cannot be verified from this result")
    return out


def assess(result):
    issues = []
    status = read_json(result/"status.json", issues, mandatory=True) or {}
    launcher = read_json(result/"launcher.json", issues) or {}
    manifest = read_json(result/"manifest.json", issues) or {}
    reference_report = read_json(ROOT/"reference/soil_final_source_report.json", issues, mandatory=True) or {}
    ref_duration = finite_number(reference_report.get("simulated_duration_s"))
    requested = finite_number(status.get("requested_duration")) or finite_number(launcher.get("duration_requested_s"))
    elapsed = finite_number(status.get("simulated_duration"))
    dt = finite_number(status.get("dt")) or finite_number(launcher.get("physics_dt"))
    steps = status.get("native_steps_completed")
    steps_valid = (type(steps) is int and steps > 0 and dt is not None and dt > 0 and
                   elapsed is not None and math.isclose(elapsed, steps*dt, abs_tol=1e-8, rel_tol=0))
    if not steps_valid:
        issues.append("Native step count and elapsed time are missing or inconsistent")
    if status.get("status") not in ("NATIVE_RUN_COMPLETED_UNASSESSED", "NATIVE_MISSION_INCOMPLETE", "NATIVE_MISSION_FAILED"):
        issues.append("Native runner status does not report a completed run")
    # Video/encoding/packaging errors do not change the native physics outcome.
    launcher_exit_ok = (not launcher or launcher.get("native_process_exit_code", 0) == 0)
    if not launcher_exit_ok:
        issues.append("Native physics process reported a nonzero exit code")
    events = status.get("physx_error_events")
    event_stream_valid = isinstance(events, list)
    if not event_stream_valid:
        issues.append("PhysX error event stream is missing")
        events = []
    if events:
        issues.append("PhysX reported %d error events" % len(events))
    native_log_errors = []
    try:
        with (result/"isaac.log").open(encoding="utf-8", errors="replace") as log:
            for line in log:
                if (re.search(r"\[(?:Error|Fatal)\].*(?:physx|PxArticulation|Physics)", line, re.I) or
                    "Articulation link must be in a scene" in line):
                    native_log_errors.append(line.strip()[:300])
    except OSError:
        pass  # The native subscribed error stream is the authoritative check.
    if native_log_errors:
        issues.append("PhysX errors also appear in isaac.log")

    gates = {name: status.get(name) is True for name in (
        "native_runtime_executed", "native_physics_initialized", "initial_rigid_state_validated_at_t0",
        "prestep_articulation_confirmed", "first_controlled_step_articulation_confirmed",
        "physx_start_simulation_called")}
    gates["first_force_error_free_before_step"] = (status.get("first_controlled_step") or {}).get("physx_error_free_before_step") is True
    gates["no_uncontrolled_warmup"] = status.get("uncontrolled_warmup_steps") == 0
    gates["no_post_initialization_pose_writes"] = status.get("state_writes_after_initialization") == 0
    gates["no_mujoco_integration"] = status.get("mujoco_integration_steps") == 0
    if not all(gates.values()):
        issues.append("One or more native control/state initialization gates did not pass")

    state_info, state_times = inspect_npz(result/"states.npz", (
        "time", "positions", "quaternions_wxyz", "linear_velocities_origin_world",
        "angular_velocities_world", "kinetic_energy_J", "potential_energy_J",
        "closure_gaps_m", "gear_errors_rad"), issues)
    wrench_info, wrench_times = inspect_npz(result/"applied_wrenches.npz", (
        "time", "efforts", "forces_world", "torques_world_about_com"), issues)
    n_bodies = len(manifest.get("bodies", [])) if isinstance(manifest.get("bodies"), list) else 0
    state_shape_ok = False
    wrench_shape_ok = False
    closures = manifest.get("closures") or []
    arm_gap = track_gap = None
    arm_status_gap = finite_number(status.get("max_arm_closure_gap_m"))
    track_status_gap = finite_number(status.get("max_track_closure_gap_m"))
    if state_info["available"] and state_times is not None:
        with np.load(result/"states.npz", allow_pickle=False) as state:
            positions = state.get("positions")
            gaps = state.get("closure_gaps_m")
            gear = state.get("gear_errors_rad")
            state_shape_ok = (n_bodies > 0 and positions is not None and
                positions.shape == (state_times.size, n_bodies, 3) and
                "quaternions_wxyz" in state and
                state["quaternions_wxyz"].shape == (state_times.size, n_bodies, 4) and
                gaps is not None and gaps.shape == (state_times.size, len(closures)) and
                gear is not None and gear.shape[0] == state_times.size)
            if state_shape_ok and all(x.get("finite") for x in state_info["fields"].values()):
                # Full-run maxima from status cover every native interval; recorded
                # array values are corroborating samples, never silently promoted.
                sample_peaks = np.max(gaps, axis=0)
                arm_sample = max((float(p) for c,p in zip(closures, sample_peaks)
                                  if not c["name"].startswith("track_")), default=None)
                track_sample = max((float(p) for c,p in zip(closures, sample_peaks)
                                    if c["name"].startswith("track_")), default=None)
                arm_gap = (max(arm_status_gap, arm_sample) if arm_status_gap is not None and arm_sample is not None
                           else arm_status_gap if arm_status_gap is not None else arm_sample)
                track_gap = (max(track_status_gap, track_sample) if track_status_gap is not None and track_sample is not None
                             else track_status_gap if track_status_gap is not None else track_sample)
                state_info["sampled_arm_closure_max_m"] = arm_sample
                state_info["sampled_track_closure_max_m"] = track_sample
                state_info["sampled_gear_max_rad"] = float(np.max(np.abs(gear)))
    if not state_shape_ok:
        issues.append("Native state shape does not match manifest body/closure inventory")
    if wrench_info["available"] and wrench_times is not None:
        shapes = wrench_info["fields"]
        wrench_shape_ok = (n_bodies > 0 and
            shapes.get("efforts",{}).get("shape") == [wrench_times.size, 10] and
            shapes.get("forces_world",{}).get("shape") == [wrench_times.size, n_bodies, 3] and
            shapes.get("torques_world_about_com",{}).get("shape") == [wrench_times.size, n_bodies, 3])
    if not wrench_shape_ok:
        issues.append("Native wrench shape does not match manifest body/actuator inventory")
    state_required = {"time", "positions", "quaternions_wxyz", "linear_velocities_origin_world",
        "angular_velocities_world", "kinetic_energy_J", "potential_energy_J", "closure_gaps_m", "gear_errors_rad"}
    wrench_required = {"time", "efforts", "forces_world", "torques_world_about_com"}
    fields_ok = (state_required.issubset(state_info["fields"]) and
        wrench_required.issubset(wrench_info["fields"]) and
        all(v["finite"] for v in state_info["fields"].values()) and
        all(v["finite"] for v in wrench_info["fields"].values()))
    if not fields_ok:
        issues.append("Native state/wrench fields are missing or not finite")
    time_ok = bool(steps_valid and state_times is not None and wrench_times is not None and
        strictly_increasing(state_times) and strictly_increasing(wrench_times) and
        len(state_times) >= 3 and len(wrench_times) == steps and
        math.isclose(state_times[0], 0., abs_tol=1e-9) and
        math.isclose(state_times[1], dt, abs_tol=1e-9) and
        math.isclose(state_times[-1], elapsed, abs_tol=1e-8) and
        math.isclose(wrench_times[0], 0., abs_tol=1e-9) and
        math.isclose(wrench_times[-1] + dt, elapsed, abs_tol=1e-8) and
        np.allclose(wrench_times, np.arange(steps)*dt, rtol=0., atol=1e-8))
    if not time_ok:
        issues.append("Native state/wrench time axis does not cover every controlled interval")

    bridge_info, lip_samples = inspect_bridge(result/"bridge_diagnostics.jsonl", issues)
    final = status.get("bridge_final_response") or status.get("bridge_summary") or {}
    metrics = final.get("last_metrics") or {}
    phases = metrics.get("events") or []
    observed_names = [x.get("phase") for x in phases if isinstance(x, dict)]
    lifted = finite_number(metrics.get("lifted_mass_kg"))
    deposited = finite_number(metrics.get("deposited_mass_kg"))
    failures = metrics.get("failures")
    if not isinstance(failures, list):
        failures = ["Missing mission failures field"]
    controller = final.get("controller") or (status.get("bridge_summary") or {}).get("controller") or {}
    fallback = controller.get("fallback_count")
    pacdm_bounds = (("pacdm_closure_max", 1e-8),
                    ("pacdm_tangent_max", 1e-8),
                    ("inverse_equilibrium_relative_max", 1e-6))
    pacdm_ok = bool(fallback == 0 and controller.get("native_state_projection") is False and
        all(finite_number(controller.get(name)) is not None and
            finite_number(controller[name]) <= limit for name, limit in pacdm_bounds))
    contact_path = result/"native_contact_forces.npz"
    contact_ok = False
    if contact_path.is_file():
        try:
            with np.load(contact_path, allow_pickle=False) as contact:
                contact_times = np.asarray(contact["time"])
                contact_forces = np.asarray(contact["reported_net_contact_forces_world"])
                contact_ok = bool(strictly_increasing(contact_times) and
                    contact_forces.shape == (len(contact_times), n_bodies, 3) and
                    np.all(np.isfinite(contact_forces)) and
                    np.any(np.abs(contact_forces) > 1e-6))
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
    structural_sample_ok = (arm_gap is not None and track_gap is not None and
        arm_gap <= ARM_CLOSURE_LIMIT_M and track_gap <= TRACK_CLOSURE_LIMIT_M)
    source_phases = reference_report.get("phase_names")
    phases_identical = observed_names == list(EXPECTED_PHASES) and source_phases == list(EXPECTED_PHASES)
    requested_full = bool(status.get("stop_when_mission_done") is True or
        launcher.get("full_mission_requested") is True or
        (requested is not None and ref_duration is not None and requested >= ref_duration))
    scope = "full_mission_attempt" if requested_full else "smoke"
    if requested_full and not contact_ok:
        issues.append("Native contact force readback is missing, invalid or all zero")
    if requested_full and not pacdm_ok:
        issues.append("PACDM residual/fallback/native projection gate did not pass")
    # Full mission certification needs extrema from *every* physics step,
    # while an old smoke archive may only contain subsampled state arrays.
    structural_ok = bool(structural_sample_ok and (not requested_full or
        (arm_status_gap is not None and track_status_gap is not None)))
    bridge_finalized = (final.get("finalized") is True and
        final.get("completed_intervals") == steps and
        finite_number(final.get("final_observation_time_s")) is not None and elapsed is not None and
        math.isclose(final["final_observation_time_s"], elapsed, abs_tol=1e-8))
    physical_evidence_ok = bool(steps_valid and launcher_exit_ok and all(gates.values()) and
        event_stream_valid and not events and
        not native_log_errors and time_ok and state_shape_ok and wrench_shape_ok and fields_ok and
        bridge_info["terminal_record"] and bridge_info["lip_metric_observations"] > 0 and bridge_finalized and
        status.get("status") == "NATIVE_RUN_COMPLETED_UNASSESSED")
    task_success = bool(requested_full and physical_evidence_ok and structural_ok and
        metrics.get("done") is True and not failures and phases_identical and
        lifted is not None and lifted >= REQUIRED_PAYLOAD_KG and
        deposited is not None and deposited >= REQUIRED_PAYLOAD_KG and pacdm_ok and contact_ok)
    comparison = compare_lip(lip_samples, ROOT/"reference/soil_final_source_trajectory.npz", issues)
    video = assess_video(result, status, launcher, issues)
    if requested_full and not task_success:
        issues.append("Full digging task did not meet all completion/soil/closure/PACDM gates")

    audit = final.get("hydraulic_energy_audit") or {}
    # This audit covers hydraulic fluid and shaft energy. Native contact
    # moments/work, all constraint work and dissipation were not measured.
    force_energy = {"full_validation": "NOT_ESTABLISHED",
        "native_force_calls_without_error": bool(not events and not native_log_errors and gates["first_force_error_free_before_step"]),
        "reported_contact_force_readback": contact_ok,
        "native_mechanical_energy_change_J": finite_number(status.get("mechanical_energy_change_J")),
        "applied_wrench_midpoint_work_J": finite_number(status.get("applied_wrench_midpoint_work_J")),
        "unaccounted_work_J": finite_number(status.get("unaccounted_work_J")),
        "fluid_finite_step_defect_J": finite_number(audit.get("fluid_finite_step_defect_J")),
        "hydraulic_native_partition_error_J": finite_number(audit.get("hydraulic_native_partition_error_J")),
        "reason": "Net contact force alone does not provide contact moment/work or constraint work; a full native mechanical energy balance is unavailable."}
    return {"assessment_version": "1.0", "result_directory": str(result),
        "source_reference": {"plant": reference_report.get("plant"),
            "report_status": reference_report.get("status"), "source_duration_s": ref_duration,
            "source_deposited_mass_kg": finite_number(reference_report.get("deposited_mass_kg")),
            "limitations": {"soil_parameter_calibrated": bool(reference_report.get("soil_parameter_calibration")),
                "resolution_timestep_convergence_tested": bool(reference_report.get("resolution_and_timestep_convergence_tested"))}},
        "scope": {"classification": scope, "requested_duration_s": requested,
                  "observed_duration_s": elapsed, "dt_s": dt, "native_steps": steps},
        "partial_progress": {"last_observed_mission_metrics": status.get("last_observed_mission_metrics"),
            "observed_native_time_lower_bound_s": status.get("observed_native_time_lower_bound_s"),
            "observed_native_steps_lower_bound": status.get("observed_native_steps_lower_bound"),
            "step_count_scope": status.get("step_count_scope"),
            "recovery_notes": status.get("recovery_notes", [])},
        "native_physics": {"runner_status": status.get("status"), "prestep_and_force_gates": gates,
            "physx_error_events": len(events), "physx_log_error_count": len(native_log_errors),
            "physx_log_error_examples": native_log_errors[:5],
            "articulation": status.get("native_articulation")},
        "measurements": {"states": state_info, "wrenches": wrench_info, "bridge_diagnostics": bridge_info,
                         "shape_valid": state_shape_ok and wrench_shape_ok, "time_valid": time_ok},
        "structural_closure": {"arm_peak_m": arm_gap, "arm_limit_m": ARM_CLOSURE_LIMIT_M,
            "track_peak_m": track_gap, "track_limit_m": TRACK_CLOSURE_LIMIT_M,
            "arm_peak_scope": "all_native_steps" if arm_status_gap is not None else "recorded_state_samples_only",
            "track_peak_scope": "all_native_steps" if track_status_gap is not None else "recorded_state_samples_only",
            "passes_declared_limits": structural_ok,
            "source_mujoco_peak_m": finite_number((reference_report.get("peak") or {}).get("closure_m"))},
        "mission": {"done": metrics.get("done") is True, "phase": metrics.get("phase_name"),
            "phase_events": phases, "observed_phase_names": observed_names,
            "phase_sequence_complete": phases_identical, "failures": failures,
            "lifted_mass_kg": lifted, "deposited_mass_kg": deposited,
            "required_payload_kg": REQUIRED_PAYLOAD_KG, "task_success": task_success},
        "pacdm_controller": {"method": controller.get("method"), "evaluations": controller.get("evaluations"),
            "fallback_count": fallback,
            "passes_declared_residual_bounds": pacdm_ok,
            "pacdm_closure_max": finite_number(controller.get("pacdm_closure_max")),
            "pacdm_tangent_max": finite_number(controller.get("pacdm_tangent_max")),
            "inverse_equilibrium_relative_max": finite_number(controller.get("inverse_equilibrium_relative_max")),
            "native_state_projection": controller.get("native_state_projection")},
        "native_model_adjustments": {
            "scope": "Recorded for information; neither item is a pass/fail gate.",
            "soil_rolling_resistance": status.get("soil_rolling_resistance"),
            "passive_spin_gauge": (final.get("passive_spin_gauge") or
                                   (status.get("bridge_summary") or {}).get("passive_spin_gauge") or
                                   (status.get("bridge_initialization") or {}).get("passive_spin_gauge"))},
        "lip_vs_source": comparison, "native_video": video, "force_energy": force_energy,
        "conclusions": {"controlled_native_run_evidence_passes": physical_evidence_ok,
            "full_digging_task_passes": task_success,
            "full_force_energy_validation": "NOT_ESTABLISHED",
            "isaac_mujoco_trajectory_equivalence": "NOT_ESTABLISHED",
            "video_evidence_available": video["video_evidence_available"]},
        "issues": list(dict.fromkeys(issues))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="Directory containing extracted Isaac result files")
    parser.add_argument("--output", type=Path, help="Write JSON assessment (default: print to stdout)")
    args = parser.parse_args(argv)
    result = args.result.resolve()
    if not result.is_dir():
        parser.error("--result must be an extracted result directory")
    report = assess(result)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
        print(str(args.output.resolve()))
    else:
        print(serialized, end="")
    return 0 if report["conclusions"]["controlled_native_run_evidence_passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
