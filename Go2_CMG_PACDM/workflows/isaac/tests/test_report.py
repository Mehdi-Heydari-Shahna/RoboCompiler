"""Unit tests use synthetic arrays only; they are not simulator evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from isaac_validation.report import CASES, CASE_PARAMETERS, LIMITS, main, report_suite, summarize_case


def make_case(root, name="nominal", *, bad_height=False, duration=26.0, run_id="test-only"):
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    stop = 0.1 if name == "no_actuation" else duration
    time = np.linspace(0, stop, round(stop * 100) + 1)
    n = len(time)
    q = np.zeros((n, 18))
    q[:, 0] = time / 26.0
    q[:, 2] = 0.31
    q_ref = q.copy()
    if name == "no_actuation" or bad_height:
        q[-1, 2] = 0.17
    support = np.full((n, 4), 20.0)
    support[n // 2:, 2:] = 0.0
    arrays = dict(time=time, q=q, q_ref=q_ref, v=np.zeros((n, 18)),
                  torque=np.zeros((n, 12)), passive_torque=np.zeros((n, 12)),
                  feet=np.zeros((n, 4, 3)), cmg_feet=np.zeros((n, 4, 3)),
                  foot_ref=np.zeros((n, 4, 3)), stance=np.ones((n, 4)),
                  predicted_force=np.zeros((n, 4, 3)), measured_support_force=support,
                  qp_violation=np.zeros(n), push=np.zeros((n, 3)))
    scale = CASE_PARAMETERS[name]["push"]
    arrays["push"][(time >= 1.2 - 1e-10) & (time < 1.35 - 1e-10), 1] = 32.0 * scale
    arrays["push"][(time >= 24.3 - 1e-10) & (time < 24.5 - 1e-10), 0] = -25.0 * scale
    metadata = dict(name=name, run_id=run_id, engine="Isaac Sim / PhysX", source_sha256={"fixture": "synthetic-not-evidence"},
                    started_utc="2026-01-01T10:00:00+00:00", finished_utc="2026-01-01T10:01:00+00:00",
                    completed=name != "no_actuation", simulated_s=stop, duration_s=duration, failure=None,
                    termination_reason="fall" if name == "no_actuation" else "duration",
                    max_qp_violation=0.0, peak_torque_limit_fraction=0.0, min_joint_margin_rad=0.1,
                    min_base_height_m=float(q[:, 2].min()), max_tilt_rad=0.0,
                    unexpected_contact_instances=0, contact_report_available=True,
                    support_force_definition="PhysX contact normal impulse magnitude / dt",
                    dt_s=0.0005 if name == "fine" else 0.001,
                    params=CASE_PARAMETERS[name],
                    external_push_impulse_Ns=[0., 32. * scale * max(0., min(stop, 1.35) - 1.2), 0.],
                    recovery_push_impulse_Ns=[-25. * scale * max(0., min(stop, 24.5) - 24.3), 0., 0.],
                    video={"path": "movie.mp4", "frames": round(stop * 25), "fps": 25,
                           "duration_s": round(stop * 25) / 25, "valid": True, "encoded": True,
                           "unique_frames": round(stop * 25), "provenance": "synthetic test fixture"})
    metadata['execution_exit_code'] = 0
    # Explicitly synthetic parent observations. Native evidence is never
    # inferred from the child metadata in either tests or real reports.
    manifest_path = Path(root)/'suite_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        schema_version=1, run_id=run_id, duration_s=duration, video_requested=True,
        requested_cases=[], cases=[], status='executed')
    if name not in manifest['requested_cases']:
        manifest['requested_cases'].append(name)
    manifest['cases'] = [item for item in manifest['cases'] if item['case'] != name]
    manifest['cases'].append(dict(case=name, status='completed', process_exit_code=0,
        exit_code=0, execution_errors=[], finished_utc='2026-01-01T10:02:00+00:00'))
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    (Path(root)/'logs').mkdir(exist_ok=True)
    (Path(root)/'logs'/f'{name}.log').write_text('Synthetic clean process log, unit test only\n')
    (path / "movie.mp4").write_bytes(b"synthetic-video-fixture-not-a-real-movie")
    scene = dict(friction=CASE_PARAMETERS[name]["friction"], point_payload_kg=CASE_PARAMETERS[name]["payload"],
                 terrain=True, source_counts=dict(rigid_bodies=13, revolute_joints=12, foot_colliders=4))
    (path / "scene_metadata.json").write_text(json.dumps(scene), encoding="utf-8")
    (path / "case.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez_compressed(path / "trajectory.npz", **arrays)
    return path, metadata, arrays


def save(path, metadata, arrays):
    (path / "case.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez_compressed(path / "trajectory.npz", **arrays)


def make_preflight(root, run_id="test-only"):
    record = dict(run_id=run_id, source_sha256={"fixture": "synthetic-not-evidence"},
                  passed=True, checks={"synthetic_check": {"passed": True}})
    for name in ("mechanics.json", "reference_validation.json"):
        (Path(root) / name).write_text(json.dumps(record), encoding="utf-8")


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.plotter = patch("isaac_validation.report._plots", return_value=([], None))
        self.plotter.start()

    def tearDown(self):
        self.plotter.stop()
        self.scratch.cleanup()

    def test_missing_evidence_never_passes(self):
        result = summarize_case(self.root / "no_actuation")
        self.assertFalse(result["expected_outcome_passed"])
        self.assertEqual(result["status"], "UNEXECUTED")
        result = report_suite(self.root, CASES)
        self.assertFalse(result["full_validation"])
        self.assertEqual(result["status"], "UNEXECUTED")
        self.assertIn("has not been executed", (self.root / "report.html").read_text())

    def test_good_nominal_metrics(self):
        path, _, _ = make_case(self.root)
        result = summarize_case(path)
        self.assertTrue(result["mission_passed"])
        self.assertEqual(result["metrics"]["final_position_error_m"], 0.0)
        self.assertTrue((path / "summary.json").is_file())

    def test_nonfinite_and_missing_measurements_fail(self):
        path, meta, arrays = make_case(self.root)
        arrays["q"][3, 0] = np.nan
        save(path, meta, arrays)
        result = summarize_case(path)
        self.assertFalse(result["passed"])
        self.assertIn("q", " ".join(result["problems"]))
        arrays["q"][3, 0] = 0.0
        del arrays["measured_support_force"]
        save(path, meta, arrays)
        self.assertFalse(summarize_case(path)["evidence_passed"])

    def test_independent_foot_fk_required(self):
        path, meta, arrays = make_case(self.root)
        arrays["feet"][-1, 0, 2] = 2.1e-5
        save(path, meta, arrays)
        result = summarize_case(path)
        self.assertFalse(result["passed"])
        self.assertFalse(result["evidence_checks"]["independent_PhysX_CMG_foot_FK"]["passed"])

    def test_per_step_extrema_cannot_hide_logged_failure(self):
        path, meta, arrays = make_case(self.root, bad_height=True)
        meta["min_base_height_m"] = 0.31
        save(path, meta, arrays)
        result = summarize_case(path)
        self.assertFalse(result["evidence_checks"]["per_step_consistency_min_base_height_m"]["passed"])
        self.assertFalse(result["passed"])

    def test_negative_control_requires_physical_zero_torque_failure(self):
        path, meta, arrays = make_case(self.root, "no_actuation")
        result = summarize_case(path)
        self.assertFalse(result["mission_passed"])
        self.assertTrue(result["expected_outcome_passed"])
        arrays["torque"][0, 0] = 1.0
        meta["peak_torque_limit_fraction"] = 1.0 / 23.7
        save(path, meta, arrays)
        self.assertFalse(summarize_case(path)["expected_outcome_passed"])
        arrays["torque"][:] = 0.0
        arrays["q"][:, 2] = 0.31
        meta["min_base_height_m"] = 0.31
        save(path, meta, arrays)
        self.assertFalse(summarize_case(path)["expected_outcome_passed"])

    def test_pd_ablation_is_descriptive(self):
        path, meta, arrays = make_case(self.root, "PD_ablation")
        result = summarize_case(path)
        self.assertTrue(result["mission_passed"])
        self.assertTrue(result["expected_outcome_passed"])
        meta.update(termination_reason="exception", completed=False, failure="Simulator connection lost")
        save(path, meta, arrays)
        self.assertFalse(summarize_case(path)["expected_outcome_passed"])

    def test_partial_never_full_pass(self):
        make_case(self.root)
        make_preflight(self.root)
        result = report_suite(self.root, ["nominal"])
        self.assertTrue(result["scoped_passed"])
        self.assertFalse(result["full_validation"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "PARTIAL")

    def test_full_requires_every_gate_and_video(self):
        for name in CASES:
            make_case(self.root, name)
        make_preflight(self.root)
        result = report_suite(self.root, CASES)
        self.assertTrue(result["full_validation"], json.dumps(result["checks"]))
        (self.root / "nominal" / "movie.mp4").unlink()
        result = report_suite(self.root, CASES)
        self.assertFalse(result["full_validation"])
        self.assertFalse(result["checks"]["nominal_video_evidence"]["passed"])
        self.assertTrue(result["cases"]["nominal"]["mission_passed"])

    def test_timestep_refinement_detects_position_and_orientation(self):
        for name in CASES:
            path, meta, arrays = make_case(self.root, name)
            if name == "fine":
                arrays["q"][:, 0] += 0.011
                arrays["q_ref"][:, 0] += 0.011
                save(path, meta, arrays)
        make_preflight(self.root)
        result = report_suite(self.root, CASES)
        self.assertTrue(result["cases"]["fine"]["mission_passed"])
        self.assertFalse(result["refinement"]["passed"])
        self.assertFalse(result["refinement"]["checks"]["refinement_final_position_m"]["passed"])
        path, meta, arrays = make_case(self.root, "fine")
        arrays["q"][:, 3] += 0.11
        arrays["q_ref"][:, 3] += 0.11
        save(path, meta, arrays)
        result = report_suite(self.root, CASES)
        self.assertFalse(result["refinement"]["checks"]["refinement_max_orientation_rad"]["passed"])

    def test_mixed_runs_and_stale_preflight_fail(self):
        for name in CASES:
            make_case(self.root, name, run_id="other" if name == "payload" else "test-only")
        make_preflight(self.root)
        result = report_suite(self.root, CASES)
        self.assertFalse(result["checks"]["same_run_identity"]["passed"])
        make_case(self.root, "payload")
        make_preflight(self.root, run_id="old-run")
        result = report_suite(self.root, CASES)
        self.assertFalse(result["checks"]["mechanics_preflight"]["passed"])

    def test_short_smoke_is_not_full_mission(self):
        path, _, _ = make_case(self.root, duration=1.0)
        result = summarize_case(path)
        self.assertFalse(result["mission_passed"])
        self.assertFalse(result["evidence_checks"]["full_mission_requested"]["passed"])

    def test_empty_success_preflight_not_accepted(self):
        make_case(self.root)
        make_preflight(self.root)
        for name in ("mechanics.json", "reference_validation.json"):
            path = self.root / name
            value = json.loads(path.read_text())
            value["checks"] = {}
            path.write_text(json.dumps(value))
        result = report_suite(self.root, ["nominal"])
        self.assertFalse(result["scoped_passed"])

    def test_video_requires_full_capture_evidence(self):
        path, meta, arrays = make_case(self.root)
        meta["video"]["frames"] = 25
        meta["video"]["duration_s"] = 1.0
        save(path, meta, arrays)
        result = summarize_case(path)
        self.assertTrue(result["mission_passed"])
        self.assertFalse(result["video"]["passed"])
        meta["video"].update(frames=650, duration_s=26.0, encoded=False)
        save(path, meta, arrays)
        self.assertFalse(summarize_case(path)["video"]["passed"])

    def test_wrong_perturbation_or_disturbance_is_rejected(self):
        path, meta, arrays = make_case(self.root, "strong_push")
        meta["external_push_impulse_Ns"] = [0., 4.8, 0.]
        save(path, meta, arrays)
        result = summarize_case(path)
        self.assertFalse(result["evidence_checks"]["external_push_impulse_Ns"]["passed"])
        path, meta, arrays = make_case(self.root, "low_friction")
        scene = json.loads((path / "scene_metadata.json").read_text())
        scene["friction"] = 0.8
        (path / "scene_metadata.json").write_text(json.dumps(scene))
        self.assertFalse(summarize_case(path)["evidence_checks"]["scene_friction"]["passed"])

    def test_smoke_cli_success_does_not_claim_mission_validation(self):
        make_case(self.root, duration=2.0)
        make_preflight(self.root)
        with patch("builtins.print"):
            code = main(["--output", str(self.root), "--suite", "smoke", "--run-id", "test-only"])
        self.assertEqual(code, 0)
        result = json.loads((self.root / "validation.json").read_text())
        self.assertTrue(result["smoke_passed"])
        self.assertFalse(result["full_validation"])
        self.assertFalse(result["cases"]["nominal"]["mission_passed"])
        self.assertEqual(result["status"], "SMOKE_PASS")

    def test_cli_rejects_wrong_run_identity(self):
        make_case(self.root)
        make_preflight(self.root)
        with patch("builtins.print"):
            code = main(["--output", str(self.root), "--suite", "nominal", "--run-id", "another-run"])
        self.assertEqual(code, 1)
        result = json.loads((self.root / "validation.json").read_text())
        self.assertFalse(result["scoped_passed"])

    def test_declared_source_thresholds(self):
        self.assertEqual(LIMITS["final_position_error_m"], 0.025)
        self.assertEqual(LIMITS["rms_body_error_m"], 0.035)
        self.assertEqual(LIMITS["max_tilt_rad"], 0.25)
        self.assertEqual(LIMITS["refinement_max_position_m"], 0.025)
        self.assertEqual(LIMITS["refinement_final_position_m"], 0.01)


if __name__ == "__main__":
    unittest.main()
