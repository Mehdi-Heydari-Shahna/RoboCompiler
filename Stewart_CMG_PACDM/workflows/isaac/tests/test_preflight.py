"""Safety gates relevant to evidence reuse, without requiring Isaac Sim."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from isaac_validation.preflight import run_preflight


ROOT = Path(__file__).resolve().parents[1]


class PreflightEvidenceTests(unittest.TestCase):
    def test_supplied_reference_passes_fresh_numerical_checks(self):
        report = run_preflight(ROOT)
        self.assertTrue(report["passed"], json.dumps(report, indent=2))
        self.assertEqual(report["checks"]["pacdm.samples_completed"]["value"], 25)
        self.assertEqual(report["details"]["optional_mujoco_check"], "Not requested; no MuJoCo imported")

    def test_changed_cmg_fails_before_reusing_cached_forces(self):
        with tempfile.TemporaryDirectory(prefix="stewart_cmg_test_") as directory:
            target = Path(directory)
            for name in ("baseline", "data", "vendor", "stewart"):
                shutil.copytree(ROOT / name, target / name)
            cmg_path = target / "data/stewart.cmg.json"
            cmg = json.loads(cmg_path.read_text())
            next(body for body in cmg["bodies"] if body["id"] == "payload")["mass_kg"] = 9.0
            cmg_path.write_text(json.dumps(cmg))
            report = run_preflight(target)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["hash.data/stewart.cmg.json"]["passed"])
            self.assertNotIn("pacdm.samples_completed", report["checks"])

    def test_changed_comparator_cannot_receive_a_pass(self):
        with tempfile.TemporaryDirectory(prefix="stewart_trace_test_") as directory:
            target = Path(directory)
            for name in ("baseline", "data", "vendor", "stewart"):
                shutil.copytree(ROOT / name, target / name)
            with (target / "baseline/pacdm.npz").open("ab") as stream:
                stream.write(b"invalid comparator evidence")
            report = run_preflight(target)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["hash.baseline/pacdm.npz"]["passed"])


if __name__ == "__main__":
    unittest.main()
