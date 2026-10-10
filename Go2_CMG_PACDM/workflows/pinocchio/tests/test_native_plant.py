"""Optional native plant checks in the pinned Pinocchio environment."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE_AVAILABLE = all(importlib.util.find_spec(name) is not None
                       for name in ('numpy', 'scipy', 'pinocchio', 'osqp'))


@unittest.skipUnless(NATIVE_AVAILABLE, 'Requires the pinned Pinocchio environment')
class NativePlantTests(unittest.TestCase):
    def test_local_contact_law_and_import_isolation(self):
        # Run the actual 50-check plant audit in a fresh import-isolated process.
        # Direct output to a temporary directory to preserve recorded results.
        code = '''
from pathlib import Path
import json, sys, tempfile
from run_pinocchio import NoMuJoCo
sys.meta_path.insert(0, NoMuJoCo())
from go2.pin_law_validation import validate_contact_law
with tempfile.TemporaryDirectory(prefix="go2_plant_test_") as tmp:
    result = validate_contact_law(Path.cwd(), output=Path(tmp)/"law.json")
    assert result["passed"], json.dumps(result["checks"])
    assert result["total_checks"] == 50
    assert result["passed_checks"] == 50
    assert "mujoco" not in sys.modules
    print("Plant/contact audit: 50/50; MuJoCo import isolation: PASS")
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Plant/contact audit: 50/50', result.stdout)


if __name__ == '__main__':
    unittest.main()
