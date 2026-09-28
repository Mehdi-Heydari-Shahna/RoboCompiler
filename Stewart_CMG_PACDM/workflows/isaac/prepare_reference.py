"""Optional regeneration in the separate Miniforge Pinocchio environment.

The Isaac mission deliberately uses the hash-verified supplied baseline.
This command regenerates independent reference evidence in prepared_reference/
and compares its values to that baseline. It never changes mission inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "prepared_reference")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output_dir.resolve()
    if out == root / "baseline" or (root / "baseline") in out.parents:
        parser.error("The supplied baseline is immutable; choose a different output directory")
    try:
        import pinocchio as pin
        import scipy
        from stewart.reference import build_reference
        from isaac_validation.preflight import run_preflight
    except ImportError as error:
        raise SystemExit("Use the separate environment-reference.yml Miniforge environment. "
                         "Do not install Pinocchio into Isaac Sim's Python. Missing dependency: " + str(error))
    preflight = run_preflight(root)
    if not preflight["passed"]:
        raise SystemExit("Preflight failed; the supplied source/model/reference has changed or failed its numerical checks")
    cmg = json.loads((root / "data/stewart.cmg.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    reference_path = out / "reference.npz"
    if reference_path.exists():
        parser.error("Output already contains reference.npz; choose a new output directory")
    statistics = build_reference(cmg, reference_path, interval=.02)
    comparisons = {}
    with np.load(root / "baseline/reference.npz", allow_pickle=False) as old, np.load(reference_path, allow_pickle=False) as new:
        for key in old.files:
            if key == "coordinate_ids":
                comparisons[key] = {"passed": bool(np.array_equal(old[key], new[key]))}
            else:
                # These diagnostic quantities can be sensitive to tied QR row
                # choices while the physical map remains unchanged.
                diagnostic = key in ("selected_block_condition", "closure_residual",
                                     "tangent_residual", "acceleration_residual")
                difference = float(np.max(np.abs(new[key] - old[key])))
                limit = 1e-5 if key in ("feedforward_force", "acceleration", "mapping") else 1e-7
                passed = bool(np.all(np.isfinite(new[key])) and
                              (diagnostic or difference <= limit))
                comparisons[key] = {"passed": passed, "max_abs_difference": difference,
                                    "limit": None if diagnostic else limit,
                                    "diagnostic_only": diagnostic}
    numerical_ok = (statistics["max_closure_residual"] <= 1e-8
                    and statistics["max_tangent_residual"] <= 1e-8
                    and statistics["max_acceleration_residual"] <= 1e-7
                    and statistics["max_target_pose_reconstruction_error"] <= 2e-8
                    and statistics["min_reduced_mass_eigenvalue"] > 1e-8)
    report = {"passed": bool(numerical_ok and all(c["passed"] for c in comparisons.values())),
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Fresh PACDM/Pinocchio reference generation; no Isaac Sim execution. "
                       "Isaac mission continues to use the shipped baseline.",
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "pinocchio": pin.__version__},
              "statistics": statistics, "comparison_with_shipped_reference": comparisons,
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (root / "vendor/pacdm_original.py", root / "data/stewart.cmg.json",
                                          root / "stewart/reference.py", root / "stewart/pin_backend.py")},
              "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest()}
    (out / "reference.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output_directory": str(out)}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
