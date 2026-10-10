"""Rebuild the CMG and PACDM reference without a simulation engine."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from panda.model import build_model
from panda.task import build_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "reference")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == (ROOT / "data").resolve():
        parser.error("Use a separate output directory to preserve the bundled reference.")
    output.mkdir(parents=True, exist_ok=True)
    model = build_model()
    stored_model = json.loads((ROOT / "data" / "panda_cmg.json").read_text())
    if model != stored_model:
        raise RuntimeError("The imported CMG differs from the bundled model.")
    reference = build_reference(model)
    arrays = {key: value for key, value in reference.items() if key != "info"}
    differences = {}
    identical = {}
    with np.load(ROOT / "data" / "reference.npz", allow_pickle=False) as stored:
        if set(stored.files) != set(arrays):
            raise RuntimeError("Reference array names differ.")
        for key in stored.files:
            actual = np.asarray(arrays[key])
            expected = stored[key]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise RuntimeError(f"Invalid rebuilt reference array: {key}")
            differences[key] = float(np.max(np.abs(actual - expected)))
            identical[key] = bool(np.array_equal(actual, expected))
    np.savez_compressed(output / "reference.npz", **arrays)
    (output / "panda_cmg.json").write_text(json.dumps(model, indent=2) + "\n")
    (output / "reference.json").write_text(json.dumps(reference["info"], indent=2) + "\n")
    report = {
        "passed": all(identical.values()),
        "scope": "CMG import and PACDM reference reconstruction",
        "arrays_identical": identical,
        "maximum_absolute_differences": differences,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pacdm_sha256": hashlib.sha256((ROOT / "vendor" / "pacdm_original.py").read_bytes()).hexdigest(),
    }
    (output / "rebuild_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'PASS' if report['passed'] else 'DIFFERENT'}: {sum(identical.values())}/{len(identical)} arrays identical")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
