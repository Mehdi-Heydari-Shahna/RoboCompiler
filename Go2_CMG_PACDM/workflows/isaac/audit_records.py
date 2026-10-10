"""Audit recorded Isaac evidence without modifying records or running physics."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
RUNS = ("20260924T134619Z_d9c46154", "20260924T143514Z_e63dcb00")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, recorded):
    """Normalize portable relative fingerprints, including Windows separators."""
    relative = Path(str(recorded).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise ValueError(f"Expected a relative evidence path: {recorded}")
    return Path(root) / relative


def _copy_for_report(source, target):
    """Copy metadata; link read-only large inputs where the platform permits."""
    source, target = Path(source), Path(target)
    # Reporting never writes native traces, scene snapshots, or logs. Videos
    # are copied because the retained report verifies their resolved containment.
    if source.suffix.lower() in {".npz", ".usda", ".log"}:
        try:
            target.symlink_to(source.resolve())
            return str(target)
        except OSError:
            pass
    return shutil.copy2(source, target)


def audit_run(records, *, source_root=ROOT, reevaluate=True):
    """Verify historical hashes and re-evaluate only an isolated temporary copy."""
    records, source_root = Path(records).resolve(), Path(source_root).resolve()
    manifest = json.loads((records / "suite_manifest.json").read_text(encoding="utf-8"))
    original = json.loads((records / "validation.json").read_text(encoding="utf-8"))
    errors, verified_sources, verified_results = [], set(), set()
    for name in manifest["requested_cases"]:
        case_dir = local_path(records, name)
        case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
        if case.get("run_id") != manifest.get("run_id"):
            errors.append(f"{name}: run identity mismatch")
        for path, expected in case["source_sha256"].items():
            source = local_path(source_root, path)
            if not source.is_file() or file_hash(source) != expected:
                errors.append(f"Source fingerprint mismatch: {path}")
            verified_sources.add(str(path).replace("\\", "/"))
        recorded_files = dict(summary["file_sha256"])
        video = case.get("video", {})
        if video.get("sha256"):
            recorded_files[video["path"]] = video["sha256"]
        capture = case.get("capture_audit", {})
        if capture.get("sha256"):
            recorded_files[capture["path"]] = capture["sha256"]
        for path, expected in recorded_files.items():
            source = local_path(case_dir, path)
            if not expected or not source.is_file() or file_hash(source) != expected:
                errors.append(f"{name}: result fingerprint mismatch: {path}")
            verified_results.add(f"{name}/{path}")

    # This integration checksum supplements, rather than replaces, original hashes.
    inventory_path = source_root / "records" / "SHA256SUMS.json"
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))["files"]
        for path, expected in inventory.items():
            if Path(path).parts[0] == records.name:
                source = local_path(records.parent, path)
                if not source.is_file() or file_hash(source) != expected:
                    errors.append(f"Record snapshot mismatch: {path}")

    reevaluated = None
    if reevaluate and not errors:
        from isaac_validation.report import main as report_main

        with tempfile.TemporaryDirectory(prefix="go2_isaac_audit_") as temporary:
            sandbox = Path(temporary) / records.name
            shutil.copytree(records, sandbox, copy_function=_copy_for_report)
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = report_main(["--output", str(sandbox), "--suite", manifest["suite"],
                                         "--run-id", manifest["run_id"]])
            reevaluated = json.loads((sandbox / "validation.json").read_text(encoding="utf-8"))
            if exit_code != 0:
                errors.append("Retained reporting code did not accept the recorded evidence")
            for key in ("status", "full_validation", "smoke_passed", "run_id"):
                if reevaluated.get(key) != original.get(key):
                    errors.append(f"Re-evaluation changed recorded {key}")
            for name, case in reevaluated["cases"].items():
                for key in ("mission_passed", "expected_outcome_passed", "completed"):
                    if case.get(key) != original["cases"][name].get(key):
                        errors.append(f"Re-evaluation changed {name}/{key}")
    return {
        "passed": not errors,
        "run": records.name,
        "run_id": manifest.get("run_id"),
        "suite": manifest.get("suite"),
        "recorded_status": original.get("status"),
        "source_files_verified": len(verified_sources),
        "case_result_files_verified": len(verified_results),
        "report_reevaluated": reevaluated is not None,
        "errors": errors,
        "scope": "Offline evidence audit; no simulation executed and original records unchanged",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, help="Audit one run; defaults to both included records")
    parser.add_argument("--hashes-only", action="store_true", help="Skip temporary report re-evaluation")
    args = parser.parse_args(argv)
    paths = [args.records] if args.records else [ROOT / "records" / name for name in RUNS]
    results = []
    for path in paths:
        try:
            results.append(audit_run(path, reevaluate=not args.hashes_only))
        except (OSError, ValueError, KeyError) as error:
            results.append({"passed": False, "run": str(path), "errors": [str(error)]})
    print(json.dumps(results, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
