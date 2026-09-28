"""Run a tracked-excavator case from the code-only distribution.

This entry point generates one native trajectory and its sampled dynamics audit.
The separate replay, task-audit, report and video commands consume that trajectory.
Case definitions and the physical simulation/controller remain unchanged.
"""
from project import SOURCE
import argparse
import hashlib
import json
from pathlib import Path

CASES = {
    'coarse': dict(dt=.001),
    'nominal': dict(dt=.0005),
    'fine': dict(dt=.00025),
    'low_traction': dict(dt=.0005, traction=.55),
    'heavy_payload': dict(dt=.0005, density_scale=1.25),
    'no_drive': dict(dt=.0005, duration=6., soil=False, drive_enabled=False),
    'no_bucket_contact': dict(dt=.0005, duration=16., bucket_contact=False),
}

SOURCE_PROVENANCE_SCOPE = (
    'Code-only distribution integrity: the included v21 source and model/configuration '
    'files listed in SOURCE_CODE_SHA256.json. Historical generated results are outside '
    'this manifest; their original release manifest is retained in provenance/.'
)


def source_integrity():
    """Verify every entry of the explicit code-only source manifest."""
    manifest_path = SOURCE/'SOURCE_CODE_SHA256.json'
    hashes = json.loads(manifest_path.read_text())
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('The code-only source manifest must be a nonempty path-to-SHA256 mapping')
    mismatches = []
    for relative, expected in hashes.items():
        path = SOURCE/relative
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or not path.resolve().is_relative_to(SOURCE.resolve())
                or not isinstance(expected, str) or len(expected) != 64
                or any(character not in '0123456789abcdef' for character in expected)):
            raise ValueError('Invalid code-only source manifest entry: '+str(relative))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
    return dict(passed=not mismatches, files=len(hashes), mismatches=mismatches,
                manifest=manifest_path.name, scope=SOURCE_PROVENANCE_SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=list(CASES), default='nominal',
                        help='Case to generate (default: nominal)')
    args = parser.parse_args()
    integrity = source_integrity()
    if not integrity['passed']:
        raise RuntimeError('Code-only source integrity check failed: '+', '.join(integrity['mismatches']))
    print(f"Verified {integrity['files']} included source/model files against {integrity['manifest']}", flush=True)
    from mobile_simulation import run_case
    run_case(args.case, **CASES[args.case])


if __name__ == '__main__':
    main()
