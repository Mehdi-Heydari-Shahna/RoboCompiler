"""Verify repository file hashes and, optionally, numerical acceptance results."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', action='store_true',
                        help='Also require a passing generated v22 numerical report')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    failed = []
    for name, expected in manifest['files'].items():
        path = ROOT/name
        if not path.is_file():
            failed.append((name, 'missing'))
        elif sha256(path) != expected:
            failed.append((name, 'SHA-256 mismatch'))
    provenance = json.loads((ROOT/'data/provenance.json').read_text(encoding='utf-8'))
    for group in ['core_sha256', 'source_files']:
        for name, expected in provenance[group].items():
            path = ROOT/name
            if not path.is_file() or sha256(path) != expected:
                failed.append((name, 'core/upstream provenance mismatch'))
    if args.results:
        path = ROOT/'results/complete_validation_v22.json'
        if not path.is_file():
            failed.append(('numerical report', 'run python run_kangaroo_v22.py first'))
        else:
            report = json.loads(path.read_text(encoding='utf-8'))
            gates = report.get('gates', [])
            passed = sum(bool(g.get('passed')) for g in gates)
            if (report.get('status') != 'PASS_RECONSTRUCTED_MODEL' or
                    len(gates) != 362 or passed != 362 or
                    report.get('gates_total') != 362 or report.get('gates_passed') != 362):
                failed.append(('numerical report', 'expected 362 passing v22 gates'))
            print(f'Generated numerical report: {passed}/{len(gates)} gates passed')
    for name, why in failed:
        print('FAIL', name, why)
    print(f"{'PASS' if not failed else 'FAIL'}: {len(manifest['files'])} repository file checks")
    if not failed:
        print('Core and upstream hashes match the recorded source provenance.')
    return not failed


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
