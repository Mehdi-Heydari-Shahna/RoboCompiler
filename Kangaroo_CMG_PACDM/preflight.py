"""Check dependencies and source-model inputs before running the benchmark."""
from pathlib import Path
import argparse
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', action='store_true', help='Also check Matplotlib')
    args = parser.parse_args()
    failed = []
    if sys.version_info[:2] != (3, 12):
        print('NOTE: the recorded benchmark environment uses Python 3.12.')
    for name in ['numpy', 'scipy', 'mujoco', 'pinocchio'] + (['matplotlib'] if args.plots else []):
        try:
            module = importlib.import_module(name)
            print(name, getattr(module, '__version__', 'available'))
        except Exception as exc:
            failed.append(name)
            print('MISSING/FAILED', name, str(exc))
    for name in ['data/contact_acceptance.json', 'data/provenance.json']:
        try:
            json.loads((ROOT/name).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            failed.append(name)
            print('MISSING/INVALID', name, str(exc))
    try:
        provenance = json.loads((ROOT/'data/provenance.json').read_text(encoding='utf-8'))
        for name in provenance['source_files']:
            if not (ROOT/name).is_file():
                failed.append(name)
                print('MISSING', name)
    except (OSError, ValueError, KeyError):
        failed.append('provenance source-files mapping')
    if failed:
        print('Install requirements_validated.txt in a Python 3.12 environment;')
        print('install requirements_plots.txt for the supplementary figures.')
        return False
    print('Ready for a first run. The benchmark generates CMG, configurations, and references.')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
