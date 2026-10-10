"""Run integrity, CPU preflight, Isaac smoke, and full numerical validation.

Uses the already-active Python environment. No packages are installed, no old
results are reused, and a failing stage prevents every subsequent stage.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import subprocess
import sys
from verify_package import verify

ROOT = Path(__file__).resolve().parent


def run_sequence(root=ROOT) -> int:
    root = Path(root).resolve()
    print('Stewart validation v4. Python: ' + sys.executable, flush=True)
    try:
        problems = verify(root)
        if problems:
            print('Package integrity FAILED:\n' + '\n'.join(problems), file=sys.stderr)
            return 1
        if importlib.util.find_spec('isaacsim') is None:
            print('Isaac is not installed in this interpreter. Activate an Isaac Sim Python environment.', file=sys.stderr)
            return 2
        print('Package integrity PASS.', flush=True)
        stages = [
            ('CPU preflight', ['--preflight-only', '--no-video']),
            ('Two-second Isaac smoke, without GUI/video', ['--smoke', '--no-video']),
            ('Full Isaac numerical validation, unchanged full-mission gates', ['--no-video']),
        ]
        for index, (label, options) in enumerate(stages, 1):
            print(f'\n[{index}/3] {label}', flush=True)
            completed = subprocess.run([sys.executable, str(root/'run_validation.py'), *options],
                                       cwd=root, check=False)
            if completed.returncode != 0:
                print(f'Stopped: {label} returned {completed.returncode}. Later stages were not run.\n'
                      'Keep the newest results_isaac folder, including logs and native_inertial_audit.json.',
                      file=sys.stderr, flush=True)
                return 1
        print('\nComplete numerical validation PASSED. Open the last printed report.html.', flush=True)
        return 0
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f'Launcher failed before acceptance: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    try:
        raise SystemExit(run_sequence())
    except KeyboardInterrupt:
        print('Interrupted. The suite is not validated.', file=sys.stderr)
        raise SystemExit(130)
