#!/usr/bin/env python
"""Explicit Miniforge dependency repair; never called implicitly by a simulation.

Uses absolute project paths. Preserves installed Isaac, Torch, and NumPy versions
with pip constraints. Does not accept NVIDIA licenses or install a new runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kangaroo_isaac.io_utils import write_json
from kangaroo_isaac.preflight import inspect_environment
from run_isaac import bundle


def package_snapshot() -> dict[str, str]:
    """Record installed distribution versions without importing any of them."""
    return dict(sorted((dist.metadata['Name'], dist.version) for dist in metadata.distributions()
                       if dist.metadata.get('Name')))


def requirement_file(isaac_version: str | None, python_version: tuple[int, int]) -> Path:
    if isaac_version and isaac_version.startswith('6.1.') and python_version == (3, 12):
        return ROOT / 'requirements-isaac61.txt'
    if isaac_version and isaac_version.startswith('5.1.') and python_version == (3, 11):
        return ROOT / 'requirements-extra.txt'
    raise ValueError('Activate the existing Isaac 6.1 / Python 3.12 environment '
                     '(or Isaac 5.1 / Python 3.11). No environment was changed.')


def protected_constraints(packages: dict[str, str]) -> str:
    """A small repair must not silently replace the installed simulation stack."""
    lines = []
    for name, version in sorted(packages.items()):
        normalized = name.lower().replace('_', '-').replace('.', '-')
        if (normalized == 'isaacsim' or normalized.startswith('isaacsim-')
                or normalized in ('torch', 'torchvision', 'torchaudio', 'numpy')):
            lines.append(f'{name}=={version}')
    return '\n'.join(lines) + '\n'


def active_conda_error() -> str | None:
    prefix = os.environ.get('CONDA_PREFIX')
    if not prefix:
        return 'Open Miniforge Prompt and run conda activate isaac61 first.'
    if os.path.normcase(str(Path(prefix).resolve())) != os.path.normcase(str(Path(sys.prefix).resolve())):
        return 'This Python does not belong to the active Conda environment. Check where python.'
    if os.environ.get('CONDA_DEFAULT_ENV') == 'base':
        return 'Do not repair the base environment. Run conda activate isaac61 first.'
    return None


def install_command(requirements: Path, constraints: Path) -> list[str]:
    return [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            '--only-binary=:all:', '--constraint', str(constraints), '-r', str(requirements)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Show intended changes only; do not install anything')
    parser.add_argument('--skip-tests', action='store_true', help='Skip the offline tests after a successful preflight')
    args = parser.parse_args(argv)
    print('Project:', ROOT, flush=True)
    print('Python:', sys.executable, flush=True)
    problem = active_conda_error()
    if problem:
        print('BLOCKED:', problem)
        return 2
    packages = package_snapshot()
    normalized = {name.lower().replace('_', '-'): version for name, version in packages.items()}
    try:
        requirements = requirement_file(normalized.get('isaacsim'), tuple(sys.version_info[:2]))
    except ValueError as exc:
        print('BLOCKED:', exc)
        return 2
    constraints_text = protected_constraints(packages)
    if args.dry_run:
        print('Requirements:', requirements)
        print(requirements.read_text(encoding='utf-8'))
        print('Protected installed versions:\n' + constraints_text)
        print('DRY RUN: no installation or native execution was performed.')
        return 0
    run_id = 'setup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    folder = ROOT / 'results' / run_id
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print('BLOCKED: cannot write project results:', exc)
        print('Place the repository in a writable local folder outside the protected directory.')
        return 2
    code = 2
    status = {'status': 'STARTED', 'certified_ready': False, 'isaac_sim_execution': 'NOT_RUN'}
    try:
        write_json(folder / 'environment_before.json', packages)
        constraints = folder / 'preserve_simulation_stack.txt'
        constraints.write_text(constraints_text, encoding='utf-8')
        command = install_command(requirements, constraints)
        write_json(folder / 'install_command.json', command)
        print('Installing project dependencies / repairing Pillow where required.', flush=True)
        print('Existing Isaac, Torch and NumPy versions are constrained, not replaced.', flush=True)
        # Foreground operation. stdout is relayed and saved for a useful failure bundle.
        with (folder / 'pip_install.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace')
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            finally:
                if process.stdout is not None:
                    process.stdout.close()
        write_json(folder / 'environment_after.json', package_snapshot())
        if code:
            status.update(status='INSTALL_FAILED', install_exit_code=code)
            return code
        info = inspect_environment()
        write_json(folder / 'preflight.json', info)
        print(json.dumps(info, indent=2), flush=True)
        if info['status'] != 'READY_FOR_NATIVE_ATTEMPT':
            status['status'] = 'PREFLIGHT_BLOCKED'
            code = 2
            return code
        if not args.skip_tests:
            command = [sys.executable, str(ROOT / 'run_isaac.py'), '--offline-test',
                       '--output', str(folder / 'offline')]
            code = subprocess.call(command, cwd=str(ROOT))
            if code:
                status.update(status='OFFLINE_TESTS_FAILED', offline_exit_code=code)
                return code
        status.update(status='SETUP_CHECKS_PASSED_NATIVE_NOT_RUN',
                      offline_tests='SKIPPED' if args.skip_tests else 'PASS')
        code = 0
        print('\nSetup checks passed. Native performance is NOT validated.')
        print('Next: RUN_KANGAROO.cmd --case nominal --headless --no-visuals')
        return code
    except KeyboardInterrupt:
        status['status'] = 'INTERRUPTED'
        code = 130
        return code
    except Exception as exc:
        import traceback
        (folder / 'setup_error.log').write_text(traceback.format_exc(), encoding='utf-8')
        status.update(status='ERROR', error=str(exc))
        print('SETUP ERROR:', exc, file=sys.stderr)
        code = 2
        return code
    finally:
        status['exit_code'] = code
        write_json(folder / 'setup_status.json', status)
        print('Setup results bundle:', bundle(folder), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
