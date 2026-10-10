"""Read-only checks of the active environment; never import Isaac or NumPy here."""
from __future__ import annotations

import csv
from importlib import metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

PACKAGE_NAMES = (
    'isaacsim', 'isaacsim-kernel', 'isaacsim-app', 'isaacsim-core',
    'isaacsim-simulation-app', 'numpy', 'scipy', 'pytest', 'matplotlib',
    'Pillow', 'pip', 'torch',
)


def installed_versions() -> dict[str, str | None]:
    """Inspect distribution metadata, without importing native extensions."""
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_package_errors(packages: dict, python_version: tuple[int, int]) -> list[str]:
    """The 6.1 application distribution is isaacsim-app, not simulation-app."""
    errors = []
    version = packages.get('isaacsim')
    if not version:
        return ['Isaac Sim is not installed in this Python environment. '
                'Activate the Miniforge environment containing Isaac Sim.']
    if version.startswith('6.1.'):
        target_python = (3, 12)
        runtime = ('isaacsim-kernel', 'isaacsim-app', 'isaacsim-core')
    elif version.startswith('5.1.'):
        target_python = (3, 11)
        runtime = ('isaacsim-kernel', 'isaacsim-simulation-app', 'isaacsim-core')
    else:
        return [f'Isaac Sim {version} is not a targeted API version. '
                'This candidate targets 5.1.x and 6.1.x, not other versions.']
    if python_version != target_python:
        errors.append(f'Isaac Sim {version} requires Python '
                      f'{target_python[0]}.{target_python[1]} in the same environment.')
    missing = [name for name in runtime if not packages.get(name)]
    if missing:
        errors.append('Missing native runtime distribution(s): ' + ', '.join(missing) + '. '
                      'The isaacsim metapackage alone does not establish a complete runtime. '
                      'See docs/MINIFORGE_SETUP.md before installing additional Isaac components.')
    mismatched = [f'{name}={packages[name]}' for name in runtime
                  if packages.get(name) and packages[name] != version]
    if mismatched:
        errors.append('Mixed Isaac Sim package versions: ' + ', '.join(mismatched) +
                      f'; expected {version}. Do not mix Isaac releases.')
    return errors


def check_package_dependencies(timeout_s: float = 60.0) -> dict:
    """Run pip's read-only dependency check in this exact Python interpreter."""
    command = [sys.executable, '-m', 'pip', 'check', '--disable-pip-version-check']
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'command': command, 'exit_code': None,
                'output': f'pip check did not finish within {timeout_s:g} seconds.'}
    except OSError as exc:
        return {'status': 'ERROR', 'command': command, 'exit_code': None, 'output': str(exc)}
    return {'status': 'PASS' if result.returncode == 0 else 'FAIL', 'command': command,
            'exit_code': result.returncode,
            'output': '\n'.join(s.strip() for s in (result.stdout, result.stderr) if s.strip())}


def detect_gpu() -> dict:
    smi = shutil.which('nvidia-smi')
    if not smi and os.name == 'nt':
        candidate = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'System32/nvidia-smi.exe'
        if candidate.exists():
            smi = str(candidate)
    if not smi:
        return {'status': 'NOT_DETECTED', 'output': None, 'devices': []}
    try:
        result = subprocess.run(
            [smi, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'ERROR', 'output': str(exc), 'devices': []}
    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 3:
            continue
        match = re.fullmatch(r'\s*(\d+)\s*(?:MiB)?\s*', row[2])
        devices.append({'name': row[0].strip(), 'driver_version': row[1].strip(),
                        'memory_total_MiB': int(match.group(1)) if match else None})
    return {'status': 'DETECTED' if result.returncode == 0 and devices else 'ERROR',
            'output': result.stdout.strip() or result.stderr.strip(), 'devices': devices}


def inspect_environment() -> dict:
    packages = installed_versions()
    errors = runtime_package_errors(packages, tuple(sys.version_info[:2]))
    warnings = []
    for name in ('numpy', 'scipy'):
        if not packages[name]:
            errors.append(f'Missing dependency {name}. Run SETUP_MINIFORGE.cmd in the active Isaac environment.')
    pip_check = check_package_dependencies()
    if pip_check['status'] != 'PASS':
        errors.append('Installed package dependencies are not verified as compatible. '
                      'Resolve the pip check output below before a native run.\n' + pip_check['output'])
        if 'pillow' in pip_check['output'].lower() and (packages.get('isaacsim') or '').startswith('6.1.'):
            errors.append('For the reported Isaac 6.1 Pillow conflict, run SETUP_MINIFORGE.cmd, '
                          'or: python -m pip install "Pillow>=12.1.1,<13" ; then rerun python -m pip check.')
    gpu = detect_gpu()
    if gpu['status'] != 'DETECTED':
        errors.append('No usable NVIDIA driver/GPU was detected by nvidia-smi. '
                      'Native Isaac execution is blocked; offline checks still work.')
    capacities = [device['memory_total_MiB'] for device in gpu['devices']
                  if device['memory_total_MiB'] is not None]
    if (packages.get('isaacsim') or '').startswith('6.1.') and capacities and max(capacities) < 16000:
        warnings.append('Detected GPU memory is below the documented Isaac Sim 6.1 16 GB minimum. '
                        'A headless/no-visuals attempt is not a hardware-compatibility guarantee. '
                        'Native startup and task performance remain unverified.')
    warnings.extend([
        'GPU detection and pip check are not an RTX/driver compatibility certificate. '
        'Consult the NVIDIA requirements for the installed Isaac release.',
        'CPU PhysX is deliberate. Isaac Sim itself still needs its supported NVIDIA/RTX runtime.',
        'The source uses a 25 microsecond timestep; faithful runs are not promised to be real-time.',
    ])
    return {'status': 'READY_FOR_NATIVE_ATTEMPT' if not errors else 'BLOCKED',
            'certified_ready': False, 'isaac_sim_execution': 'NOT_RUN',
            'hardware_compatibility': 'NOT_CERTIFIED',
            'python_executable': sys.executable, 'python': sys.version,
            'platform': platform.platform(), 'conda_environment': os.environ.get('CONDA_DEFAULT_ENV'),
            'project_directory': str(Path(__file__).resolve().parents[1]),
            'working_directory': str(Path.cwd()),
            'packages': packages, 'pip_check': pip_check, 'gpu': gpu,
            'errors': errors, 'warnings': warnings}
