"""Startup regression tests. Mocked metadata/GPU data are NOT native execution."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kangaroo_isaac import preflight as pf
from kangaroo_isaac.model import ROOT
from tools import repair_miniforge as repair


def complete_packages(version='6.1.0.0'):
    p = {name: None for name in pf.PACKAGE_NAMES}
    p.update({'isaacsim': version, 'isaacsim-kernel': version, 'isaacsim-core': version,
              'numpy': '2.3.1', 'scipy': '1.17.0', 'pytest': '9.1.1', 'Pillow': '12.3.0', 'pip': '26.2'})
    p['isaacsim-app' if version.startswith('6.1.') else 'isaacsim-simulation-app'] = version
    return p


def mock_environment(monkeypatch, pip_status='PASS', pip_output='No broken requirements found.', memory=8188):
    monkeypatch.setattr(pf, 'installed_versions', complete_packages)
    monkeypatch.setattr(pf.sys, 'version_info', (3, 12, 14))
    monkeypatch.setattr(pf, 'check_package_dependencies', lambda: {
        'status': pip_status, 'command': ['python', '-m', 'pip', 'check'],
        'exit_code': 0 if pip_status == 'PASS' else 1, 'output': pip_output})
    monkeypatch.setattr(pf, 'detect_gpu', lambda: {
        'status': 'DETECTED', 'output': 'MOCKED GPU for unit test only',
        'devices': [{'name': 'MOCKED GPU', 'driver_version': '596.08', 'memory_total_MiB': memory}]})


def test_original_pillow_failure_blocks_readiness(monkeypatch):
    text = 'isaacsim-kernel 6.1.0.0 has requirement Pillow<13,>=12.1.1, but you have pillow 11.3.0.'
    mock_environment(monkeypatch, 'FAIL', text)
    p = complete_packages()
    p['Pillow'] = '11.3.0'
    monkeypatch.setattr(pf, 'installed_versions', lambda: p)
    info = pf.inspect_environment()
    assert info['status'] == 'BLOCKED'
    assert info['packages']['Pillow'] == '11.3.0'
    assert any(text in e for e in info['errors'])
    assert any('SETUP_MINIFORGE.cmd' in e for e in info['errors'])
    assert info['isaac_sim_execution'] == 'NOT_RUN' and not info['certified_ready']


@pytest.mark.parametrize('status', ['FAIL', 'ERROR', 'TIMEOUT'])
def test_incomplete_dependency_check_never_claims_ready(monkeypatch, status):
    mock_environment(monkeypatch, status, 'Cannot complete dependency check')
    assert pf.inspect_environment()['status'] == 'BLOCKED'


def test_complete_metadata_with_small_gpu_is_attempt_not_certification(monkeypatch):
    mock_environment(monkeypatch)
    result = pf.inspect_environment()
    assert result['status'] == 'READY_FOR_NATIVE_ATTEMPT'
    assert result['hardware_compatibility'] == 'NOT_CERTIFIED'
    assert any('16 GB' in w for w in result['warnings'])
    assert result['isaac_sim_execution'] == 'NOT_RUN'
    assert not result['certified_ready']


def test_missing_gpu_still_blocks_native(monkeypatch):
    mock_environment(monkeypatch)
    monkeypatch.setattr(pf, 'detect_gpu', lambda: {'status': 'NOT_DETECTED', 'output': None, 'devices': []})
    assert pf.inspect_environment()['status'] == 'BLOCKED'


def test_large_gpu_still_not_certified(monkeypatch):
    mock_environment(monkeypatch, memory=49140)
    result = pf.inspect_environment()
    assert not any('below the documented' in w for w in result['warnings'])
    assert result['hardware_compatibility'] == 'NOT_CERTIFIED'


@pytest.mark.parametrize('version,py', [('6.1.0.0', (3, 12)), ('5.1.0.0', (3, 11))])
def test_matching_runtime_metadata_has_no_version_errors(version, py):
    assert pf.runtime_package_errors(complete_packages(version), py) == []


def test_61_does_not_require_obsolete_distribution_name():
    p = complete_packages()
    assert p['isaacsim-simulation-app'] is None
    assert pf.runtime_package_errors(p, (3, 12)) == []


@pytest.mark.parametrize('missing', ['isaacsim-kernel', 'isaacsim-app', 'isaacsim-core'])
def test_bare_or_incomplete_metapackage_is_blocked(missing):
    p = complete_packages()
    p[missing] = None
    assert any(missing in e for e in pf.runtime_package_errors(p, (3, 12)))


def test_mixed_native_versions_blocked():
    p = complete_packages()
    p['isaacsim-core'] = '5.1.0.0'
    assert any('Mixed' in e for e in pf.runtime_package_errors(p, (3, 12)))


@pytest.mark.parametrize('version,py', [(None, (3, 12)), ('6.0.0.0', (3, 12)), ('6.1.0.0', (3, 11)), ('5.1.0.0', (3, 12))])
def test_missing_unsupported_or_wrong_python_is_blocked(version, py):
    p = complete_packages()
    p['isaacsim'] = version
    assert pf.runtime_package_errors(p, py)


@pytest.mark.parametrize('code,status', [(0, 'PASS'), (1, 'FAIL'), (2, 'FAIL')])
def test_pip_check_preserves_output_and_returncode(monkeypatch, code, status):
    def run(command, **kwargs):
        assert command[:4] == [sys.executable, '-m', 'pip', 'check']
        assert kwargs['timeout'] == 60.0
        assert kwargs['encoding'] == 'utf-8'
        return SimpleNamespace(returncode=code, stdout='stdout\n', stderr='stderr\n')
    monkeypatch.setattr(pf.subprocess, 'run', run)
    result = pf.check_package_dependencies()
    assert result['status'] == status and result['exit_code'] == code
    assert result['output'] == 'stdout\nstderr'


@pytest.mark.parametrize('kind', ['timeout', 'oserror'])
def test_pip_check_operating_errors_are_reported(monkeypatch, kind):
    def run(*args, **kwargs):
        if kind == 'timeout':
            raise subprocess.TimeoutExpired(args[0], 60)
        raise OSError('test-only missing executable')
    monkeypatch.setattr(pf.subprocess, 'run', run)
    result = pf.check_package_dependencies()
    assert result['status'] == ('TIMEOUT' if kind == 'timeout' else 'ERROR')
    assert result['exit_code'] is None


@pytest.mark.parametrize('output,code,status', [
    ('NVIDIA GeForce RTX 4060 Laptop GPU, 596.08, 8188 MiB\n', 0, 'DETECTED'),
    ('', 0, 'ERROR'), ('not a GPU row', 1, 'ERROR'),
])
def test_gpu_csv_parsing(monkeypatch, output, code, status):
    monkeypatch.setattr(pf.shutil, 'which', lambda name: '/test-only/nvidia-smi')
    monkeypatch.setattr(pf.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout=output, stderr='', returncode=code))
    result = pf.detect_gpu()
    assert result['status'] == status
    if status == 'DETECTED':
        assert result['devices'][0]['memory_total_MiB'] == 8188


def test_project_requirement_path_is_absolute_and_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = repair.requirement_file('6.1.0.0', (3, 12))
    assert path.is_absolute() and path.is_file()
    assert 'Pillow>=12.1.1,<13' in path.read_text()
    assert repair.requirement_file('5.1.0.0', (3, 11)).name == 'requirements-extra.txt'


@pytest.mark.parametrize('version,py', [(None, (3, 12)), ('6.1.0.0', (3, 13)), ('5.1.0.0', (3, 12)), ('9.0.0', (3, 12))])
def test_setup_does_not_guess_release_or_python(version, py):
    with pytest.raises(ValueError):
        repair.requirement_file(version, py)


def test_repair_cannot_replace_simulation_stack():
    text = repair.protected_constraints({'isaacsim': '6.1.0.0', 'isaacsim-kernel': '6.1.0.0',
        'torch': '2.11.0+cu128', 'numpy': '2.3.1', 'scipy': '1.17.0', 'Pillow': '11.3.0'})
    assert 'isaacsim==6.1.0.0' in text and 'isaacsim-kernel==6.1.0.0' in text
    assert 'torch==2.11.0+cu128' in text and 'numpy==2.3.1' in text
    assert 'Pillow' not in text and 'scipy' not in text


def test_installer_is_explicit_and_uses_current_python_and_constraints(tmp_path):
    req = tmp_path / 'folder with spaces' / 'requirements.txt'
    constraint = tmp_path / 'protected.txt'
    command = repair.install_command(req, constraint)
    assert command[:4] == [sys.executable, '-m', 'pip', 'install']
    assert command[command.index('-r') + 1] == str(req)
    assert command[command.index('--constraint') + 1] == str(constraint)
    assert '--upgrade' not in command and '--force-reinstall' not in command


def test_setup_rejects_missing_conda(monkeypatch):
    monkeypatch.delenv('CONDA_PREFIX', raising=False)
    assert repair.active_conda_error()


def test_setup_rejects_wrong_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv('CONDA_PREFIX', str(tmp_path))
    assert 'does not belong' in repair.active_conda_error()


def test_setup_rejects_base(monkeypatch):
    monkeypatch.setenv('CONDA_PREFIX', sys.prefix)
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'base')
    assert 'base' in repair.active_conda_error()


def test_active_conda_matches_current_python(monkeypatch):
    monkeypatch.setenv('CONDA_PREFIX', sys.prefix)
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'isaac61')
    assert repair.active_conda_error() is None


def test_dry_run_never_calls_installer(monkeypatch, capsys):
    monkeypatch.setattr(repair, 'active_conda_error', lambda: None)
    monkeypatch.setattr(repair, 'package_snapshot', lambda: {'isaacsim': '6.1.0.0', 'Pillow': '11.3.0'})
    monkeypatch.setattr(repair.sys, 'version_info', (3, 12, 14))
    monkeypatch.setattr(repair.subprocess, 'Popen', lambda *a, **k: pytest.fail('dry-run tried to install'))
    assert repair.main(['--dry-run']) == 0
    assert 'no installation or native execution' in capsys.readouterr().out


def test_cli_can_run_from_unrelated_folder_with_space(tmp_path):
    elsewhere = tmp_path / 'not the project'
    elsewhere.mkdir()
    for script in [ROOT / 'run_isaac.py', ROOT / 'tools' / 'repair_miniforge.py']:
        result = subprocess.run([sys.executable, str(script), '--help'], cwd=elsewhere,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_pytest_does_not_write_optional_cache(tmp_path):
    (tmp_path / 'test_example.py').write_text('def test_example():\n    assert 2 + 2 == 4\n')
    cache = tmp_path / '.pytest_cache'
    cache.write_text('A file intentionally occupies the cache directory name.\n')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-c', str(ROOT / 'pytest.ini'),
        str(tmp_path / 'test_example.py'), '-o', f'cache_dir={cache}', '-q'], cwd=tmp_path,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PytestCacheWarning' not in result.stdout + result.stderr
    assert cache.read_text().startswith('A file intentionally')


@pytest.mark.parametrize('version', [(3, 11), (3, 12)])
def test_setup_parses_on_target_python_versions(version):
    ast.parse((ROOT / 'tools/repair_miniforge.py').read_text(), feature_version=version)


@pytest.mark.parametrize('filename', ['SETUP_MINIFORGE.cmd', 'RUN_KANGAROO.cmd'])
def test_windows_launchers_quote_interpreter_and_project_paths(filename):
    source = (ROOT / filename).read_text()
    assert '"%CONDA_PREFIX%\\python.exe" "%~dp0' in source
    assert 'if not defined CONDA_PREFIX' in source
    assert 'DisableDelayedExpansion' in source
    assert 'exit /b %errorlevel%' in source
    assert 'cd C:' not in source and 'cd /d C:' not in source
    assert 'OMNI_KIT_ACCEPT_EULA' not in source
