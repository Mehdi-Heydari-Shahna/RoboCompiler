"""Test launcher order/failure propagation, without starting an Isaac process."""
import hashlib
import json
import subprocess
import types
from pathlib import Path
import pytest
import run_checked
from verify_package import verify


def setup_runner(monkeypatch, codes):
    calls = []
    monkeypatch.setattr(run_checked, 'verify', lambda root: [])
    monkeypatch.setattr(run_checked.importlib.util, 'find_spec', lambda name: object())
    def child(command, cwd, check):
        calls.append(command)
        assert check is False
        assert Path(command[1]).name == 'run_validation.py'
        assert Path(cwd).is_absolute()
        return types.SimpleNamespace(returncode=codes[len(calls)-1])
    monkeypatch.setattr(run_checked.subprocess, 'run', child)
    return calls


def test_three_stages_in_order(monkeypatch, tmp_path):
    calls = setup_runner(monkeypatch, [0, 0, 0])
    assert run_checked.run_sequence(tmp_path) == 0
    assert [c[2:] for c in calls] == [['--preflight-only', '--no-video'], ['--smoke', '--no-video'], ['--no-video']]


@pytest.mark.parametrize('codes,expected_calls', [([1], 1), ([0, 1], 2), ([0, 0, 1], 3), ([0, -1073741819], 2)])
def test_failure_stops_later_stages(monkeypatch, tmp_path, codes, expected_calls):
    calls = setup_runner(monkeypatch, codes)
    assert run_checked.run_sequence(tmp_path) != 0
    assert len(calls) == expected_calls


def test_integrity_failure_prevents_all_execution(monkeypatch, tmp_path):
    calls = setup_runner(monkeypatch, [])
    monkeypatch.setattr(run_checked, 'verify', lambda root: ['Changed: runtime.py'])
    assert run_checked.run_sequence(tmp_path) != 0 and not calls


def test_missing_isaac_prevents_all_execution(monkeypatch, tmp_path):
    calls = setup_runner(monkeypatch, [])
    monkeypatch.setattr(run_checked.importlib.util, 'find_spec', lambda name: None)
    assert run_checked.run_sequence(tmp_path) == 2 and not calls


def test_launch_exception_cannot_report_success(monkeypatch, tmp_path):
    setup_runner(monkeypatch, [])
    def fail(*args, **kwargs): raise OSError('Test process startup failure')
    monkeypatch.setattr(run_checked.subprocess, 'run', fail)
    assert run_checked.run_sequence(tmp_path) == 1


def test_integrity_pass_changed_and_missing_file(tmp_path):
    content = b'test original'
    (tmp_path/'module.py').write_bytes(content)
    (tmp_path/'SHA256SUMS.json').write_text(json.dumps({'module.py': hashlib.sha256(content).hexdigest()}))
    assert verify(tmp_path) == []
    (tmp_path/'module.py').write_bytes(b'changed')
    assert verify(tmp_path) == ['Changed: module.py']
    (tmp_path/'module.py').unlink()
    assert verify(tmp_path) == ['Missing: module.py']


def test_manifest_path_cannot_escape_package(tmp_path):
    (tmp_path/'SHA256SUMS.json').write_text(json.dumps({'../outside.py': '0'*64}))
    with pytest.raises(ValueError, match='Invalid path'): verify(tmp_path)
