"""Execution-gate regressions. All generated traces here are test stubs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import launch
import run_isaac


def write_case(root, case='nominal', run_id='test-run', duration=26., **changes):
    directory = root / case
    directory.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    (root/'logs'/f'{case}.log').write_text('Synthetic clean process log, unit test only\n')
    record = dict(name=case, case=case, run_id=run_id, engine='Isaac Sim / PhysX',
                  source_sha256={'test-only': 'fingerprint'}, execution_exit_code=0,
                  duration_s=duration, simulated_s=duration, completed=True,
                  failure=None, termination_reason='duration')
    record.update(changes)
    launch.write_json(directory / 'case.json', record)
    # This gate only checks the artifact exists. The report does schema checks.
    (directory / 'trajectory.npz').write_bytes(b'unit-test stub, NOT simulator evidence')
    return record


def write_preflight(root, run_id='test-run', **changes):
    record = dict(run_id=run_id, passed=True, source_sha256={'test-only': 'fingerprint'})
    record.update(changes)
    for name in ('mechanics.json', 'reference_validation.json'):
        launch.write_json(root / name, record)


def test_completed_current_run_passes_execution_gate(tmp_path):
    write_case(tmp_path)
    write_preflight(tmp_path)
    assert launch.inspect_case_execution(tmp_path, 'nominal', 'test-run', 26., require_preflight=True) == []


@pytest.mark.parametrize('changes', [
    {'execution_exit_code': 1},
    {'termination_reason': 'exception', 'failure': "Failed to create simulation view with backend 'physx'",
     'simulated_s': 0., 'completed': False},
    {'termination_reason': 'interrupted'},
    {'run_id': 'stale-run'},
    {'case': 'payload'},
    {'engine': 'MuJoCo'},
    {'source_sha256': {}},
    {'simulated_s': 0.},
    {'simulated_s': True},
    {'simulated_s': '26'},
    {'simulated_s': 27.},
    {'simulated_s': 2.},
    {'duration_s': 2.},
    {'completed': False},
    {'failure': 'Failure after logging'},
])
def test_zero_os_status_does_not_override_bad_case_evidence(tmp_path, changes):
    write_case(tmp_path, **changes)
    assert launch.inspect_case_execution(tmp_path, 'nominal', 'test-run', 26.)


@pytest.mark.parametrize('name', ['nominal', 'PD_ablation', 'no_actuation'])
def test_real_physical_fall_is_execution_not_startup_error(tmp_path, name):
    write_case(tmp_path, case=name, simulated_s=.2, completed=False,
               termination_reason='fall', failure='Physical fall threshold reached')
    assert launch.inspect_case_execution(tmp_path, name, 'test-run', 26.) == []


@pytest.mark.parametrize('changes', [{'passed': False}, {'run_id': 'old'}, {'source_sha256': {'old': 'code'}}])
def test_stale_or_failed_preflight_is_rejected(tmp_path, changes):
    write_case(tmp_path)
    write_preflight(tmp_path, **changes)
    assert launch.inspect_case_execution(tmp_path, 'nominal', 'test-run', 26., require_preflight=True)


def test_missing_preflight_is_rejected_even_when_case_completed(tmp_path):
    write_case(tmp_path)
    errors = launch.inspect_case_execution(tmp_path, 'nominal', 'test-run', 26., require_preflight=True)
    assert any('mechanics.json' in error for error in errors)
    assert any('reference_validation.json' in error for error in errors)


@pytest.mark.parametrize('contents', [None, '{broken', '[]', '{}'])
def test_missing_or_malformed_case_record(tmp_path, contents):
    directory = tmp_path / 'nominal'
    directory.mkdir()
    if contents is not None:
        (directory / 'case.json').write_text(contents)
    assert launch.inspect_case_execution(tmp_path, 'nominal', 'test-run', 26.)


@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_empty_trace_is_not_an_execution(tmp_path, missing):
    write_case(tmp_path)
    trace = tmp_path / 'nominal' / 'trajectory.npz'
    trace.unlink() if missing else trace.write_bytes(b'')
    assert any('trajectory.npz' in error for error in launch.inspect_case_execution(
        tmp_path, 'nominal', 'test-run', 26.))


@pytest.mark.parametrize('mode', ['exception_with_zero_exit', 'missing_preflight', 'nonzero_exit'])
def test_launcher_blocks_remaining_cases_after_first_failure(tmp_path, monkeypatch, mode):
    runtime = tmp_path / 'python.exe'
    runtime.touch()
    output = tmp_path / 'results'
    monkeypatch.setattr(launch, 'controller_probe', lambda: {'ffmpeg': str(tmp_path / 'unused_ffmpeg')})
    calls = []

    def run_logged(command, log_path, env):
        if '--case' in command:
            case = command[command.index('--case') + 1]
            run_id = command[command.index('--run-id') + 1]
            calls.append(case)
            changes = {}
            if mode == 'exception_with_zero_exit':
                changes = dict(termination_reason='exception', completed=False, simulated_s=0.,
                               failure="Failed to create simulation view with backend 'physx'")
            write_case(output, case=case, run_id=run_id, **changes)
            if mode == 'nonzero_exit':
                write_preflight(output, run_id=run_id)
                return 1
            return 0  # Reproduces the OS code found in the supplied manifest.
        return 1  # Report remains a failure; no fabricated validation pass.

    monkeypatch.setattr(launch, 'run_logged', run_logged)
    result = launch.main(['--isaac-python', str(runtime), '--output', str(output),
                          '--suite', 'full', '--no-video'])
    manifest = json.loads((output / 'suite_manifest.json').read_text())
    assert result == 1
    assert calls == ['nominal']
    assert manifest['cases'][0]['status'] == 'failed'
    assert manifest['cases'][0]['exit_code'] == 1
    assert manifest['cases'][0]['process_exit_code'] == (1 if mode == 'nonzero_exit' else 0)
    assert [item['status'] for item in manifest['cases'][1:]] == ['blocked'] * 6


def test_valid_nominal_does_not_block_following_cases(tmp_path, monkeypatch):
    runtime = tmp_path / 'python.exe'
    runtime.touch()
    output = tmp_path / 'results'
    calls = []
    monkeypatch.setattr(launch, 'controller_probe', lambda: {'ffmpeg': 'unused'})

    def run_logged(command, log_path, env):
        if '--case' in command:
            case = command[command.index('--case') + 1]
            run_id = command[command.index('--run-id') + 1]
            calls.append(case)
            write_case(output, case=case, run_id=run_id)
            if case == 'nominal':
                write_preflight(output, run_id=run_id)
            return 0
        return 1  # Fake traces must never receive a real report pass.

    monkeypatch.setattr(launch, 'run_logged', run_logged)
    assert launch.main(['--isaac-python', str(runtime), '--output', str(output),
                        '--suite', 'full', '--no-video']) == 1
    assert calls == list(launch.FULL_CASES)
    manifest = json.loads((output / 'suite_manifest.json').read_text())
    assert all(entry['status'] == 'completed' for entry in manifest['cases'])


@pytest.mark.parametrize('failure, expected_code', [(None, 0), (RuntimeError('test failure'), 1),
                                                   (KeyboardInterrupt(), 130)])
def test_runner_saves_status_before_closing_app(tmp_path, monkeypatch, failure, expected_code):
    args = SimpleNamespace(output=tmp_path, case='nominal', run_id='test-run', duration=26., headless=True)
    monkeypatch.setattr(run_isaac, 'parse_args', lambda: args)
    observed = []

    class App:
        def __init__(self, config):
            assert config['fast_shutdown'] is True

        def close(self, exit_code=0):
            record = json.loads((tmp_path / 'nominal' / 'case.json').read_text())
            observed.append((exit_code, record['execution_exit_code']))

    module = ModuleType('isaacsim')
    module.SimulationApp = App
    monkeypatch.setitem(sys.modules, 'isaacsim', module)

    def execute(args, app, result):
        if failure is not None:
            raise failure
        result.update(completed=True, simulated_s=26., termination_reason='duration')

    monkeypatch.setattr(run_isaac, 'execute', execute)
    assert run_isaac.main() == expected_code
    assert observed == [(expected_code, expected_code)]
    assert not (tmp_path / 'nominal' / 'case.json.tmp').exists()


def test_runner_records_python_shutdown_failure_after_successful_rollout(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path, case='nominal', run_id='test-run', duration=26., headless=True)
    monkeypatch.setattr(run_isaac, 'parse_args', lambda: args)
    class App:
        def __init__(self, config): pass
        def close(self, exit_code=0): raise RuntimeError('test native close wrapper failure')
    module = ModuleType('isaacsim')
    module.SimulationApp = App
    monkeypatch.setitem(sys.modules, 'isaacsim', module)
    monkeypatch.setattr(run_isaac, 'execute', lambda args, app, result: result.update(
        completed=True, simulated_s=26., termination_reason='duration'))
    assert run_isaac.main() == 1
    record = json.loads((tmp_path/'nominal/case.json').read_text())
    assert record['completed'] is True  # physical completion is not erased
    assert record['execution_exit_code'] == 1
    assert record['application_shutdown']['state'] == 'exception'
