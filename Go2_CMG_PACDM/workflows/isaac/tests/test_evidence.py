"""Synthetic regressions for the recorded shutdown and false-SMOKE_PASS failure."""
import json
from types import SimpleNamespace

import pytest

import launch
from isaac_validation.evidence import case_process_evidence, inspect_native_log, exit_code_text
from isaac_validation.report import main as report_main, summarize_case
from test_report import make_case, make_preflight
from test_launch import write_case, write_preflight

RENDER_ERROR = '[Error] [omni.replicator.core.scripts.orchestrator] Error while stepping, renderer failed to advance to the scheduled frame.\n'


def edit_manifest(root, **changes):
    path = root/'suite_manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize('code', [3221225477, -1073741819, 1, 130, True, '0', None])
def test_completed_child_never_cancels_failed_or_unknown_process(tmp_path, code):
    path, metadata, _ = make_case(tmp_path, duration=2.0)
    manifest = json.loads((tmp_path/'suite_manifest.json').read_text())
    manifest['cases'][0].update(process_exit_code=code)
    (tmp_path/'suite_manifest.json').write_text(json.dumps(manifest))
    evidence = case_process_evidence(tmp_path, 'nominal', metadata)
    assert not evidence['passed']
    assert not summarize_case(path)['video']['passed']


@pytest.mark.parametrize('mode', ['crash', 'renderer_error_with_zero_exit', 'missing_manifest', 'missing_log'])
def test_smoke_cli_rejects_completed_two_second_failed_execution(tmp_path, monkeypatch, mode):
    make_case(tmp_path, duration=2.0)
    make_preflight(tmp_path)
    monkeypatch.setattr('isaac_validation.report._plots', lambda *a: ([], None))
    if mode == 'crash':
        manifest = json.loads((tmp_path/'suite_manifest.json').read_text())
        manifest['cases'][0].update(status='failed', process_exit_code=3221225477, exit_code=3221225477)
        (tmp_path/'suite_manifest.json').write_text(json.dumps(manifest))
        (tmp_path/'logs/nominal.log').write_text(RENDER_ERROR+'Windows fatal exception: access violation\n')
    elif mode == 'renderer_error_with_zero_exit':
        (tmp_path/'logs/nominal.log').write_text(RENDER_ERROR)
    elif mode == 'missing_manifest':
        (tmp_path/'suite_manifest.json').unlink()
    else:
        (tmp_path/'logs/nominal.log').unlink()
    assert report_main(['--output', str(tmp_path), '--suite', 'smoke', '--run-id', 'test-only']) == 1
    report = json.loads((tmp_path/'validation.json').read_text())
    summary = json.loads((tmp_path/'nominal/summary.json').read_text())
    assert report['status'] == summary['status'] == 'SMOKE_FAIL'
    assert report['physics_diagnostic_passed'] is True
    assert report['full_validation'] is False
    assert summary['mission_status'] == 'NOT_EVALUATED'
    assert not summary['mission_evaluated']


@pytest.mark.parametrize('broken', ['[]', '{broken', '{}'])
def test_corrupt_manifest_is_failed_evidence_not_report_exception(tmp_path, broken):
    _, metadata, _ = make_case(tmp_path)
    (tmp_path/'suite_manifest.json').write_text(broken)
    assert not case_process_evidence(tmp_path, 'nominal', metadata)['passed']


@pytest.mark.parametrize('changes', [
    {'run_id': 'different'}, {'duration_s': 2}, {'requested_cases': []}, {'cases': []}, {'cases': 'bad'}])
def test_stale_missing_or_malformed_parent_fields(tmp_path, changes):
    _, metadata, _ = make_case(tmp_path)
    edit_manifest(tmp_path, **changes)
    assert not case_process_evidence(tmp_path, 'nominal', metadata)['passed']


def test_native_warning_is_retained_but_not_misclassified_as_crash(tmp_path):
    path = tmp_path/'native.log'
    path.write_text('[Warning] A different OpenUSD build was already imported. Mixing USD builds may conflict.\n')
    record = inspect_native_log(path)
    assert record['passed']
    assert record['warnings'] and record['issue_count'] == 0


def test_log_scan_is_not_limited_to_tail_or_excerpt_count(tmp_path):
    path = tmp_path/'native.log'
    path.write_text(RENDER_ERROR*60 + ('normal log line\n'*10000))
    record = inspect_native_log(path, max_issues=3)
    assert not record['passed']
    assert record['issue_count'] == 60 and len(record['issues']) == 3
    assert record['omitted_issues'] == 57


def test_windows_signed_and_unsigned_status_show_same_native_hex():
    assert '0xC0000005' in exit_code_text(3221225477)
    assert '0xC0000005' in exit_code_text(-1073741819)


def test_parent_gate_catches_silently_logged_renderer_error(tmp_path):
    write_case(tmp_path)
    write_preflight(tmp_path)
    (tmp_path/'logs/nominal.log').write_text(RENDER_ERROR)
    errors = launch.inspect_case_execution(tmp_path, 'nominal', 'test-run', 26,
                                          require_preflight=True, process_exit_code=0)
    assert any('renderer failed' in item for item in errors)


def test_full_suite_blocks_after_completed_child_native_crash(tmp_path, monkeypatch):
    runtime = tmp_path/'python.exe'
    runtime.touch()
    output = tmp_path/'results'
    calls = []
    monkeypatch.setattr(launch, 'controller_probe', lambda: {'ffmpeg': 'unused'})

    def run_logged(command, log_path, env):
        if '--case' not in command:
            return 1
        case = command[command.index('--case')+1]
        run_id = command[command.index('--run-id')+1]
        calls.append(case)
        write_case(output, case=case, run_id=run_id)
        write_preflight(output, run_id=run_id)
        log_path.write_text('Windows fatal exception: access violation\n')
        return 3221225477

    monkeypatch.setattr(launch, 'run_logged', run_logged)
    assert launch.main(['--isaac-python', str(runtime), '--suite', 'full', '--output', str(output), '--no-video']) == 1
    manifest = json.loads((output/'suite_manifest.json').read_text())
    assert calls == ['nominal']
    assert manifest['cases'][0]['process_exit_code'] == 3221225477
    assert [entry['status'] for entry in manifest['cases'][1:]] == ['blocked']*6


def test_all_sample_rms_keeps_distinct_scope(tmp_path):
    path, metadata, arrays = make_case(tmp_path, duration=2.0)
    arrays['q'][100, 1] = 0.01
    from test_report import save
    save(path, metadata, arrays)
    metrics = summarize_case(path)['metrics']
    assert metrics['rms_body_error_after_2s_samples'] == 1
    assert metrics['rms_body_error_m'] == 0
    assert metrics['rms_body_error_all_samples_m'] > 0


def test_explicit_no_video_smoke_is_diagnostic_only(tmp_path, monkeypatch):
    make_case(tmp_path, duration=2)
    make_preflight(tmp_path)
    edit_manifest(tmp_path, video_requested=False)
    (tmp_path/'nominal/movie.mp4').unlink()
    monkeypatch.setattr('isaac_validation.report._plots', lambda *a: ([], None))
    assert report_main(['--output', str(tmp_path), '--suite', 'smoke', '--run-id', 'test-only']) == 0
    report = json.loads((tmp_path/'validation.json').read_text())
    assert report['smoke_passed'] and not report['full_validation']
    assert not report['cases']['nominal']['video']['passed']


@pytest.mark.parametrize('problem', [None, 'missing_audit', 'changed_time', 'changed_velocity', 'wrong_hash', 'missing_cleanup'])
def test_revision2_requires_real_cleanup_and_capture_audit_fields(tmp_path, problem):
    import hashlib
    from test_report import save
    path, metadata, arrays = make_case(tmp_path, duration=2.0)
    metadata.update(schema_version=2, video_requested=True,
                    resource_cleanup={'passed': True, 'steps': [{'name': 'synthetic_cleanup', 'passed': True}], 'errors': []})
    mode = 'render'
    def guard():
        return dict(passed=True, max_abs_dq=0., max_abs_dv=0., abs_dt_s=0.)
    audit = dict(mode=mode, run_id=metadata['run_id'], physics_unchanged=True,
                 frame_count=50, warmup=guard(), frames=[dict(time_s=i/25, **guard()) for i in range(50)])
    if problem == 'changed_time': audit['frames'][17]['abs_dt_s'] = 0.001
    if problem == 'changed_velocity': audit['frames'][17]['max_abs_dv'] = 0.001
    audit_path = path/'capture_timing.json'
    audit_path.write_text(json.dumps(audit))
    metadata['capture_audit'] = dict(sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                                    frames=50, mode=mode, physics_unchanged=True)
    metadata['video']['capture_mode'] = mode
    if problem == 'missing_audit': audit_path.unlink()
    if problem == 'wrong_hash': metadata['capture_audit']['sha256'] = 'wrong'
    if problem == 'missing_cleanup': metadata.pop('resource_cleanup')
    save(path, metadata, arrays)
    result = summarize_case(path)
    assert result['video']['passed'] is (problem is None)


def test_nominal_scope_cannot_satisfy_requested_video_with_missing_movie(tmp_path, monkeypatch):
    make_case(tmp_path)
    make_preflight(tmp_path)
    (tmp_path/'nominal/movie.mp4').unlink()
    monkeypatch.setattr('isaac_validation.report._plots', lambda *a: ([],None))
    assert report_main(['--output',str(tmp_path),'--suite','nominal','--run-id','test-only']) == 1


@pytest.mark.parametrize('problem', [None, 'warmup_not_ready', 'no_samples', 'wrong_shape',
                                     'failed_after_good_frame', 'disabled_playback_capture',
                                     'downgraded_audit', 'missing_ready_metadata'])
def test_revision3_requires_image_readiness_not_just_unchanged_physics(tmp_path, problem):
    import hashlib
    from test_report import save
    path, metadata, arrays = make_case(tmp_path, duration=2.0)
    metadata.update(schema_version=2, code_revision=3, capture_audit_schema_version=2, video_requested=True,
                    resource_cleanup={'passed': True, 'steps': [{'name': 'synthetic', 'passed': True}], 'errors': []})
    def guard():
        return dict(passed=True, rgb_ready=True, max_abs_dq=0., max_abs_dv=0., abs_dt_s=0.,
                    render_passes=4, max_render_passes=64, last_rgb_shape=[2,4,4], last_rgb_dtype='uint8',
                    samples=[dict(render_pass=4, shape=[2,4,4], dtype='uint8', empty=False)])
    audit = dict(schema_version=2, mode='render', run_id=metadata['run_id'], physics_unchanged=True,
                 frame_count=50, camera_ready=True, failed=False, resolution=[4,2], annotator_device='cpu',
                 capture_on_play=True, warmup=guard(), frames=[dict(time_s=i/25, **guard()) for i in range(50)])
    if problem == 'warmup_not_ready': audit['warmup']['rgb_ready'] = False
    if problem == 'no_samples': audit['warmup']['samples'] = []
    if problem == 'wrong_shape': audit['frames'][2]['samples'][-1]['shape'] = [0]
    if problem == 'failed_after_good_frame': audit['failed'] = True
    if problem == 'disabled_playback_capture': audit['capture_on_play'] = False
    if problem == 'downgraded_audit': audit['schema_version'] = 1
    audit_path = path/'capture_timing.json'
    audit_path.write_text(json.dumps(audit))
    metadata['capture_audit'] = dict(sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                                    frames=50, mode='render', physics_unchanged=True,
                                    schema_version=2, camera_ready=True)
    if problem == 'missing_ready_metadata': metadata['capture_audit'].pop('camera_ready')
    metadata['video']['capture_mode'] = 'render'
    save(path, metadata, arrays)
    result = summarize_case(path)
    assert result['video']['checks']['camera_rgb_readiness']['passed'] is (problem is None)
    assert result['video']['passed'] is (problem is None)
