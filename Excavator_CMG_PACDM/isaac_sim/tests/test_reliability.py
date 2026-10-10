"""Synthetic regression tests, not evidence of native Isaac mission completion."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from measurement_store import CheckpointStore, atomic_json, atomic_npz, assemble_measurements
from result_status import recover_status
from replay_video import select_frames, prepare_plan, render_video, file_hash
from render_replay_worker import write_rgb_png, pose_digest
from video_capture import _validate_png
from launch import assess_native_result, make_results_archive
from assess_results import assess_video
from process_utils import python_command


def state_row(t, n=2, closures=1, gears=1):
    return (t, np.full((n,3),t), np.tile([1.,0,0,0],(n,1)),
            np.full((n,3),t+1), np.zeros((n,3)), t+2, t+3, t+4,
            np.zeros(closures), np.zeros(gears))


def control_row(t, n=2):
    return (t, np.array([t, t+1]), np.full((n,3), t), np.full((n,3), -t))


def create_replay_result(tmp_path):
    out = tmp_path/'result'
    out.mkdir()
    times = np.r_[0.,.001,np.arange(.01,.311,.01)]
    n = 2
    p = np.zeros((len(times),n,3))
    p[:,:,0] = times[:,None]
    q = np.tile([1.,0,0,0],(len(times),n,1))
    atomic_npz(out/'states.npz', time=times, positions=p, quaternions_wxyz=q,
               body_names=np.array(['a','b']))
    atomic_json(out/'manifest.json', {'dt': .001, 'bodies':[{'name':'a'}, {'name':'b'}]})
    atomic_json(out/'status.json', {'dt':.001, 'native_steps_completed':310,
                                   'status':'NATIVE_RUN_COMPLETED_UNASSESSED'})
    scene = tmp_path/'scene.usda'
    scene.write_text('#usda 1.0\n')
    return out, scene


def test_atomic_json_rejects_nan_without_replacing(tmp_path):
    path = tmp_path/'status.json'
    atomic_json(path, {'value':1})
    with pytest.raises(ValueError):
        atomic_json(path, {'value':float('nan')})
    assert json.loads(path.read_text()) == {'value':1}


def test_chunk_roundtrip_and_buffer_release(tmp_path):
    store = CheckpointStore(tmp_path, ['a','b'], .001)
    expected_rows = [state_row(t) for t in (0,.001,.002,.003)]
    controls_all = [control_row(t) for t in (0.,.001,.002)]
    rows, controls, contact = expected_rows[:2], controls_all[:1], [(.001,np.ones((2,3)))]
    status = {'native_steps_completed':1,'simulated_duration':.001,'status':'RUNNING'}
    store.flush(rows, controls, contact, status)
    assert rows == controls == contact == []
    rows.extend(expected_rows[2:]); controls.extend(controls_all[1:])
    contact.append((.003, np.full((2,3),2.)))
    status.update(native_steps_completed=3, simulated_duration=.003)
    store.flush(rows, controls, contact, status)
    report = assemble_measurements(tmp_path)
    assert report['errors'] == []
    with np.load(tmp_path/'states.npz') as data:
        assert np.array_equal(data['positions'], np.array([r[1] for r in expected_rows]))
        assert np.array_equal(data['linear_velocities'], data['linear_velocities_origin_world'])
        assert data['body_names'].tolist() == ['a','b']
    with np.load(tmp_path/'applied_wrenches.npz') as data:
        assert np.array_equal(data['time'], [0.,.001,.002])
        assert np.array_equal(data['forces_world'], [r[2] for r in controls_all])
    with np.load(tmp_path/'native_contact_forces.npz') as data:
        assert data['reported_net_contact_forces_world'].shape == (2,2,3)
        assert float(data['dt']) == .001
    assert report['last_committed_status']['native_steps_completed'] == 3
    assert len(list((tmp_path/'checkpoints').glob('*.npz'))) == 2
    # The same exact evidence can be reassembled without duplicate samples.
    assert assemble_measurements(tmp_path)['errors'] == []


def test_empty_closure_and_gear_axes(tmp_path):
    store = CheckpointStore(tmp_path, ['a','b'], .001)
    store.flush([state_row(0.,closures=0,gears=0)], [], [], {'native_steps_completed':0})
    assert assemble_measurements(tmp_path)['errors'] == []
    with np.load(tmp_path/'states.npz') as data:
        assert data['closure_gaps_m'].shape == (1,0)


def test_partial_tmp_is_not_a_committed_chunk(tmp_path):
    store = CheckpointStore(tmp_path,['a','b'],.001)
    store.flush([state_row(0.)],[],[],{'native_steps_completed':0})
    (tmp_path/'checkpoints'/'chunk_000001.npz.tmp').write_bytes(b'interrupted-write')
    report = assemble_measurements(tmp_path)
    assert report['chunks'] == 1 and not report['errors']


def test_checkpoint_numbering_gap_rejected(tmp_path):
    store = CheckpointStore(tmp_path,['a','b'],.001)
    store.flush([state_row(0.)],[],[],{})
    (tmp_path/'checkpoints'/'chunk_000000.npz').rename(tmp_path/'checkpoints'/'chunk_000001.npz')
    assert assemble_measurements(tmp_path)['errors']
    assert not (tmp_path/'states.npz').exists()


def test_overlapping_chunk_times_rejected(tmp_path):
    store = CheckpointStore(tmp_path,['a','b'],.001)
    store.flush([state_row(.001)],[],[],{})
    store.flush([state_row(.001)],[],[],{})
    assert any('Overlapping' in e for e in assemble_measurements(tmp_path)['errors'])


def test_crashed_result_is_not_zero_steps_or_success(tmp_path):
    launcher = {'physics_dt':.001,'duration_requested_s':65.,'full_mission_requested':True,'video_requested':True}
    (tmp_path/'bridge_diagnostics.jsonl').write_text(
        json.dumps({'state_time':5.69,'bridge_response':{'metrics':{'done':False,'phase':'lower below surface'}}})+'\n{truncated')
    status = recover_status(tmp_path,launcher,3221225477)
    assert status['native_steps_completed'] is None
    assert status['observed_native_steps_lower_bound'] == 5690
    assert status['status'] == 'NATIVE_PROCESS_INCOMPLETE'
    assert status['stop_when_mission_done'] and status['video_requested']
    assert status['last_observed_mission_metrics']['done'] is False


def test_recovery_uses_last_committed_progress(tmp_path):
    atomic_json(tmp_path/'status.json',{'status':'STARTED','native_steps_completed':0})
    checkpoint = {'status':'RUNNING','native_steps_completed':250,'simulated_duration':.25,'dt':.001}
    status = recover_status(tmp_path,{'physics_dt':.001},1,{'last_committed_status':checkpoint})
    assert status['native_steps_completed'] == 250
    assert status['status'] == 'NATIVE_PROCESS_INCOMPLETE'
    assert 'lower bound' in status['recovery_notes'][0]


def test_missing_measurements_not_described_as_measured_violations(tmp_path):
    atomic_json(tmp_path/'status.json',{'status':'NATIVE_PROCESS_INCOMPLETE','native_steps_completed':None})
    issues = assess_native_result(tmp_path,False,65.,.001,1,True,False)
    assert any('fallback count is unavailable' in i for i in issues)
    assert any('Arm closure measurement unavailable' in i for i in issues)
    assert not any('used a fallback' in i or 'outside declared bound' in i or 'exceeds the declared' in i for i in issues)


def test_frame_selection_is_measured_only_and_includes_terminal():
    take, steps = select_frames(np.array([0,.001,.01,.1,.2,.24]),.001,100)
    assert take.tolist() == [1,3,4,5]
    assert steps.tolist() == [1,100,200,240]
    with pytest.raises(ValueError):
        select_frames(np.array([0,.001,.001]),.001,100)
    with pytest.raises(ValueError):
        select_frames(np.array([0,.001,.0105]),.001,100)


def test_png_writer_and_resolution_check(tmp_path):
    path = tmp_path/'a.png'
    rgb = np.arange(10*20*4,dtype=np.uint8).reshape(10,20,4)
    digest = write_rgb_png(path,rgb)
    _validate_png(path,(20,10))
    assert digest == file_hash(path)
    from PIL import Image
    assert np.array_equal(np.asarray(Image.open(path)),rgb[:,:,:3])
    with pytest.raises(RuntimeError):
        _validate_png(path,(10,20))


def test_video_plan_hashes_exact_native_poses(tmp_path):
    out, scene = create_replay_result(tmp_path)
    plan = prepare_plan(out,scene)
    assert [f['native_step'] for f in plan['frames']] == [1,100,200,300,310]
    assert plan['source_states_sha256'] == file_hash(out/'states.npz')
    with np.load(out/'states.npz') as source:
        for f in plan['frames']:
            i=f['state_sample_index']
            assert f['pose_sha256'] == pose_digest(source['positions'][i],source['quaternions_wxyz'][i])
    assert prepare_plan(out,scene) == plan
    with pytest.raises(ValueError):
        prepare_plan(out,scene,fps=30)


def fake_render_worker(command, **kwargs):
    result = Path(command[command.index('--result')+1])
    start,stop = [int(command[command.index(key)+1]) for key in ('--start','--stop')]
    plan=json.loads((result/'video_plan.json').read_text())
    (result/'frames').mkdir(exist_ok=True)
    (result/'frame_records').mkdir(exist_ok=True)
    width,height=plan['resolution']
    for i in range(start,stop):
        image=np.zeros((height,width,3),dtype=np.uint8)
        image[:,:,0]=(i*40)%255
        image[:,(i*15)%width:((i*15)%width)+5,1]=255
        digest=write_rgb_png(result/'frames'/('frame_%06d.png'%i),image)
        record=dict(plan['frames'][i],png_sha256=digest,source_states_sha256=plan['source_states_sha256'],
                    mode=plan['mode'],capture_timeline_delta_s=0.0)
        atomic_json(result/'frame_records'/('frame_%06d.json'%i),record)
    return SimpleNamespace(returncode=0)


def test_batched_video_and_native_state_integrity(tmp_path):
    out,scene=create_replay_result(tmp_path)
    before=file_hash(out/'states.npz')
    with patch('replay_video.subprocess.run',side_effect=fake_render_worker) as run:
        # Encoder also uses subprocess.run; mock just the encoder for this test.
        with patch('replay_video.encode_png_sequence',return_value={'mp4_created':True}):
            capture,encoding=render_video(out,sys.executable,scene,batch_frames=2,renderer='isaac')
    assert run.call_count == 3
    assert capture['png_complete'] and encoding['mp4_created']
    assert capture['captured_frames'] == 5
    assert file_hash(out/'states.npz') == before
    assert json.loads((out/'status.json').read_text())['status'] == 'NATIVE_RUN_COMPLETED_UNASSESSED'


def test_failed_render_is_bounded_and_keeps_physics(tmp_path):
    out,scene=create_replay_result(tmp_path)
    before=file_hash(out/'states.npz')
    with patch('replay_video.subprocess.run',return_value=SimpleNamespace(returncode=1)) as run:
        cap,enc=render_video(out,sys.executable,scene,renderer='isaac')
    assert run.call_count == 2
    assert not cap['png_complete'] and not enc['mp4_created']
    assert file_hash(out/'states.npz') == before
    assert json.loads((out/'status.json').read_text())['status'] == 'NATIVE_RUN_COMPLETED_UNASSESSED'


def test_video_resume_skips_saved_frames(tmp_path):
    out,scene=create_replay_result(tmp_path)
    plan=prepare_plan(out,scene)
    fake_render_worker(['--result',str(out),'--start','0','--stop','2'])
    first=file_hash(out/'frames'/'frame_000000.png')
    with patch('replay_video.subprocess.run',side_effect=fake_render_worker) as run:
        with patch('replay_video.encode_png_sequence',return_value={'mp4_created':True}):
            cap,_=render_video(out,sys.executable,scene,batch_frames=2,renderer='isaac')
    first_cmd=run.call_args_list[0].args[0]
    assert first_cmd[first_cmd.index('--start')+1] == '2'
    assert cap['png_complete'] and file_hash(out/'frames'/'frame_000000.png') == first


def test_video_provenance_rejects_changed_pose(tmp_path):
    out,scene=create_replay_result(tmp_path)
    with patch('replay_video.subprocess.run',side_effect=fake_render_worker):
        with patch('replay_video.encode_png_sequence',return_value={'mp4_created':True}):
            render_video(out,sys.executable,scene,renderer='isaac')
    # Synthetic MP4 container marker, only for testing the provenance gate.
    (out/'isaac_native.mp4').write_bytes(b'\x00\x00\x00\x18ftypmp42'+b'0'*2048)
    status=json.loads((out/'status.json').read_text())
    issues=[]
    assert assess_video(out,status,{'video_requested':True},issues)['video_evidence_available']
    meta=json.loads((out/'video_frames.json').read_text())
    meta['frames'][0]['pose_sha256']='not-the-measured-pose'
    atomic_json(out/'video_frames.json',meta)
    issues=[]
    assert not assess_video(out,status,{'video_requested':True},issues)['video_evidence_available']
    assert any('provenance failed' in i for i in issues)


def test_zip_reports_missing_arrays(tmp_path):
    out=tmp_path/'result';out.mkdir()
    (out/'status.json').write_text('{}')
    (out/'checkpoints').mkdir()
    (out/'checkpoints'/'chunk_000000.npz').write_bytes(b'placeholder')
    zip_path=make_results_archive(out,compact=False)
    with zipfile.ZipFile(zip_path) as archive:
        record=json.loads(archive.read('result/archive_contents.json'))
        assert record['full_native_measurement_arrays_included'] is False
        assert len(record['missing_native_measurement_arrays']) == 3
        assert not any('/checkpoints/' in name for name in archive.namelist())


def test_native_runner_has_no_live_capture_call():
    tree=ast.parse((ROOT/'run_isaac.py').read_text())
    imports=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n.module in ('video_capture','omni.replicator.core') for n in imports)
    assert 'video.capture(' not in (ROOT/'run_isaac.py').read_text()


def test_windows_python_exe_argument_spaces(tmp_path):
    command=python_command(tmp_path/'python.exe',tmp_path/'folder with spaces'/'script.py',['--output','C:\\Users\\A B'])
    assert isinstance(command,list) and command[-1]=='C:\\Users\\A B'


def test_real_ffmpeg_encode_and_decode_frame_count(tmp_path):
    """Real encoder test on synthetic pixels; no Isaac rendering is implied."""
    import subprocess
    from encode_video import encode_png_sequence, _find_ffmpeg
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('FFmpeg unavailable in this test environment')
    frames = tmp_path/'frames'
    frames.mkdir()
    width, height, count = 64, 36, 6
    for i in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, i*8:i*8+8, :] = 255
        write_rgb_png(frames/('frame_%06d.png' % i), image)
    movie = tmp_path/'test.mp4'
    result = encode_png_sequence(frames, movie, fps=20, expected_count=count)
    assert result['mp4_created'], result
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(movie),
                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                             capture_output=True, timeout=30, check=True)
    assert len(decoded.stdout) == width*height*3*count
    assert len(list(frames.glob('frame_*.png'))) == count


def test_zero_step_placeholder_is_discarded_when_logs_prove_progress(tmp_path):
    atomic_json(tmp_path/'status.json', {'status':'FAILED', 'native_steps_completed':0})
    (tmp_path/'bridge_diagnostics.jsonl').write_text(
        json.dumps({'state_time':5.69, 'bridge_response':{'metrics':{'done':False}}})+'\n')
    result = recover_status(tmp_path, {'physics_dt':.001, 'full_mission_requested':True}, 1)
    assert result['native_steps_completed'] is None
    assert result['observed_native_steps_lower_bound'] == 5690
    assert result['stop_when_mission_done'] is True
