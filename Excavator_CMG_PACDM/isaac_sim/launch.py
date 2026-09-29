"""One-command coordinator; use the controller environment's Python.

Example: python launch.py --isaac-python C:\\isaacsim\\python.bat
Native execution uses Isaac's own Python; dependencies stay in separate envs.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np
from measurement_store import atomic_json, assemble_measurements
from result_status import recover_status
from process_utils import python_command

ROOT=Path(__file__).resolve().parent


def make_results_archive(out, compact, archive_path=None):
    """Package results as a ZIP; a compact archive holds sampled arrays and a summary."""
    archive=Path(archive_path) if archive_path else out.with_suffix('.zip')
    if compact:
        frames=sorted((out/'frames').glob('frame_*.png'))
        if len(frames)>=2:
            preview_dir=out/'preview_frames'
            preview_dir.mkdir(exist_ok=True)
            for label,index in (('first',0),('middle',len(frames)//2),('last',len(frames)-1)):
                shutil.copy2(frames[index],preview_dir/(label+'.png'))
        hashes={}
        for filename in ('states.npz','applied_wrenches.npz','native_contact_forces.npz',
                         'bridge_diagnostics.jsonl','isaac_native.mp4'):
            path=out/filename
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        h.update(block)
                hashes[filename]={'sha256':h.hexdigest(),'bytes':path.stat().st_size}
        (out/'full_artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
        sampling_errors=[]
        for filename in ('states.npz','applied_wrenches.npz','native_contact_forces.npz'):
            original=out/filename
            if not original.is_file():continue
            try:
                with np.load(original,allow_pickle=False) as data:
                    n=len(data['time'])
                    take=np.unique(np.linspace(0,n-1,min(501,n),dtype=np.int64))
                    sampled={key:(data[key][take] if key != 'body_names' and
                              data[key].ndim>0 and data[key].shape[0]==n else data[key])
                             for key in data.files}
                    sampled['source_full_sha256']=np.asarray(hashes[filename]['sha256'])
                    np.savez_compressed(out/('summary_'+filename),**sampled)
            except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as exc:
                sampling_errors.append('%s: %s: %s' % (filename,type(exc).__name__,exc))
        diagnostics=out/'bridge_diagnostics.jsonl'
        if diagnostics.is_file():
            with (diagnostics.open(encoding='utf-8') as source,
                  (out/'summary_bridge_diagnostics.jsonl').open('w',encoding='utf-8') as target):
                last=''
                for index,line in enumerate(source):
                    last=line
                    if index%20==0:target.write(line)
                if last and (not last.endswith('\n') or index%20):target.write(last)
        if sampling_errors:
            (out/'summary_sampling_errors.json').write_text(json.dumps(sampling_errors,indent=2)+'\n',encoding='utf-8')
    def package(skip_movie=False,skip_full_diagnostics=False):
        movie=out/'isaac_native.mp4'
        include_movie=movie.is_file() and not skip_movie and (not compact or movie.stat().st_size<=20*1024*1024)
        archive_info={'scope':'compact summary' if compact else 'complete consolidated result artifacts (without redundant checkpoint copies)',
            'full_native_measurement_arrays_included':bool(not compact and all((out/name).is_file() for name in
                ('states.npz','applied_wrenches.npz','native_contact_forces.npz'))),
            'missing_native_measurement_arrays':[name for name in ('states.npz','applied_wrenches.npz',
                'native_contact_forces.npz') if not (out/name).is_file()],
            'full_native_arrays_remain_in_local_result_directory':bool(compact),
            'sampled_measurements_are_not_sufficient_for_independent_full_audit':bool(compact),
            'mp4_included':include_movie,
            'mp4_remains_in_local_result_directory':bool(movie.is_file() and not include_movie),
            'full_artifact_hashes':'full_artifact_hashes.json' if compact else None,
            'omitted_from_compact_archive':['states.npz','applied_wrenches.npz','native_contact_forces.npz'] +
                (['isaac_native.mp4'] if movie.is_file() and not include_movie else []) if compact else [],
            'full_bridge_diagnostics_included':bool((out/'bridge_diagnostics.jsonl').is_file() and not skip_full_diagnostics)}
        (out/'archive_contents.json').write_text(json.dumps(archive_info,indent=2)+'\n',encoding='utf-8')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(out.rglob('*')):
                if not file.is_file():continue
                rel=file.relative_to(out)
                # Exact originals are consolidated above. Do not duplicate raw
                # chunks, staging arrays, or partial atomic writes in a ZIP.
                if ('checkpoints' in rel.parts or any(part.startswith('.assemble_') for part in rel.parts) or
                    file.name == 'replay_states.npz' or file.name.endswith('.tmp')):
                    continue
                if compact and (file.name in ('states.npz','applied_wrenches.npz','native_contact_forces.npz') or
                                'frames' in rel.parts or 'replicator_rgb' in rel.parts or
                                (file.name=='isaac_native.mp4' and (skip_movie or file.stat().st_size>20*1024*1024)) or
                                (file.name=='bridge_diagnostics.jsonl' and skip_full_diagnostics)):
                    continue
                z.write(file,Path(out.name)/rel)
    package()
    if compact and archive.stat().st_size>30*1024*1024:
        package(skip_movie=True)
    if compact and archive.stat().st_size>30*1024*1024:
        package(skip_movie=True,skip_full_diagnostics=True)
    return archive


def relay(stream, log, ready):
    for line in iter(stream.readline,''):
        log.write(line);log.flush();print(line,end='',flush=True)
        if line.startswith('EXCAVATOR_BRIDGE_READY '): ready.put(True)


def assess_native_result(out, preflight, duration, dt, native_code, full_mission=False, video_requested=False):
    """Accept only a completed native run with its required measurements."""
    issues = []
    status_file = out/'status.json'
    try:
        status = json.loads(status_file.read_text(encoding='utf-8'))
        if not isinstance(status, dict):
            raise ValueError('Native status is not a JSON object')
    except (OSError, ValueError) as exc:
        return ['Native status unavailable; executed step count is unknown: '+str(exc)]
    if native_code != 0:
        issues.append('Isaac process exited with code '+str(native_code))
    if status.get('physx_error_events'):
        issues.append('Native status contains PhysX error events')
    if status.get('status') in ('PREFLIGHT_COMPLETED_NO_DYNAMIC_VALIDATION', 'NATIVE_RUN_COMPLETED_UNASSESSED'):
        probe = status.get('articulation_probe')
        observed = probe.get('observed_articulation') if isinstance(probe, dict) else None
        expected = probe.get('expected_articulation') if isinstance(probe, dict) else None
        if (not isinstance(probe, dict) or
            not isinstance(observed, dict) or not isinstance(expected, dict) or
            observed != expected or observed.get('count') != 1 or
            not isinstance(probe.get('sampled_runtime_object_types'), dict)):
            issues.append('Native articulation diagnostic or confirmed link/DOF inventory is missing')
    if preflight:
        if (status.get('status') != 'PREFLIGHT_COMPLETED_NO_DYNAMIC_VALIDATION' or
            status.get('preflight_passed') is not True or status.get('native_physics_initialized') is not True or
            status.get('native_steps_completed') != 0 or status.get('native_runtime_executed') is not False):
            issues.append('Native preflight did not finish and confirm physics initialization')
    else:
        maximum_steps = round(duration/dt)
        expected_steps = status.get('native_steps_completed') if full_mission else maximum_steps
        elapsed = status.get('simulated_duration')
        if (type(expected_steps) is not int or expected_steps < 1 or
            expected_steps > maximum_steps or
            not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or
            not math.isclose(elapsed, expected_steps*dt, abs_tol=1e-8)):
            issues.append('Native step count or elapsed time is invalid')
            expected_steps = maximum_steps
        if (status.get('status') != 'NATIVE_RUN_COMPLETED_UNASSESSED' or
            status.get('native_steps_completed') != expected_steps or
            status.get('native_runtime_executed') is not True or
            status.get('native_physics_initialized') is not True or
            status.get('initial_rigid_state_validated_at_t0') is not True or
            status.get('prestep_articulation_confirmed') is not True or
            status.get('physx_start_simulation_called') is not True or
            status.get('first_controlled_step_articulation_confirmed') is not True or
            (status.get('first_controlled_step') or {}).get('physx_error_free_before_step') is not True or
            status.get('uncontrolled_warmup_steps') != 0 or
            status.get('mujoco_integration_steps') != 0 or
            status.get('state_writes_after_initialization') != 0):
            issues.append('Native run did not confirm all %d controlled physics intervals' % expected_steps)
        if full_mission:
            summary = status.get('bridge_summary') or {}
            metrics = summary.get('last_metrics') or {}
            events = metrics.get('events') or []
            phase_names = ('settle soil', 'lower below surface', 'draw through soil',
                           'curl bucket', 'lift clear', 'hold load', 'slew to receiver',
                           'dump material', 'wait for deposition', 'close bucket', 'return')
            if (status.get('stop_when_mission_done') is not True or
                metrics.get('done') is not True or metrics.get('failures') or
                [e.get('phase') for e in events if isinstance(e, dict)] != list(phase_names)):
                issues.append('The guarded 11-phase digging mission did not finish without failures')
            for metric in ('lifted_mass_kg', 'deposited_mass_kg'):
                mass = metrics.get(metric)
                if not isinstance(mass, (int, float)) or not math.isfinite(mass):
                    issues.append('Terminal soil-mass measurement unavailable: '+metric)
                elif mass < 1.0:
                    issues.append('Terminal '+metric+' is below the required 1 kg')
            controller=summary.get('controller') or {}
            if 'fallback_count' not in controller:
                issues.append('Terminal PACDM fallback count is unavailable')
            elif controller['fallback_count'] != 0:
                issues.append('Terminal PACDM report records fallback use')
            if 'native_state_projection' not in controller:
                issues.append('Terminal PACDM native-state projection field is unavailable')
            elif controller['native_state_projection'] is not False:
                issues.append('Terminal PACDM report does not confirm projection-free operation')
            for name,limit in (('pacdm_closure_max',1e-8),
                               ('pacdm_tangent_max',1e-8),
                               ('inverse_equilibrium_relative_max',1e-6)):
                value=controller.get(name)
                if not isinstance(value,(int,float)) or not math.isfinite(value):
                    issues.append('PACDM residual measurement unavailable or nonfinite: '+name)
                elif value > limit:
                    issues.append('PACDM controller residual outside declared bound: '+name)
            arm_gap = status.get('max_arm_closure_gap_m')
            track_gap = status.get('max_track_closure_gap_m')
            for label, gap, limit in (('Arm', arm_gap, 1e-4), ('Track', track_gap, 1e-3)):
                if not isinstance(gap,(int,float)) or not math.isfinite(gap):
                    issues.append(label+' closure measurement unavailable or nonfinite')
                elif gap > limit:
                    issues.append(label+' closure residual exceeds the declared structural limit')
            contact_path=out/'native_contact_forces.npz'
            if not contact_path.is_file():
                issues.append('Native net contact-force measurements are missing')
            else:
                try:
                    with np.load(contact_path,allow_pickle=False) as contact:
                        times=np.asarray(contact['time'])
                        forces=np.asarray(contact['reported_net_contact_forces_world'])
                        if (times.ndim!=1 or len(times)<2 or
                            forces.shape!=(len(times),248,3) or
                            not np.all(np.isfinite(forces)) or
                            not np.any(np.abs(forces)>1e-6)):
                            raise ValueError('contact measurements are missing, invalid or all zero')
                except (OSError,ValueError,KeyError,EOFError) as exc:
                    issues.append('Invalid native contact-force measurement: '+str(exc))
        for name in ('states.npz', 'applied_wrenches.npz', 'bridge_diagnostics.jsonl'):
            path = out/name
            if not path.is_file() or path.stat().st_size == 0:
                issues.append('Missing native measurement: '+name)
                continue
            try:
                if name.endswith('.npz'):
                    with np.load(path, allow_pickle=False) as data:
                        times = np.asarray(data['time'])
                        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
                            raise ValueError('Invalid measurement time array')
                        if name == 'states.npz' and (times.size < 3 or
                            not math.isclose(float(times[0]), 0.0, abs_tol=1e-9) or
                            not math.isclose(float(times[1]), dt, abs_tol=1e-9) or
                            not math.isclose(float(times[-1]), expected_steps*dt, abs_tol=1e-9)):
                            raise ValueError('State measurements do not span the requested duration')
                        if name == 'applied_wrenches.npz' and (
                            times.size != expected_steps or
                            not math.isclose(float(times[0]), 0.0, abs_tol=1e-9) or
                            not math.isclose(float(times[-1])+dt, expected_steps*dt, abs_tol=1e-9) or
                            not np.allclose(times,np.arange(expected_steps)*dt,rtol=0.,atol=1e-9)):
                            raise ValueError('Effort measurements do not reach the final step')
                else:
                    with path.open(encoding='utf-8') as stream:
                        last = None
                        for line in stream:
                            if line.strip():
                                last = json.loads(line)
                    if not isinstance(last, dict) or last.get('terminal') is not True:
                        raise ValueError('Missing terminal controller record')
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                issues.append('Invalid native measurement %s: %s' % (name, exc))
    if video_requested:
        capture = status.get('video_capture') or {}
        encoding_path = out/'video_encoding.json'
        try:
            encoding = json.loads(encoding_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            encoding = {'mp4_created':False, 'error':str(exc)}
        frame_manifest = out/'video_frames.json'
        if (capture.get('png_complete') is not True or
            type(capture.get('captured_frames')) is not int or
            capture['captured_frames'] < 2 or
            encoding.get('mp4_created') is not True or
            not (out/'isaac_native.mp4').is_file()):
            issues.append('A complete native-frame MP4 was not produced')
        frames=sorted((out/'frames').glob('frame_*.png'))
        if len(frames)>=2:
            digests=[]
            for frame in (frames[0],frames[len(frames)//2],frames[-1]):
                digests.append(hashlib.sha256(frame.read_bytes()).hexdigest())
            if len(set(digests))<2:
                issues.append('Sampled native video frames are identical')
        try:
            frames = json.loads(frame_manifest.read_text(encoding='utf-8'))
            stamps = [item['sim_time_s'] for item in frames['frames']]
            if (len(stamps) != capture.get('captured_frames') or
                not all(a < b for a,b in zip(stamps,stamps[1:])) or
                not math.isclose(stamps[-1],float(status.get('simulated_duration',-1)),abs_tol=1e-8)):
                issues.append('Captured video timestamps do not span the native run')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append('Native video frame provenance is invalid: '+str(exc))
    return issues


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--isaac-python', type=Path, required=True)
    p.add_argument('--case', choices=['soil_final'], default='soil_final')
    p.add_argument('--duration', type=float, default=None)
    p.add_argument('--full-mission', action='store_true', help='Run all guarded phases, at most 65 simulation seconds')
    p.add_argument('--video', action='store_true', help='Render saved native states AFTER physics exits')
    p.add_argument('--video-renderer', choices=('cpu', 'isaac'), default='cpu',
                   help='cpu (default): NumPy renderer in this controller environment; '
                        'isaac: RTX/Replicator worker processes')
    p.add_argument('--grain-rolling-friction', type=float, default=None,
                   help='Soil-grain rolling-resistance coefficient in metres, applied as explicit '
                        'torques because PhysX has none; default derives 0.008 m from the manifest, 0 disables it')
    p.add_argument('--dt', type=float, default=.001)
    p.add_argument('--headless', action='store_true', help='Disable live viewport rendering (recommended)')
    p.add_argument('--checkpoint-dt', type=float, default=.25)
    p.add_argument('--video-batch-frames', type=int, default=32)
    p.add_argument('--capture-stride', type=int, default=100)
    p.add_argument('--full-results-zip', action='store_true', help='Also ZIP all consolidated native arrays and video frames')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    args.duration = (65. if args.full_mission else 2.) if args.duration is None else args.duration
    if (not math.isfinite(args.dt) or abs(args.dt-.001) > 1e-12 or
        not math.isfinite(args.duration) or args.duration <= 0 or args.duration > 65 or
        not math.isclose(round(args.duration/args.dt)*args.dt, args.duration, abs_tol=1e-9) or
        (not args.full_mission and args.duration != 2.)):
        p.error('Use a two-second smoke test or --full-mission (cap <=65 s), with dt=0.001')
    if (not math.isfinite(args.checkpoint_dt) or args.checkpoint_dt <= 0 or
        args.video_batch_frames < 1 or args.capture_stride < 1):
        p.error('Checkpoint interval, video batch size and capture stride must be positive')
    if args.grain_rolling_friction is not None and (not math.isfinite(args.grain_rolling_friction) or
                                                    args.grain_rolling_friction < 0):
        p.error('--grain-rolling-friction must be finite and nonnegative')
    exe = args.isaac_python.resolve()
    if not exe.is_file():
        p.error('Isaac Python launcher not found: '+str(exe))
    manifest = ROOT/'generated'/args.case/'manifest.json'
    if not manifest.is_file() or abs(json.loads(manifest.read_text()).get('physics_dt', -1)-args.dt)>1e-12:
        p.error('Packaged manifest is missing or has a different timestep')
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = (args.output or ROOT/'results'/f'{args.case}_{timestamp}').resolve()
    out.mkdir(parents=True, exist_ok=False)
    launcher = {'case': args.case, 'physics_dt': args.dt, 'duration_requested_s': args.duration,
        'full_mission_requested': args.full_mission, 'video_requested': args.video,
        'video_mode': 'post_run_measured_state_replay' if args.video else 'disabled',
        'video_renderer': args.video_renderer if args.video else None,
        'grain_rolling_friction_request_m': ('derived from manifest' if args.grain_rolling_friction is None
                                             else args.grain_rolling_friction),
        'isaac_launcher': str(exe), 'controller_python': sys.executable,
        'checkpoint_interval_sim_s': args.checkpoint_dt, 'headless': args.headless,
        'status': 'STARTING'}
    # Persist intent BEFORE launching anything, so early failures retain scope.
    atomic_json(out/'launcher.json', launcher)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    runtime_args = ['--manifest', manifest, '--duration', args.duration, '--port', port,
                    '--case', args.case, '--mode', 'mission', '--output', out,
                    '--checkpoint-dt', args.checkpoint_dt]
    if args.full_mission:
        runtime_args += ['--stop-when-done', '--require-contact-forces']
    if args.grain_rolling_friction is not None:
        runtime_args += ['--grain-rolling-friction', repr(float(args.grain_rolling_friction))]
    if args.video:
        runtime_args += ['--video']  # intent only, not live camera capture
    if args.headless:
        runtime_args += ['--headless']
    code, interrupted = 2, False
    native_issues, video_issues = [], []
    bridge = native = relay_thread = None
    ready = queue.Queue()
    blog = (out/'bridge.log').open('w', encoding='utf-8')
    ilog = (out/'isaac.log').open('w', encoding='utf-8')
    try:
        bridge = subprocess.Popen([sys.executable, '-u', str(ROOT/'bridge_server.py'), '--port', str(port)],
                                  cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', bufsize=1)
        relay_thread = threading.Thread(target=relay, args=(bridge.stdout, blog, ready), daemon=True)
        relay_thread.start()
        ready.get(timeout=60)
        native = subprocess.Popen(python_command(exe, ROOT/'run_isaac.py', runtime_args),
                                  cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', bufsize=1)
        for line in iter(native.stdout.readline, ''):
            ilog.write(line)
            ilog.flush()
            print(line, end='', flush=True)
        code = native.wait()
    except KeyboardInterrupt:
        interrupted = True
        native_issues.append('Run interrupted by the user; preserving committed measurements')
        code = 130
    except Exception as exc:
        native_issues.append('Launcher stopped: %s: %s' % (type(exc).__name__, exc))
    finally:
        for process in (native, bridge):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if native is not None and not interrupted:
            code = native.returncode
        if relay_thread is not None:
            relay_thread.join(timeout=10)
        blog.close()
        ilog.close()
    # Native process is now gone, so assembling/rendering cannot damage it.
    launcher.update(native_process_exit_code=code, status='POSTPROCESSING')
    atomic_json(out/'launcher.json', launcher)
    print('Assembling committed native measurement chunks...', flush=True)
    assembly = assemble_measurements(out)
    native_issues.extend('Measurement assembly: '+e for e in assembly.get('errors', []))
    recover_status(out, launcher, code, assembly)
    if args.video and not interrupted:
        try:
            from replay_video import render_video
            print('Rendering post-run video with the %s renderer (physics already saved)...' % args.video_renderer, flush=True)
            capture, encoding = render_video(out, exe, manifest.with_name('scene.usda'),
                stride=args.capture_stride, batch_frames=args.video_batch_frames,
                renderer=args.video_renderer)
            if not capture['png_complete'] or not encoding.get('mp4_created'):
                video_issues.append('Requested post-run video did not complete; native measurements are preserved')
        except Exception as exc:
            video_issues.append('Post-run video failed: %s: %s' % (type(exc).__name__, exc))
            atomic_json(out/'video_encoding.json', {'mp4_created': False, 'mp4_error': video_issues[-1]})
    elif args.video:
        video_issues.append('Post-run video skipped after user interruption')
    native_issues.extend(assess_native_result(out, False, args.duration, args.dt, code, args.full_mission, False))
    detailed = {}
    try:
        from assess_results import assess
        detailed = assess(out)
        atomic_json(out/'comparison.json', detailed)
        if args.full_mission and not detailed['conclusions']['full_digging_task_passes']:
            native_issues.append('Independent full-mission physical task assessment did not pass')
        if not detailed['conclusions']['controlled_native_run_evidence_passes']:
            native_issues.append('Independent native measurement audit did not pass')
        if args.video and not detailed['conclusions']['video_evidence_available']:
            video_issues.append('Requested video evidence could not be fully verified')
    except Exception as exc:
        native_issues.append('Independent assessment failed: %s: %s' % (type(exc).__name__, exc))
    native_issues = list(dict.fromkeys(native_issues))
    video_issues = list(dict.fromkeys(video_issues))
    issues = native_issues + video_issues
    effective = code if code else (1 if issues else 0)
    launcher.update(effective_exit_code=effective, status='FINISHED',
                    native_run_complete=not native_issues,
                    all_requested_outputs_complete=not issues)
    atomic_json(out/'launcher.json', launcher)
    # Task physics and optional presentation are deliberately separate gates.
    assessment = {'scope': 'native physical task and separately rendered measured-state replay',
        'native_task_validation_passed': bool(args.full_mission and not native_issues),
        'smoke_run_complete': bool(not args.full_mission and not native_issues),
        'requested_video_complete': bool(args.video and not video_issues),
        'all_requested_outputs_complete': not issues,
        'full_force_energy_validation': 'NOT_RUN', 'cross_backend_dynamics_equivalence': 'NOT_ESTABLISHED',
        'native_issues': native_issues, 'video_issues': video_issues, 'issues': issues}
    atomic_json(out/'validation.json', assessment)
    if issues:
        atomic_json(out/'launcher_error.json', {'errors': issues, 'native_validation_passed': not native_issues})
        for issue in issues:
            print('NOT VERIFIED: '+issue, file=sys.stderr)
    try:
        archive = make_results_archive(out, compact=args.full_mission or args.video)
        print('Results archive: '+str(archive), flush=True)
        if args.full_results_zip:
            full = out.with_name(out.name+'_full.zip')
            # Each archive records its own contents; the compact archive above
            # is left unchanged.
            make_results_archive(out, compact=False, archive_path=full)
            print('Complete consolidated results ZIP: '+str(full), flush=True)
    except Exception as exc:
        print('ZIP creation failed; all saved artifacts remain at '+str(out)+': '+str(exc), file=sys.stderr)
        atomic_json(out/'packaging_error.json', {'error': str(exc)})
        effective = effective or 1
        launcher.update(effective_exit_code=effective, all_requested_outputs_complete=False,
                        packaging_error=str(exc))
        assessment.update(all_requested_outputs_complete=False, packaging_error=str(exc))
        atomic_json(out/'launcher.json', launcher)
        atomic_json(out/'validation.json', assessment)
    print('Native measurements: '+str(out), flush=True)
    if (out/'isaac_native.mp4').is_file():
        print('Measured-state replay video: '+str(out/'isaac_native.mp4'), flush=True)
    return effective


if __name__ == '__main__':
    raise SystemExit(main())
