#!/usr/bin/env python3
"""Controller-side, restartable video rendering after native physics has exited.

Only rendering can resume. Frames come from measured Isaac body states, never
from the MuJoCo reference. Native arrays are never modified by this module.

Renderers:
  cpu   (default) NumPy z-buffer renderer in this controller process; no Isaac,
        no GPU. Robust fallback after the Isaac/Kit replay crashes.
  isaac Separate Isaac Sim RTX/Replicator worker processes (original method).
Frames from different renderers are never mixed in one movie.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from measurement_store import atomic_json, atomic_npz
from process_utils import python_command
from render_replay_worker import pose_digest, write_rgb_png
from video_capture import _validate_png
from encode_video import encode_png_sequence

ROOT = Path(__file__).resolve().parent
RENDERERS = {'cpu': 'cpu_zbuffer_v1', 'isaac': 'isaac_rtx_replicator'}
LEGACY_RENDERER = 'isaac_rtx_replicator'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def select_frames(times, dt, stride):
    """Select existing samples only, including first fetched and final states."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or
        not np.all(np.diff(times) > 0) or not math.isfinite(dt) or dt <= 0 or stride < 1):
        raise ValueError('Invalid native sampling times, timestep, or stride')
    steps = np.rint(times/dt).astype(np.int64)
    if not np.allclose(steps*dt, times, atol=1e-8, rtol=0):
        raise ValueError('Native timestamps are not on the measured physics grid')
    valid = np.flatnonzero(steps > 0)
    if len(valid) < 2:
        raise ValueError('At least two post-step native states are needed for video')
    take = np.unique(np.r_[valid[0], valid[steps[valid] % stride == 0], valid[-1]])
    return take, steps[take]


def prepare_plan(result, scene, *, stride=100, fps=20, resolution=(640, 360)):
    result, scene = Path(result), Path(scene)
    status = json.loads((result/'status.json').read_text(encoding='utf-8'))
    manifest = json.loads((result/'manifest.json').read_text(encoding='utf-8'))
    dt = float(status.get('dt') or manifest['dt'])
    if (not isinstance(fps, int) or not 1 <= fps <= 120 or
        len(resolution) != 2 or any(type(d) is not int or d < 2 or d % 2 for d in resolution)):
        raise ValueError('Use integer fps in 1..120 and positive, even image dimensions')
    measured_path = result/'states.npz'
    state_hash, scene_hash = file_hash(measured_path), file_hash(scene)
    with np.load(measured_path, allow_pickle=False) as data:
        times = data['time']
        take, steps = select_frames(times, dt, stride)
        p, q, names = data['positions'][take], data['quaternions_wxyz'][take], data['body_names']
    n = len(manifest['bodies'])
    if (p.shape != (len(take), n, 3) or q.shape != (len(take), n, 4) or
        names.tolist() != [b['name'] for b in manifest['bodies']] or
        not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))):
        raise ValueError('Measured pose arrays do not match the native body inventory')
    frames = [{'frame_index': i, 'state_sample_index': int(sample),
               'native_step': int(step), 'sim_time_s': float(times[sample]),
               'pose_sha256': pose_digest(p[i], q[i])}
              for i, (sample, step) in enumerate(zip(take, steps))]
    plan = {'mode': 'post_run_measured_state_replay',
            'scope': 'rendering of saved native PhysX states, not an additional physics run or live recording',
            'source_states_sha256': state_hash, 'scene_sha256': scene_hash,
            'scene_path': str(scene.resolve()), 'capture_stride_native_steps': stride,
            'physics_dt': dt, 'fps': fps, 'resolution': list(resolution),
            'camera_position_m': [7., -20., 12.], 'camera_look_at_m': [4., 1., .8],
            'frame_count': len(frames), 'frames': frames}
    plan_path = result/'video_plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding='utf-8'))
        # Absolute scene path can change when moving a result directory.
        ignored = {'scene_path'}
        if {k:v for k,v in previous.items() if k not in ignored} != {k:v for k,v in plan.items() if k not in ignored}:
            raise ValueError('Existing video plan differs; preserve that output and use a fresh result copy')
    atomic_json(plan_path, plan)
    atomic_npz(result/'replay_states.npz', positions=p, quaternions_wxyz=q,
               time=times[take], native_steps=steps, body_names=names)
    return plan


def verified_frame(result, plan, index, renderer=None):
    try:
        expected = plan['frames'][index]
        record = json.loads((result/'frame_records'/('frame_%06d.json' % index)).read_text(encoding='utf-8'))
        png = result/'frames'/('frame_%06d.png' % index)
        _validate_png(png, tuple(plan['resolution']))
        if (any(record.get(k) != v for k, v in expected.items()) or
            record.get('source_states_sha256') != plan['source_states_sha256'] or
            record.get('png_sha256') != file_hash(png) or
            record.get('mode') != plan['mode'] or
            (renderer is not None and record.get('renderer', LEGACY_RENDERER) != renderer)):
            return None
        return record
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def _render_cpu(result, plan, verified):
    """Render every unverified frame in this process from measured poses only."""
    from software_replay import Camera, ReplayGeometry, render_pose, resolve_source_scene
    manifest, source_scene = resolve_source_scene(result)
    geometry = ReplayGeometry(manifest, source_scene)
    width, height = plan['resolution']
    camera = Camera(plan['camera_position_m'], plan['camera_look_at_m'], width, height)
    (result/'frames').mkdir(exist_ok=True)
    (result/'frame_records').mkdir(exist_ok=True)
    started = time.monotonic()
    rendered = 0
    info = {'renderer': RENDERERS['cpu'], 'geometry_source': str(source_scene),
            'geometry_source_sha256': manifest['source_scene_sha256'],
            'visible_geoms': geometry.geom_count, 'triangles': geometry.triangle_count,
            'camera_horizontal_fov_deg': 47.2,
            'scope': 'flat-shaded CPU rendering of measured native body poses; not RTX'}
    with np.load(result/'replay_states.npz', allow_pickle=False) as data:
        positions, quaternions = data['positions'], data['quaternions_wxyz']
        if data['body_names'].tolist() != [b['name'] for b in manifest['bodies']]:
            raise ValueError('Replay body order does not match the native manifest')
        for index, expected in enumerate(plan['frames']):
            if verified[index] is not None:
                continue
            p, q = positions[index], quaternions[index]
            if pose_digest(p, q) != expected['pose_sha256']:
                raise ValueError('Replay pose fingerprint mismatch at frame %d' % index)
            image = render_pose(geometry, camera, p, q,
                                label='t = %.2f s  step %d' % (expected['sim_time_s'], expected['native_step']))
            png_hash = write_rgb_png(result/'frames'/('frame_%06d.png' % index), image)
            record = dict(expected, png_sha256=png_hash, source_states_sha256=plan['source_states_sha256'],
                          mode=plan['mode'], capture_timeline_delta_s=0.0, renderer=RENDERERS['cpu'])
            atomic_json(result/'frame_records'/('frame_%06d.json' % index), record)
            verified[index] = verified_frame(result, plan, index, RENDERERS['cpu'])
            rendered += 1
            if rendered % 25 == 0:
                print('POST-RUN VIDEO (cpu): %d frames rendered, %.1f s' % (rendered, time.monotonic()-started), flush=True)
    info.update(frames_rendered_this_call=rendered, wall_seconds=time.monotonic()-started)
    return info


def _refuse_to_replace_finished_video(result, plan, renderer_id):
    """Never overwrite a complete movie or frame set made by the other renderer.

    Incomplete attempts may be re-rendered; a verified MP4 or a complete verified
    frame set may not.
    """
    others = [r for r in RENDERERS.values() if r != renderer_id]
    try:
        encoding = json.loads((result/'video_encoding.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        encoding = {}
    movie_renderer = encoding.get('renderer', LEGACY_RENDERER)
    if ((result/'isaac_native.mp4').is_file() and encoding.get('mp4_created') is True and
            movie_renderer in others):
        raise ValueError('This result already has a verified MP4 from renderer %s. Move isaac_native.mp4, '
                         'video_encoding.json, frames/ and frame_records/ aside, or use a copy of the '
                         'result directory, before rendering with %s.' % (movie_renderer, renderer_id))
    for other in others:
        if all(verified_frame(result, plan, i, other) is not None for i in range(len(plan['frames']))):
            raise ValueError('This result already has a complete verified frame set from renderer %s. '
                             'Move frames/ and frame_records/ aside, or use a copy of the result '
                             'directory, before rendering with %s.' % (other, renderer_id))


def render_video(result, isaac_python, scene, *, stride=100, fps=20,
                 resolution=(640, 360), batch_frames=32, worker_timeout=600, renderer='cpu'):
    result = Path(result).resolve()
    if batch_frames < 1 or worker_timeout <= 0:
        raise ValueError('Batch size and worker timeout must be positive')
    if renderer not in RENDERERS:
        raise ValueError('Unknown renderer %r; use one of %s' % (renderer, sorted(RENDERERS)))
    renderer_id = RENDERERS[renderer]
    plan = prepare_plan(result, scene, stride=stride, fps=fps, resolution=resolution)
    _refuse_to_replace_finished_video(result, plan, renderer_id)
    (result/'render_batches').mkdir(exist_ok=True)
    attempts = []
    failures_without_progress = 0
    batch_id = len(list((result/'render_batches').glob('batch_*.json')))
    verified = [verified_frame(result, plan, i, renderer_id) for i in range(len(plan['frames']))]
    renderer_info = {'renderer': renderer_id}
    if renderer == 'cpu' and any(record is None for record in verified):
        try:
            renderer_info = _render_cpu(result, plan, verified)
        except Exception as exc:  # reported, never hidden; native data untouched
            renderer_info = {'renderer': renderer_id, 'error': '%s: %s' % (type(exc).__name__, exc)}
        attempts.append(dict(renderer_info, verified_frames=sum(r is not None for r in verified)))
        atomic_json(result/'video_render_progress.json', {'attempts': attempts,
            'verified_frames': sum(record is not None for record in verified),
            'planned_frames': len(verified), 'mode': plan['mode'], 'renderer': renderer_id})
    while renderer == 'isaac' and any(record is None for record in verified):
        start = next(i for i, record in enumerate(verified) if record is None)
        stop = min(start+batch_frames, len(verified))
        # Never overwrite a later verified frame when repairing a gap.
        for i in range(start+1, stop):
            if verified[i] is not None:
                stop = i
                break
        stem = 'batch_%04d' % batch_id
        batch_id += 1
        cmd = python_command(isaac_python, ROOT/'render_replay_worker.py', [
            '--result', result, '--scene', Path(scene).resolve(), '--start', start,
            '--stop', stop, '--batch-report', result/'render_batches'/(stem+'.json')])
        print('POST-RUN VIDEO: frames %d..%d of %d (physics results already saved)' %
              (start, stop-1, len(verified)), flush=True)
        attempt = {'first_frame': start, 'stop_frame': stop, 'log': 'render_batches/'+stem+'.log'}
        with (result/'render_batches'/(stem+'.log')).open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=worker_timeout)
                attempt['exit_code'] = process.returncode
            except subprocess.TimeoutExpired:
                attempt.update(exit_code=None, error='Render worker timeout; saved physics is unchanged')
            except OSError as exc:
                attempt.update(exit_code=None, error=str(exc))
        before = sum(record is not None for record in verified)
        for i in range(start, stop):
            verified[i] = verified_frame(result, plan, i, renderer_id)
        gained = sum(record is not None for record in verified) - before
        attempt['verified_frames_gained'] = gained
        attempts.append(attempt)
        atomic_json(result/'video_render_progress.json', {'attempts': attempts,
            'verified_frames': sum(record is not None for record in verified),
            'planned_frames': len(verified), 'mode': plan['mode'], 'renderer': renderer_id})
        if gained == 0:
            failures_without_progress += 1
            # Bounded retries do not alter resolution, scene, or physics inputs.
            if failures_without_progress >= 2:
                break
        else:
            failures_without_progress = 0
    # Incomplete recordings have a verified prefix, never an invented last frame.
    prefix = []
    for record in verified:
        if record is None:
            break
        prefix.append(record)
    complete = len(prefix) == len(plan['frames'])
    metadata = {key: value for key, value in plan.items() if key != 'frames'}
    metadata.update(frame_count=len(prefix), planned_frame_count=len(plan['frames']),
                    frames=prefix, capture_timeline_delta_s=0.0, png_complete=complete,
                    renderer=renderer_id, renderer_details=renderer_info)
    atomic_json(result/'video_frames.json', metadata)
    capture = {'mode': plan['mode'], 'png_complete': complete, 'captured_frames': len(prefix),
               'planned_frames': len(plan['frames']), 'resolution': list(resolution), 'fps': fps,
               'source_states_sha256': plan['source_states_sha256'],
               'frame_manifest': str(result/'video_frames.json'), 'capture_timeline_delta_s': 0.0,
               'render_attempts': attempts, 'native_physics_modified_by_video': False,
               'renderer': renderer_id}
    if not complete:
        capture['error'] = ('Post-run rendering incomplete; no native data was erased. See '
                            + ('video_render_progress.json' if renderer == 'cpu' else 'render_batches logs') + '.')
        encoding = {'mp4_created': False, 'mp4_error': capture['error'], 'renderer': renderer_id}
    elif (result/'isaac_native.mp4').exists() and (result/'video_encoding.json').exists():
        encoding = json.loads((result/'video_encoding.json').read_text(encoding='utf-8'))
        if encoding.get('mp4_created') is not True:
            encoding = {'mp4_created': False, 'mp4_error': 'Existing MP4 is not marked verified; preserve it before retrying'}
        elif encoding.get('renderer', LEGACY_RENDERER) != renderer_id:
            encoding = {'mp4_created': False, 'renderer': renderer_id,
                        'mp4_error': 'Existing MP4 came from another renderer; move isaac_native.mp4 aside before re-encoding'}
    else:
        encoding = encode_png_sequence(result/'frames', result/'isaac_native.mp4',
                                       fps=fps, expected_count=len(prefix))
        encoding['renderer'] = renderer_id
    atomic_json(result/'video_encoding.json', encoding)
    status = json.loads((result/'status.json').read_text(encoding='utf-8'))
    status.update(video_requested=True, video_capture=capture)
    atomic_json(result/'status.json', status)
    return capture, encoding


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--result', required=True, type=Path)
    ap.add_argument('--renderer', choices=sorted(RENDERERS), default=None,
                    help='cpu (default, no Isaac) or isaac (RTX worker processes; the default when '
                         '--isaac-python is given, as in the earlier command)')
    ap.add_argument('--isaac-python', type=Path, help='Isaac python.exe; required only with --renderer isaac')
    ap.add_argument('--scene', type=Path, default=ROOT/'generated/soil_final/scene.usda')
    ap.add_argument('--batch-frames', type=int, default=32)
    ap.add_argument('--capture-stride', type=int, default=100)
    args = ap.parse_args()
    if args.renderer is None:
        args.renderer = 'isaac' if args.isaac_python is not None else 'cpu'
    if args.renderer == 'isaac' and args.isaac_python is None:
        ap.error('--isaac-python is required with --renderer isaac')
    print('Renderer: %s' % args.renderer, flush=True)
    cap, enc = render_video(args.result, args.isaac_python, args.scene, renderer=args.renderer,
                            batch_frames=args.batch_frames, stride=args.capture_stride)
    print(json.dumps({'capture': cap, 'encoding': enc}, indent=2))
    print('Video metadata was updated. Existing comparison/validation/ZIP files are not rewritten by this command.')
    raise SystemExit(0 if cap['png_complete'] and enc.get('mp4_created') else 1)
