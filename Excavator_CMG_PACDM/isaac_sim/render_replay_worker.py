#!/usr/bin/env python3
"""Render a bounded batch of saved PhysX states, never run robot dynamics.

Run only in the Isaac environment. All edits are in the USD session layer of
this separate process. Original scene and native measurements are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import traceback
import zlib

import numpy as np
from measurement_store import atomic_json, _replace


def write_rgb_png(path, rgb):
    """Synchronous, bounded-memory RGB PNG output without an async writer queue."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8:
        raise ValueError('RGB annotator must supply an H x W x (3 or 4) uint8 image')
    height, width = rgb.shape[:2]
    pixels = np.ascontiguousarray(rgb[:, :, :3])
    filtered = b''.join(b'\x00' + row.tobytes() for row in pixels)
    def chunk(kind, payload):
        return (struct.pack('>I', len(payload)) + kind + payload +
                struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff))
    data = (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(filtered, 3)) + chunk(b'IEND', b''))
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def pose_digest(positions, quaternions):
    h = hashlib.sha256()
    for a in (positions, quaternions):
        h.update(np.ascontiguousarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result', required=True, type=Path)
    ap.add_argument('--scene', required=True, type=Path)
    ap.add_argument('--start', required=True, type=int)
    ap.add_argument('--stop', required=True, type=int)
    ap.add_argument('--batch-report', required=True, type=Path)
    args = ap.parse_args()
    result = args.result.resolve()
    plan = json.loads((result/'video_plan.json').read_text(encoding='utf-8'))
    manifest = json.loads((result/'manifest.json').read_text(encoding='utf-8'))
    width, height = plan['resolution']
    if not 0 <= args.start < args.stop <= len(plan['frames']):
        ap.error('Invalid replay batch range')
    with np.load(result/'replay_states.npz', allow_pickle=False) as data:
        positions = data['positions'][args.start:args.stop]
        quaternions = data['quaternions_wxyz'][args.start:args.stop]
        if data['body_names'].tolist() != [b['name'] for b in manifest['bodies']]:
            raise ValueError('Replay body order does not match the native manifest')
    if hashlib.sha256(args.scene.read_bytes()).hexdigest() != plan['scene_sha256']:
        raise ValueError('Replay scene differs from the requested scene')
    report = {'status': 'STARTED', 'start_frame': args.start, 'stop_frame': args.stop,
              'mode': 'post_run_measured_state_replay', 'physics_steps': 0,
              'source_states_sha256': plan['source_states_sha256'], 'frames_written': 0}
    atomic_json(args.batch_report, report)
    # Omniverse/pxr imports must happen only after SimulationApp construction.
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'disable_viewport_updates': True,
        'renderer': 'RaytracedLighting', 'width': width, 'height': height,
        'anti_aliasing': 0, 'denoiser': False, 'multi_gpu': False, 'max_gpu_count': 1,
        'extra_args': ['--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=0.15']})
    rp = annotator = None
    rc = 0
    try:
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import UsdGeom, UsdPhysics, UsdLux, Gf
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene)):
            raise RuntimeError('Unable to open the replay scene')
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        hidden = 0
        # Disable dynamics BEFORE the first rendering/app update. No native
        # simulation interface or controller bridge is created in this process.
        # These schema calls resolve composed USD state; do not use an Sdf.ChangeBlock.
        for prim in list(stage.Traverse()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(False)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            # Exported opacity-zero collision meshes need no ray tracing.
            # Only display geometry in this replay copy is hidden.
            if prim.IsA(UsdGeom.Gprim):
                opacity = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()
                if opacity is not None and len(opacity) and all(float(v) <= 0 for v in opacity):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden += 1
        transforms = []
        for body in manifest['bodies']:
            prim = stage.GetPrimAtPath(body['path'])
            if not prim:
                raise ValueError('Missing replay body: ' + body['path'])
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            translation = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, 'replay')
            orientation = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, 'replay')
            # The stored transforms are world-frame, including all soil grains.
            xf.SetResetXformStack(True)
            transforms.append((translation, orientation))
        camera = UsdGeom.Camera.Define(stage, '/World/NativeReplayCamera')
        camera.CreateFocalLengthAttr().Set(24.)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(.1, 200.))
        cam_matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*plan['camera_position_m']),
                                          Gf.Vec3d(*plan['camera_look_at_m']),
                                          Gf.Vec3d(0, 0, 1)).GetInverse()
        UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(cam_matrix)
        light = UsdLux.DomeLight.Define(stage, '/World/NativeReplayDome')
        light.CreateIntensityAttr().Set(500.)
        app.reset_render_settings()
        rep.orchestrator.set_capture_on_play(False)
        rp = rep.create.render_product(str(camera.GetPath()), (width, height), name='NativeReplayRGB')
        rp.hydra_texture.set_updates_enabled(False)
        annotator = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        annotator.attach(rp)
        (result/'frames').mkdir(exist_ok=True)
        (result/'frame_records').mkdir(exist_ok=True)
        report['hidden_opacity_zero_shapes'] = hidden
        for frame in range(args.start, args.stop):
            local = frame - args.start
            p, q = positions[local], quaternions[local]
            expected = plan['frames'][frame]
            if pose_digest(p, q) != expected['pose_sha256']:
                raise ValueError('Replay pose fingerprint mismatch')
            for i, (translation, orientation) in enumerate(transforms):
                translation.Set(Gf.Vec3d(*map(float, p[i])))
                norm = float(np.linalg.norm(q[i]))
                if not math.isfinite(norm) or norm < 1e-12:
                    raise ValueError('Invalid measured quaternion')
                w, x, y, z = map(float, q[i]/norm)
                orientation.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
            before = float(timeline.get_current_time())
            context.reset_renderer_accumulation()
            rp.hydra_texture.set_updates_enabled(True)
            try:
                # Two zero-time renders of the SAME measured pose allow one
                # refresh without carrying prior-pose temporal reconstruction.
                for _ in range(2):
                    rep.orchestrator.step(delta_time=0.0, wait_for_render=True, pause_timeline=True)
                rgb = np.asarray(annotator.get_data())
                if rgb.shape not in ((height, width, 3), (height, width, 4)):
                    raise RuntimeError('Unexpected RGB image shape: ' + str(rgb.shape))
                if timeline.is_playing() or not math.isclose(float(timeline.get_current_time()), before, abs_tol=1e-9):
                    raise RuntimeError('Replay rendering unexpectedly advanced the timeline')
                png_hash = write_rgb_png(result/'frames'/('frame_%06d.png' % frame), rgb)
            finally:
                rp.hydra_texture.set_updates_enabled(False)
            frame_record = dict(expected, png_sha256=png_hash,
                source_states_sha256=plan['source_states_sha256'],
                mode='post_run_measured_state_replay', capture_timeline_delta_s=0.0,
                renderer='isaac_rtx_replicator')
            atomic_json(result/'frame_records'/('frame_%06d.json' % frame), frame_record)
            report['frames_written'] += 1
            atomic_json(args.batch_report, report)
            print('REPLAY frame %d, native time %.3f s' % (frame, expected['sim_time_s']), flush=True)
        report['status'] = 'COMPLETED'
    except BaseException as exc:
        report.update(status='FAILED', error='%s: %s' % (type(exc).__name__, exc),
                      traceback=traceback.format_exc())
        print(report['error'], file=sys.stderr, flush=True)
        rc = 1
    finally:
        # Metadata must survive Kit closing Python or a teardown crash.
        atomic_json(args.batch_report, report)
        try:
            if annotator is not None and rp is not None:
                annotator.detach(rp)
            if rp is not None:
                rp.destroy()
        finally:
            app.close()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
