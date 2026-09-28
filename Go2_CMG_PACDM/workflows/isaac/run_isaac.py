"""Run one genuine Isaac Sim / PhysX validation case. Use launch.py normally."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
CASES = {
    'nominal': dict(dt=.001, friction=.8, payload=0., push=1., feedforward=True, actuation=True),
    'fine': dict(dt=.0005, friction=.8, payload=0., push=1., feedforward=True, actuation=True),
    'low_friction': dict(dt=.001, friction=.55, payload=0., push=1., feedforward=True, actuation=True),
    'payload': dict(dt=.001, friction=.8, payload=1.5, push=1., feedforward=True, actuation=True),
    'strong_push': dict(dt=.001, friction=.8, payload=0., push=1.25, feedforward=True, actuation=True),
    'no_actuation': dict(dt=.001, friction=.8, payload=0., push=1., feedforward=True, actuation=False),
    'PD_ablation': dict(dt=.001, friction=.8, payload=0., push=1., feedforward=False, actuation=True),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    def convert(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        raise TypeError(type(x).__name__)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller-python', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--run-id', required=True)
    p.add_argument('--case', choices=CASES, default='nominal')
    p.add_argument('--duration', type=float, default=26.)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--ffmpeg')
    p.add_argument('--capture-mode', choices=('render', 'replicator'), default='render')
    p.add_argument('--shutdown-mode', choices=('auto', 'fast', 'graceful'), default='auto')
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    if not 0 < a.duration <= 26 or abs(a.duration/.004-round(a.duration/.004)) > 1e-8:
        p.error('duration must be a positive multiple of .004 s, at most 26 s')
    if not a.no_video and not a.ffmpeg:
        p.error('--ffmpeg is required for video; launch.py locates it automatically')
    if a.preflight and a.case != 'nominal':
        p.error('Run preflight on nominal parameters only')
    a.output = a.output.resolve()
    return a


def execute(args, app, result):
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaac_validation.client import WorkerClient
    from isaac_validation.scene import build_scene
    from isaac_validation.physics import PhysicsRobot, ContactMonitor
    from isaac_validation.preflight import audit
    from isaac_validation.video import camera, VideoWriter
    from isaac_validation.capture import PhysicsFrameCapture, configure_capture_camera
    from isaac_validation.lifecycle import cleanup_actions
    from isaac_validation.runtime import select_physx_engine, current_stage_id

    cfg = CASES[args.case]
    dt = cfg['dt']
    out = args.output / args.case
    cmg = json.loads((ROOT/'data/go2_cmg.json').read_text(encoding='utf-8'))
    # Immutable artifacts contain the exact reference sample at each log time.
    with np.load(ROOT/'data/reference.npz', allow_pickle=False) as z:
        reference = {k: z[k].copy() for k in z.files}
    worker = WorkerClient(args.controller_python, ROOT, out/'controller.log')
    writer = monitor = capture = world = robot = None
    log = {k: [] for k in ['time', 'q', 'v', 'q_ref', 'torque', 'passive_torque',
           'feet', 'foot_ref', 'cmg_feet', 'stance', 'predicted_force', 'qp_violation',
           'push', 'measured_support_force']}
    try:
        info = worker.init(duration=args.duration, feedforward=cfg['feedforward'])
        result['controller_sha256'] = info['controller_sha256']
        result['physics_runtime'] = select_physx_engine()
        world = World(physics_dt=dt, rendering_dt=.04, stage_units_in_meters=1.,
                      backend='numpy', device='cpu')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(False)
        context.set_broadphase_type('MBP')
        context.set_solver_type('TGS')
        stage = omni.usd.get_context().get_stage()
        metadata = build_scene(stage, ROOT, friction=cfg['friction'], payload=cfg['payload'])
        save(out/'scene_metadata.json', metadata)
        camera_path = camera(stage)
        if not args.no_video:
            result['camera_configuration'] = configure_capture_camera(stage, camera_path)
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        world.reset()
        stage_id = current_stage_id(expected_stage=stage)
        result['physics_runtime'].update(stage_id=stage_id, tensor_frontend='numpy',
                                         device='cpu', initialization='World.reset + explicit stage ID')
        print(f'Initializing NumPy PhysX tensors for USD stage {stage_id}', flush=True)
        robot = PhysicsRobot(metadata, cmg, info['joint_names'], stage_id=stage_id)
        result['physx_joint_order'] = robot.dof_names
        result['physx_link_order'] = robot.link_names
        if args.preflight:
            print('Running PACDM reference validation...', flush=True)
            rv = worker.validate_reference()
            rv.update(run_id=args.run_id, engine='Isaac Sim / PhysX', generated_utc=utc(),
                      source_sha256=result['source_sha256'])
            save(args.output/'reference_validation.json', rv)
            if not rv['passed']:
                raise RuntimeError('PACDM reference validation failed; see reference_validation.json')
            print('Running native PhysX mechanics checks...', flush=True)
            try:
                mv = audit(world, robot, worker, info, cmg)
            except Exception as exc:
                mv = dict(passed=False, checks=[], failure=str(exc), traceback=traceback.format_exc())
            mv.update(run_id=args.run_id, generated_utc=utc(), source_sha256=result['source_sha256'])
            save(args.output/'mechanics.json', mv)
            if not mv['passed']:
                raise RuntimeError('PhysX mechanics preflight failed; see mechanics.json')
        else:
            # Never reuse an unrelated or failed mechanics certificate.
            for filename in ('mechanics.json', 'reference_validation.json'):
                path = args.output/filename
                if not path.is_file():
                    raise RuntimeError('Run the nominal case with --preflight before other cases')
                certificate = json.loads(path.read_text(encoding='utf-8'))
                if (certificate.get('run_id') != args.run_id or certificate.get('passed') is not True
                        or certificate.get('source_sha256') != result['source_sha256']):
                    raise RuntimeError(f'Invalid current-run preflight certificate: {filename}')
        world.set_simulation_dt(physics_dt=dt, rendering_dt=.04)
        info = worker.init(duration=args.duration, feedforward=cfg['feedforward'])
        robot.set_state(np.array(info['q']), np.array(info['v']))
        # Each stage contains native rigid-body collisions. No pose setters below this line.
        monitor = ContactMonitor(metadata, dt)
        if not args.no_video:
            result['execution_phase'] = 'camera_initialization'
            capture = PhysicsFrameCapture(world, robot, camera_path, mode=args.capture_mode)
            capture.start()
            writer = VideoWriter(out/'Go2_IsaacSim.mp4', args.ffmpeg)
        control_every = round(.004/dt)
        log_every = round(.01/dt)
        frame_every = round(.04/dt)
        steps = round(args.duration/dt)
        lower, upper, limits = (np.asarray(info[x]) for x in ('lower', 'upper', 'limits'))
        damping, friction = (np.asarray(info[x])[6:] for x in ('damping', 'frictionloss'))
        motor = np.zeros(12)
        passive = np.zeros(12)
        predicted = np.zeros((4, 3))
        qp = 0.
        impulse = np.zeros((2, 3))
        result.update(max_qp_violation=0., peak_torque_limit_fraction=0.,
                      min_joint_margin_rad=100., min_base_height_m=100., max_tilt_rad=0.,
                      completed=False, failure=None, termination_reason='duration')
        initial_world_time = world.current_time
        result['execution_phase'] = 'rollout'
        for step in range(steps+1):
            if not app.is_running():
                raise RuntimeError('Isaac Sim was closed before the case completed')
            if not world.is_playing():
                raise RuntimeError('Isaac timeline stopped during the scripted case')
            t = step*dt
            q, v = robot.state()
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise RuntimeError('Non-finite PhysX state')
            result['simulated_s'] = t
            result['min_base_height_m'] = min(result['min_base_height_m'], float(q[2]))
            result['max_tilt_rad'] = max(result['max_tilt_rad'], float(np.linalg.norm(q[4:6])))
            result['min_joint_margin_rad'] = min(result['min_joint_margin_rad'], float(np.min(np.minimum(q[6:]-lower, upper-q[6:]))))
            fell = q[2] < .18 or np.linalg.norm(q[4:6]) > .65
            if step % control_every == 0 and not fell:
                command = worker.command(t, q, v)
                motor = np.asarray(command['tau']) if cfg['actuation'] else np.zeros(12)
                predicted = np.asarray(command['predicted_forces'])
                qp = float(command['qp_violation'])
                result['max_qp_violation'] = max(result['max_qp_violation'], qp)
                result['peak_torque_limit_fraction'] = max(result['peak_torque_limit_fraction'], float(np.max(abs(motor)/limits)))
            # Motor limits apply to motors; passive joint loads are separate physical forces.
            passive = -damping*v[6:]-friction*np.tanh(v[6:]/.02)
            push = np.zeros(3)
            if 1.2 <= t < 1.35:
                push = cfg['push']*np.array([0., 32., 0.])
            elif 24.3 <= t < 24.5:
                push = cfg['push']*np.array([-25., 0., 0.])
            if step % log_every == 0 or fell or step == steps:
                # Saved reference grid; interpolation only for an early terminal sample.
                nearest = int(round(t/.01))
                if abs(reference['time'][nearest]-t) < 1e-8:
                    qr = reference['q'][nearest]
                    active = reference['active'][nearest]
                    stance = reference['stance'][nearest]
                else:
                    # Negative-control terminal state may precede a reference-grid sample.
                    ix = np.searchsorted(reference['time'], t)
                    a = (t-reference['time'][ix-1])/.01
                    qr = (1-a)*reference['q'][ix-1]+a*reference['q'][ix]
                    active = (1-a)*reference['active'][ix-1]+a*reference['active'][ix]
                    stance = reference['stance'][ix-1]
                # CMG FK must use this very state, not the previous controller update.
                mechanics = worker.audit_state(q, v)
                row = [t, q, v, qr, motor.copy(), passive.copy(), robot.foot_positions(),
                       active[6:].reshape(4,3), mechanics['feet'], stance,
                       predicted.copy(), qp, push.copy(), monitor.normal.copy()]
                for key, value in zip(log, row):
                    log[key].append(value)
            if writer and step % frame_every == 0 and step < steps:
                writer.append(capture.frame(t))
            if fell:
                result.update(termination_reason='fall', failure='Physical fall threshold reached')
                break
            if step == steps:
                result['completed'] = True
                break
            robot.effort(motor + passive)
            robot.push(push)
            monitor.begin_step(t)
            world.step(render=False)
            impulse[int(t >= 20)] += push*dt
            if monitor.error:
                raise RuntimeError('Contact reporting failed: '+monitor.error)
            if step % max(1, round(1/dt)) == 0:
                print(f'{args.case}: {t:.1f}/{args.duration:.1f} simulated s, z={q[2]:.3f} m', flush=True)
        result['world_time_delta_s'] = float(world.current_time-initial_world_time)
        if abs(result['world_time_delta_s']-result['simulated_s']) > dt*1.1:
            raise RuntimeError('Simulator elapsed time differs from logged integration steps')
        result['external_push_impulse_Ns'] = impulse[0].tolist()
        result['recovery_push_impulse_Ns'] = impulse[1].tolist()
        result['final_speed_m_s'] = float(np.linalg.norm(v[:3]))
        robot.effort(np.zeros(12))
    finally:
        # Finish application-owned resources BEFORE closing the USD stage or
        # unloading extensions. Every cleanup is attempted; preserve the
        # original simulation exception when there is one.
        active_exception = sys.exc_info()[0] is not None
        actions = []
        if monitor is not None:
            result.update(unexpected_contact_instances=monitor.bad,
                          unexpected_contact_events=monitor.events,
                          contact_report_available=monitor.points > 0 and monitor.error is None,
                          contact_callback_count=monitor.callbacks, contact_point_count=monitor.points,
                          support_force_definition='Sum of positive normal contact impulse magnitudes / dt per foot; not signed world-z reaction')
            actions.append(('contact_monitor.close', monitor.close))
        if log['time']:
            actions.append(('save_trajectory', lambda: np.savez_compressed(
                out/'trajectory.npz', **{k: np.asarray(v) for k, v in log.items()})))
        if writer is not None:
            def finish_video():
                result['video'] = writer.close()
                result['video']['run_id'] = args.run_id
                result['video']['capture_mode'] = args.capture_mode
                video = result['video']
                if video.get('encoder_error') or (video.get('encoder_started') and video.get('encoder_exit_code') != 0):
                    raise RuntimeError('Video encoder failed; see Go2_IsaacSim.ffmpeg.log: ' + str(video))
            actions.append(('video_encoder.close', finish_video))
        if capture is not None:
            def save_capture_audit():
                audit_record = capture.audit()
                audit_record['run_id'] = args.run_id
                path = out/'capture_timing.json'
                save(path, audit_record)
                result['capture_audit'] = dict(
                    path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    physics_unchanged=audit_record['physics_unchanged'],
                    camera_ready=audit_record['camera_ready'], schema_version=audit_record['schema_version'],
                    frames=audit_record['frame_count'], mode=args.capture_mode)
            actions.extend([('save_capture_audit', save_capture_audit),
                            ('capture.close', capture.close)])
        if robot is not None:
            actions.append(('physics_tensor_views.close', robot.close))
        if world is not None:
            actions.extend([('world.stop', world.stop),
                            ('world.clear_instance', World.clear_instance)])
        actions.append(('controller_worker.close', worker.close))
        result['resource_cleanup'] = cleanup_actions(actions)
        if not result['resource_cleanup']['passed'] and not active_exception:
            raise RuntimeError('Application resource cleanup failed: ' +
                               '; '.join(result['resource_cleanup']['errors']))


def main():
    args = parse_args()
    out = args.output / args.case
    out.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT/'run_isaac.py', ROOT/'launch.py', ROOT/'environment-controller.yml',
                    ROOT/'vendor/pacdm_original.py', ROOT/'data/go2_cmg.json',
                    ROOT/'data/reference.npz', ROOT/'upstream/unitree_go2/go2.xml']
    source_files += sorted((ROOT/'isaac_validation').glob('*.py'))
    source_files += sorted((ROOT/'go2').glob('*.py'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result = dict(schema_version=2, code_revision=3, capture_audit_schema_version=2,
                  execution_phase='initialization', video_requested=not getattr(args, 'no_video', False),
                  name=args.case, case=args.case, run_id=args.run_id, started_utc=utc(),
                  engine='Isaac Sim / PhysX', source_sha256=hashes,
                  params=CASES[args.case], dt_s=CASES[args.case]['dt'], duration_s=args.duration,
                  simulated_s=0., completed=False, failure=None, video=None)
    app = None
    code = 1
    try:
        from isaacsim import SimulationApp
        from isaac_validation.runtime import shutdown_configuration
        result['shutdown_policy'] = shutdown_configuration(SimulationApp, getattr(args, 'shutdown_mode', 'auto'))
        print('Shutdown policy: ' + result['shutdown_policy']['selected'], flush=True)
        app = SimulationApp({'headless': args.headless, 'width':1280, 'height':720,
                             'renderer':'RayTracedLighting', 'sync_loads':True, 'fast_shutdown':result['shutdown_policy']['fast_shutdown']})
        try:
            from isaacsim.core.version import get_version
            result['isaac_version'] = list(get_version())
        except ImportError:
            result['isaac_version'] = 'version API unavailable'
        execute(args, app, result)
        code = 0
    except KeyboardInterrupt:
        code = 130
        result.update(completed=False, failure='Interrupted by user', termination_reason='interrupted',
                      traceback=traceback.format_exc())
        traceback.print_exc()
    except Exception as exc:
        result.update(completed=False, failure=str(exc), termination_reason='exception',
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['finished_utc'] = utc()
        # This is the PRE-CLOSE Python status, not an observed process exit.
        # The launcher is the only authority on whether native shutdown exited
        # successfully. A native fault cannot be caught by Python here.
        result['execution_exit_code'] = code
        result['application_shutdown'] = dict(state='pending' if app else 'not_started')
        try:
            save(out/'case.json', result)
        except Exception:
            code = 1
            result['execution_exit_code'] = code
            traceback.print_exc()
        if app:
            from isaac_validation.runtime import close_simulation_app
            try:
                close_simulation_app(app, code)
            except Exception as exc:
                code = 1 if code == 0 else code
                result['execution_exit_code'] = code
                result['application_shutdown'] = dict(state='exception', error=str(exc),
                                                       traceback=traceback.format_exc())
                traceback.print_exc()
            else:
                # Fast shutdown may terminate inside close, in which case the
                # parent still observes the final OS status in the manifest.
                result['application_shutdown'] = dict(state='returned')
            save(out/'case.json', result)
    return code


if __name__ == '__main__':
    sys.exit(main())
