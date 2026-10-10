"""One force-driven Isaac Sim / PhysX case. Imported after SimulationApp starts."""
from __future__ import annotations
import json
import platform
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .mechanics import StateMapping, disturbance, phase
from .scoring import score_case
from .control import ReferenceController
from .protocol import RuntimeGuard, validate_request


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def run_case(app, root, output, name, run_id, dt=.002, duration=22., video=False):
    validate_request(name, dt, duration)
    import carb
    import omni.usd
    from .compat import start_physics, BodyView, as_numpy
    from .usd_scene import build_scene
    from .inertial_audit import audit_native_properties
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    cmg=json.loads((root/'data/stewart.cmg.json').read_text())
    if name=='heavy_payload':
        payload=next(b for b in cmg['bodies'] if b['id']=='payload')
        ratio=14./payload['mass_kg']
        payload['mass_kg']=14.
        payload['inertia_kg_m2']=(np.array(payload['inertia_kg_m2'])*ratio).tolist()
    write_json(output/f'{name}.cmg.json',cmg)
    controller = ReferenceController(cmg, root/'baseline/reference.npz', name)
    ref = {'q': np.array([controller.initial_q])}
    guard = RuntimeGuard(cmg)
    settings=carb.settings.get_settings()
    settings.set_bool('/physics/updateToUsd',True)
    settings.set_bool('/physics/updateVelocitiesToUsd',True)
    settings.set_bool('/physics/suppressReadback',False)
    omni.usd.get_context().new_stage()
    stage=omni.usd.get_context().get_stage()
    scene=build_scene(cmg,ref['q'][0],stage=stage,dt=dt)
    if not scene['stage'].GetRootLayer().Export(str(output/f'{name}.usda')):
        raise OSError(f'Could not export {name}.usda')
    write_json(output/f'{name}.model_audit.json',scene['audit'])
    context,physics,startup_route=start_physics(app,dt)
    body_view=BodyView(physics.create_rigid_body_view('/World/Stewart/Bodies/*'))
    paths=list(body_view.prim_paths)
    inverse={p:n for n,p in scene['body_paths'].items()}
    if len(paths)!=19 or set(paths)!=set(inverse):
        raise RuntimeError(f'Expected 19 independent rigid bodies, got {paths}')
    names=[inverse[p] for p in paths]
    indices=np.arange(len(paths),dtype=np.int32)
    # Some Isaac releases warm-start physics while initializing. Establish the
    # assembled rest state once after that lifecycle step, before the mission.
    # These are the only native state assignments in the entire rollout.
    initial_transforms=[]
    for body_name in names:
        T=np.asarray(scene['audit']['body_properties'][body_name]['initial_transform'])
        initial_transforms.append(np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat()])
    body_view.set_transforms(np.asarray(initial_transforms,dtype=np.float32),indices)
    body_view.set_velocities(np.zeros((len(paths),6),dtype=np.float32),indices)
    com_frames=np.asarray(body_view.get_coms(),dtype=float)
    masses=np.asarray(body_view.get_masses(),dtype=float)
    # Native get_inertias() is a full COM-centered tensor in BODY axes.
    # Do not apply get_coms()'s principal rotation to it a second time.
    inertias=np.asarray(body_view.get_inertias(),dtype=float)
    native_audit=audit_native_properties(names,masses,com_frames,inertias,
                scene['audit']['expected_body_properties'],paths)
    native_audit.update(run_id=run_id,case=name,startup_route=startup_route,
                        tensor_frontend=body_view.frontend)
    write_json(output/f'{name}.native_inertial_audit.json',native_audit)
    if not native_audit['passed']:
        raise RuntimeError('PhysX inertial import mismatch: '+ '; '.join(native_audit['errors'])+
                           f'. Raw evidence: {name}.native_inertial_audit.json')
    coms=com_frames[:,:3]
    mass_error=native_audit['mass_max_error_kg']
    com_error=native_audit['com_max_error_m']
    inertia_error=native_audit['inertia_max_error_kg_m2']
    print(f'{name}: native inertia audit PASS (max error {inertia_error:.6g} kg m2)',flush=True)
    native_gravity=np.asarray(as_numpy(physics.get_gravity()),dtype=float)
    if native_gravity.shape!=(3,) or not np.all(np.isfinite(native_gravity)) or not np.allclose(native_gravity,cmg['gravity_m_s2'],atol=1e-6,rtol=0):
        raise RuntimeError(f'PhysX gravity mismatch: {native_gravity}')
    mapping=StateMapping(cmg,names,coms)
    initial=mapping.read(body_view.get_transforms().copy(),body_view.get_velocities().copy())
    if np.max(np.abs(initial['q']-ref['q'][0]))>2e-5 or np.max(abs(initial['velocity']))>1e-5:
        raise RuntimeError('Physics initialization changed the assembled zero-velocity initial state')
    if initial['closure_error_m']>2e-6:
        raise RuntimeError('Native initial loop closure is invalid')
    n=round(duration/dt)+1
    if abs((n-1)*dt-duration)>1e-9:
        raise ValueError('Duration must be an integer multiple of the physics timestep')
    shapes={'time':(), 'engine_time':(), 'q':(24,), 'velocity':(24,), 'target_pose':(6,),
            'force':(6,), 'requested_force':(6,), 'wrench':(6,), 'pose_error_m':(),
            'angle_error_rad':(), 'closure_error_m':(), 'joint_error_m':(),
            'joint_error_rad':(), 'actuator_error_m':(6,), 'power_W':(),
            'body_transforms':(len(names),7), 'body_velocities':(len(names),6)}
    data={k:np.empty((n,)+shape) for k,shape in shapes.items()}
    active=mapping.active
    ff=controller.feedforward
    recorder=None; video_error=None; video_meta=None
    if video:
        try:
            from .video import LiveVideo
            recorder=LiveVideo(stage,context,output/'Stewart_IsaacSim.mp4',run_id=run_id)
        except Exception as error:
            video_error=f'{type(error).__name__}: {error}'
            print('VIDEO INITIALIZATION FAILED: '+video_error,flush=True)
    start_clock=float(context.current_time)
    first_index=int(context.current_time_step_index)
    started=time.perf_counter(); recorded=0; next_frame=0; fps=30
    try:
        for k in range(n):
            if not app.is_running():
                raise RuntimeError('Isaac Sim closed before case completion')
            t=k*dt
            native_transforms=body_view.get_transforms(); native_velocities=body_view.get_velocities()
            state=mapping.read(native_transforms,native_velocities)
            actual_t=float(context.current_time)-start_clock
            if abs(actual_t-t)>max(2e-7,dt*1e-4) or int(context.current_time_step_index)-first_index!=k:
                raise RuntimeError(f'Physics clock mismatch at sample {k}: {actual_t} versus {t}')
            target, target_v, requested, force, wrench = controller.evaluate(t, state['q'], state['velocity'])
            pos_error=float(np.linalg.norm(state['q'][:3]-target[:3]))
            rotation_error=float((Rotation.from_euler('ZYX',target[3:6]).inv()*Rotation.from_euler('ZYX',state['q'][3:6])).magnitude())
            values=dict(time=t,engine_time=actual_t,q=state['q'],velocity=state['velocity'],target_pose=target[:6],
                        force=force,requested_force=requested,wrench=wrench,pose_error_m=pos_error,
                        angle_error_rad=rotation_error,closure_error_m=state['closure_error_m'],
                        body_transforms=native_transforms,body_velocities=native_velocities,
                        joint_error_m=state['joint_error_m'],joint_error_rad=state['joint_error_rad'],
                        actuator_error_m=state['q'][active]-target[active],power_W=force@state['velocity'][active])
            for key in data:data[key][k]=values[key]
            recorded=k+1
            if recorder is not None and t+dt/2>=next_frame/fps:
                try:
                    # Capture occurs before the next force application; rendering never advances physics.
                    recorder.capture(t,pos_error,float(np.rad2deg(rotation_error)),state['closure_error_m'],phase(t))
                    next_frame+=1
                except Exception as error:
                    video_error=f'{type(error).__name__}: {error}'
                    print('VIDEO CAPTURE FAILED: '+video_error,flush=True)
                    try:recorder.finish()
                    except Exception:pass
                    recorder=None
            if k % max(1,round(2/dt)) == 0:
                print(f'{name}: t={t:5.2f}s position={pos_error*1000:.3f}mm closure={state["closure_error_m"]*1000:.4f}mm',flush=True)
            guard.check(name, state, pos_error, t)
            if k<n-1:
                forces,torques=mapping.forces(state,force,wrench)
                body_view.apply_forces_and_torques_at_position(np.asarray(forces,dtype=np.float32),
                    np.asarray(torques,dtype=np.float32),np.asarray(state['com'],dtype=np.float32),indices,True)
                context.step(render=False)
        metrics=score_case(data,cmg,name,dt,duration,ff)
        metrics.update(run_id=run_id,engine='Isaac Sim / PhysX',completed=True,
                       elapsed_wall_s=time.perf_counter()-started,startup_route=startup_route,tensor_frontend=body_view.frontend,
                       native_mass_max_error_kg=mass_error,native_com_max_error_m=com_error,
                       native_inertia_max_error_kg_m2=inertia_error,
                       native_inertial_audit_passed=True,
                       native_inertia_frame=native_audit['inertia_frame'],
                       native_gravity_m_s2=native_gravity.tolist(),
                       physics_state_assignment_after_initialization=False,
                       native_body_count=len(paths),reference_source='baseline/reference.npz')
        try:
            import omni.kit.app
            metrics['kit_version']=omni.kit.app.get_app().get_build_version()
        except Exception:metrics['kit_version']='unavailable'
        metrics['python_version']=platform.python_version()
        write_json(output/f'{name}.json',metrics)
    finally:
        # Even failed cases retain only recorded data, never uninitialized arrays.
        np.savez_compressed(output/f'{name}.npz',**{k:v[:recorded] for k,v in data.items()},
                            coordinate_ids=np.array(cmg['coordinate_ids']),run_id=np.array(run_id),
                            body_names=np.array(names),body_coms=coms)
        if recorder is not None:
            try:video_meta=recorder.finish()
            except Exception as error:video_error=f'{type(error).__name__}: {error}'
        if video:
            if video_error is not None:
                video_meta=dict(passed=False,error=video_error,run_id=run_id,source='live PhysX physics')
            if video_meta is None:video_meta=dict(passed=False,error='No recording completed',run_id=run_id)
            write_json(output/'video_metadata.json',video_meta)
        context.stop()
    return metrics
