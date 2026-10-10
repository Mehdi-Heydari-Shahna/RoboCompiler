"""Native Isaac Sim / CPU PhysX execution. Not a MuJoCo or kinematic fallback.

SimulationApp is imported inside run(), before importing any Omniverse module.
Every API mismatch aborts with a saved error; no missing API is treated as PASS.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json,platform,time,traceback
import numpy as np
from scipy.spatial.transform import Rotation
from .model import Model,FOOT_NAMES,finite,poses_from_xyzw,inertias_match
from .control import Reference,Controller,Config,external_push
from .io_utils import write_json,name_map
from .observations import Recorder
from .contact_gate import ContactGate, ControllerTrace
from .gates import assess
from .runtime_bridge import managed_simulation_view, TensorIO, require_methods, fill_disturbance_buffers
from .scene_paths import BODIES, SCENE
from .version import VERSION, REPAIR


def initialize_tensors(world,model,P0,ref,manager=None,binding_audit=None):
    binding_audit={} if binding_audit is None else binding_audit
    view=managed_simulation_view(world,manager,binding_audit)
    art=view.create_articulation_view(BODIES+'/'+model.c['root_body'])
    if art.count!=1:raise RuntimeError(f'Expected one articulation, got {art.count}')
    require_methods(art, ('get_dof_positions', 'get_dof_velocities', 'get_link_transforms',
        'get_link_velocities', 'set_dof_positions', 'set_dof_velocities', 'set_root_transforms',
        'set_root_velocities', 'set_dof_stiffnesses', 'set_dof_dampings', 'set_dof_armatures',
        'set_dof_position_targets', 'set_dof_velocity_targets', 'set_dof_actuation_forces',
        'set_dof_max_forces', 'get_dof_friction_properties', 'set_dof_friction_properties',
        'get_masses', 'get_coms', 'get_inertias', 'get_dof_dampings', 'get_dof_armatures',
        'get_dof_limits', 'get_dof_stiffnesses', 'apply_forces_and_torques_at_position'),
        'PhysX articulation view')
    io=TensorIO(art.get_dof_positions())
    binding_audit['tensor_frontend']=io.frontend
    binding_audit['tensor_device']='cpu'
    meta=art.shared_metatype
    if meta.link_count!=len(model.names) or meta.dof_count!=model.n:
        raise RuntimeError(f'Native topology differs: links={meta.link_count}, dofs={meta.dof_count}')
    if meta.fixed_base:raise RuntimeError('Floating-base robot was incorrectly fixed to world')
    qm=name_map(model.ids,list(meta.dof_names));bm=name_map(model.names,list(meta.link_names))
    idx=io.array(np.array([0],dtype=np.uint32),indices=True)
    def ordered(values):
        a=np.empty((1,model.n),np.float32);a[0,qm]=values;return io.array(a)
    # Initialisation only; these setters are never called by the stepping loop.
    art.set_dof_positions(ordered(ref.data['q'][0]),idx)
    art.set_dof_velocities(io.array(np.zeros((1,model.n),np.float32)),idx)
    root=np.r_[P0[model.root,:3,3],Rotation.from_matrix(P0[model.root,:3,:3]).as_quat()]
    art.set_root_transforms(io.array(root[None]),idx)
    art.set_root_velocities(io.array(np.zeros((1,6),np.float32)),idx)
    art.set_dof_stiffnesses(io.array(np.zeros((1,model.n),np.float32)),idx)
    art.set_dof_dampings(ordered(model.damping),idx)
    art.set_dof_armatures(ordered(model.armature),idx)
    art.set_dof_position_targets(io.array(np.zeros((1,model.n),np.float32)),idx)
    art.set_dof_velocity_targets(io.array(np.zeros((1,model.n),np.float32)),idx)
    art.set_dof_actuation_forces(io.array(np.zeros((1,model.n),np.float32)),idx)
    # Limits on motor force are applied per physical motor before submission.
    # Native drive max force must NOT clip the independent passive damping.
    art.set_dof_max_forces(io.array(np.full((1,model.n),1e10,np.float32)),idx)
    friction=io.read(art.get_dof_friction_properties(),(1,model.n,3))
    art.set_dof_friction_properties(io.array(np.zeros_like(friction)),idx)
    view.update_articulations_kinematic()
    contacts=[]
    for name in FOOT_NAMES:
        cv=view.create_rigid_contact_view(BODIES+'/'+name,filter_patterns=[],max_contact_data_count=64)
        # A separate exact body path per view avoids relying on shape order.
        require_methods(cv, ('get_net_contact_forces',), 'PhysX foot contact view')
        if io.read(cv.get_net_contact_forces(dt=world.get_physics_dt()),(1,3),'foot force').shape!=(1,3):
            raise RuntimeError('Contact sensor did not bind exactly one source foot')
        contacts.append(cv)
    # Read back native properties, not just the authored JSON/USD.
    mass=finite(io.read(art.get_masses(),(1,78))[0,bm],(78,),'native mass')
    coms=finite(io.read(art.get_coms(),(1,78,7))[0,bm],(78,7),'native local COM poses')
    # Tensor get_inertias is about COM, already expressed in ACTOR axes.
    # Do NOT rotate it again by get_coms().quaternion (USD principal axes).
    # See Isaac Lab ArticulationData.default_inertia documentation.
    inertia=finite(io.read(art.get_inertias(),(1,78,9))[0,bm].reshape(78,3,3),(78,3,3),'native inertia')
    damp=io.read(art.get_dof_dampings(),(1,76))[0,qm].astype(float)
    arm=io.read(art.get_dof_armatures(),(1,76))[0,qm].astype(float)
    limits=io.read(art.get_dof_limits(),(1,76,2))[0,qm].astype(float)
    pose=io.read(art.get_link_transforms(),(1,78,7))[0,bm].astype(float)
    P=poses_from_xyzw(pose)
    checks={
        'body_mass_matches_source':bool(np.allclose(mass,model.mass,rtol=2e-6,atol=1e-8)),
        'local_com_matches_source':bool(np.allclose(coms[:,:3],model.com_local,rtol=2e-6,atol=2e-7)),
        'body_inertia_matches_source':inertias_match(inertia,model.I_local),
        'native_damping_SI':bool(np.allclose(damp,model.damping,rtol=0,atol=1e-6)),
        'native_armature_SI':bool(np.allclose(arm,model.armature,rtol=0,atol=1e-7)),
        'native_limits_SI':bool(np.allclose(limits,np.column_stack((model.lower,model.upper)),rtol=2e-6,atol=2e-6)),
        'initial_body_poses':bool(np.allclose(P,P0,rtol=0,atol=5e-6)),
        'no_native_stiffness':bool(np.all(io.read(art.get_dof_stiffnesses(),(1,76))==0)),
        'no_native_extra_friction':bool(np.all(io.read(art.get_dof_friction_properties(),(1,76,3))==0)),
    }
    audit={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
           'source_to_native_dof_indices':qm,'source_to_native_body_indices':bm,
           'native_dof_names':list(meta.dof_names),'native_body_names':list(meta.link_names),
           'body_count':int(meta.link_count),'tree_dofs':int(meta.dof_count),'fixed_base':bool(meta.fixed_base),
           'mass_kg':float(mass.sum()),'maximum_initial_pose_error':float(abs(P-P0).max()),
           'maximum_native_inertia_error':float(abs(inertia-model.I_local).max()),
           'initial_state_setter_calls':4,'runtime_state_projection':False,
           'tensor_frontend':io.frontend,'physics_engine':'CPU PhysX','managed_binding':binding_audit,
           'velocity_convention':'world linear COM velocity, world angular velocity',
           'pose_convention':'body origin xyz, quaternion xyzw',
           'inertia_convention':'about COM, expressed in actor/link axes; not principal axes',
           'native_mass_kg':mass,'native_local_com_pose_xyzw':coms,'native_inertia_actor_kg_m2':inertia,
           'native_damping_SI':damp,'native_armature_SI':arm,'native_limits_SI':limits}
    if not all(checks.values()):raise RuntimeError('Native model readback failed: '+json.dumps(audit,default=lambda x:x.tolist()))
    return view,art,contacts,qm,bm,idx,audit,io


def run(case,cfg:Config,output,headless=False,visuals=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'worker_status.json',{'status':'STARTING','engine':'Isaac Sim / PhysX','case':case,'certified_ready':False})
    app=None;recorder=None;control_trace=None;completed=False;native_audit={'status':'NOT_RUN'};start=time.perf_counter()
    startup={'release':VERSION,'repair':REPAIR,'phase':'BEFORE_SIMULATION_APP',
             'native_execution':'NOT_STARTED','certified_ready':False}
    def phase(name,**extra):
        startup.update(phase=name,**extra)
        write_json(output/'startup_diagnostics.json',startup)
        print(f'[Kangaroo {VERSION}] {name}',flush=True)
    phase('STARTING_SIMULATION_APP')
    try:
        # Do not import pxr, omni, core, or tensors before this point.
        import isaacsim
        from isaacsim.simulation_app import SimulationApp
        from .solver_settings import app_configuration
        app=SimulationApp(app_configuration(cfg,headless))
        phase('SIMULATION_APP_READY',native_execution='APP_STARTED')
        from isaacsim.core.api import World
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('omni.physx.tensors')
        from .usd_builder import build,audit_stage
        model=Model();ref=Reference();controller=Controller(model,ref,cfg)
        contact_gate=ContactGate(cfg);control_trace=ControllerTrace(cfg)
        world=World(physics_dt=cfg.dt_s,rendering_dt=cfg.dt_s,stage_units_in_meters=1.,backend='numpy',device='cpu')
        phase('BUILDING_NATIVE_MODEL')
        P0=build(world.stage,model,ref,cfg,visuals=visuals)
        usd_audit=audit_stage(world.stage,model,cfg)
        write_json(output/'usd_audit.json',usd_audit)
        world.stage.GetRootLayer().Export(str(output/'kangaroo_scene.usda'))
        if not headless:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([2.1,-2.4,1.7]),target=np.array([0,0,.65]),camera_prim_path='/OmniverseKit_Persp')
        import omni.usd
        context=omni.usd.get_context()
        if context.get_stage() is None or context.get_stage()!=world.stage:
            raise RuntimeError('World.stage is not the current Isaac USD stage')
        startup['usd_stage_id']=int(context.get_stage_id())
        startup['usd_root_layer']=world.stage.GetRootLayer().identifier
        startup['physics_scene_path']=SCENE
        phase('RESETTING_NATIVE_WORLD')
        world.reset()
        binding={}
        startup['physics_binding']=binding
        phase('BINDING_MANAGED_PHYSICS_VIEW')
        view,art,contacts,qm,bm,idx,native_audit,io=initialize_tensors(world,model,P0,ref,binding_audit=binding)
        from .solver_settings import read_stage_solver_settings
        native_audit['post_reset_solver_settings']=read_stage_solver_settings(world.stage,cfg)
        native_audit['feedforward_contact_gate']=contact_gate.description()
        from .contact_model import describe as describe_contact_model
        native_audit['contact_model']=describe_contact_model(cfg)
        native_audit['authored_contact_material']=usd_audit.get('authored_contact_material')
        phase('NATIVE_MODEL_READBACK_PASSED')
        write_json(output/'native_audit.json',native_audit)
        write_json(output/'configuration.json',dict(asdict(cfg),case=case,headless=headless,visuals=visuals))
        recorder=Recorder(model,ref,cfg)
        nsteps=round(cfg.duration_s/cfg.dt_s);sample_stride=round(cfg.sample_period_s/cfg.dt_s)
        render_stride=max(1,round(cfg.render_period_s/cfg.dt_s))
        effort=np.zeros((1,76),np.float32)
        bodyforce=np.zeros((1,78,3),np.float32);bodytorque=np.zeros_like(bodyforce)
        bodyposition=np.zeros_like(bodyforce)
        effort_buffer=io.buffer(effort);force_buffer=io.buffer(bodyforce)
        torque_buffer=io.buffer(bodytorque);position_buffer=io.buffer(bodyposition)
        startup['command_buffers_share_host_memory']=all(b.shared for b in
            (effort_buffer,force_buffer,torque_buffer,position_buffer))
        startup['disturbance_application_point']='measured torso center of mass, world frame'
        phase('RUNNING_NATIVE_PHYSICS',native_execution='RUNNING')
        basetime=float(world.current_time);initial_step_index=int(world.current_time_step_index)
        from .timing import StepTiming
        timing=StepTiming()
        loop_start=time.perf_counter();last_print=-1
        def state():
            view.update_articulations_kinematic()
            xyzw=finite(io.read(art.get_link_transforms(),(1,78,7))[0,bm].copy(),(78,7),'native poses')
            return (poses_from_xyzw(xyzw),
                    finite(io.read(art.get_link_velocities(),(1,78,6))[0,bm].copy(),(78,6),'native COM velocities'),
                    finite(io.read(art.get_dof_positions(),(1,76))[0,qm].copy(),(76,),'native q'),
                    finite(io.read(art.get_dof_velocities(),(1,76))[0,qm].copy(),(76,),'native qd'),xyzw)
        P,V,q,qd,xyzw=state()
        footforce=np.zeros((2,3))
        recorder.observe(0,P,V,q,qd,np.zeros(12),controller.command,footforce,np.zeros(3),xyzw,save=True)
        write_json(output/'worker_status.json',{'status':'RUNNING','engine':'Isaac Sim / PhysX','case':case,'certified_ready':False})
        for k in range(nsteps):
            timing.begin()
            t=k*cfg.dt_s
            if not app.is_running() or not world.is_playing():raise RuntimeError('Application closed or timeline paused during experiment')
            if k%controller.control_stride==0:
                # Causal tick: geometry, force and motor state are all read at t_k,
                # before stepping. Force is the preceding step's impulse / dt.
                contact=contact_gate.evaluate(model.foot_points(P),footforce)
                controller.update_command(k,q[model.active],qd[model.active],contact.selected)
                control_trace.record(k,contact,q[model.active],qd[model.active],controller)
            force=controller.force.astype(np.float32).astype(float)
            effort.fill(0.);effort[0,qm[model.active]]=force
            art.set_dof_actuation_forces(effort_buffer.native,idx)
            push=external_push(t,cfg)
            fill_disturbance_buffers(model,P,bm,push,bodyforce,bodyposition)
            # Only the prescribed disturbance is a body wrench. All feedback
            # control enters through twelve named prismatic force ports.
            art.apply_forces_and_torques_at_position(force_data=force_buffer.native,torque_data=torque_buffer.native,
                position_data=position_buffer.native,indices=idx,is_global=True)
            preV=V;preqd=qd
            timing.mark('control_and_submission')
            world.step(render=(not headless and (k+1)%render_stride==0))
            timing.mark('physics_step')
            P,V,q,qd,xyzw=state()
            footforce=np.vstack([io.read(x.get_net_contact_forces(dt=cfg.dt_s),(1,3),'foot force') for x in contacts]).astype(float)
            timing.mark('state_and_contact_readback')
            recorder.integrate_step(force,preqd,qd,push,preV,V)
            controller.advance_filter()
            recorder.observe(k+1,P,V,q,qd,force,controller.command,footforce,push,xyzw,
                             save=((k+1)%sample_stride==0 or k+1==nsteps))
            if not (cfg.no_contact or cfg.no_loops or cfg.passive):
                if P[model.root,2,3]<.1 or P[model.root,2,2]<np.cos(np.radians(45.)):
                    raise RuntimeError('Safety stop: the native robot fell')
                if recorder.metrics['maximum_loop_gap_m']>.05:
                    raise RuntimeError('Safety stop: native closure gap exceeded 50 mm')
            timing.mark('measurement_and_safety')
            current_second=int((k+1)*cfg.dt_s)
            if current_second!=last_print:
                last_print=current_second
                actual_steps=int(world.current_time_step_index)-initial_step_index
                if actual_steps!=k+1:raise RuntimeError(f'Unexpected native step count: {actual_steps} != {k+1}')
                if abs(float(world.current_time)-basetime-(k+1)*cfg.dt_s)>max(1e-7,1e-7*(k+1)*cfg.dt_s):
                    raise RuntimeError('Native and experiment clocks diverged')
                elapsed=time.perf_counter()-loop_start
                print(f'{case}: t={(k+1)*cfg.dt_s:.3f}s / {cfg.duration_s:g}s; '
                      f'loop={recorder.metrics["maximum_loop_gap_m"]:.3g}m; '
                      f'tilt={recorder.metrics["maximum_tilt_deg"]:.3g}deg; '
                      f'measured RTF={(k+1)*cfg.dt_s/max(elapsed,1e-9):.4f}',flush=True)
                write_json(output/'progress.json',{'case':case,'steps':k+1,'simulated_s':(k+1)*cfg.dt_s,
                    'wall_s':elapsed,'maximum_loop_gap_m':recorder.metrics['maximum_loop_gap_m'],
                    'profiling':timing.report()})
        completed=True
        phase('NATIVE_STEPS_COMPLETED',native_execution='COMPLETED')
        elapsed=time.perf_counter()-loop_start
        metrics=recorder.summarize()
        metrics.update(wall_simulation_s=elapsed,measured_real_time_factor=cfg.duration_s/elapsed,
                       saturation_updates=controller.saturation_updates,control_updates=controller.update_count,
                       runtime_base_control_wrench=False,runtime_state_projection=False,
                       native_step_count=int(world.current_time_step_index)-initial_step_index)
        metrics['feedforward_contact_gate']=control_trace.summary()
        control_trace.save(output,completed=True)
        write_json(output/'timing.json',timing.report())
        validation=assess(case,cfg,metrics,native_audit,completed)
        result={'release':VERSION,'engine':'Isaac Sim / PhysX','case':case,'configuration':asdict(cfg),'metrics':metrics,
                'validation':validation,'completed':True,'certified_ready':False}
        np.savez_compressed(output/'native_trace.npz',**recorder.arrays())
        write_json(output/'result.json',result)
        write_json(output/'worker_status.json',{'status':'COMPLETED','engine':'Isaac Sim / PhysX','case':case,
            'functional_status':validation['functional_status'],'certified_ready':False})
        return 0 if validation['functional_status'] in ('FUNCTIONAL_GATES_PASSED','EXPECTED_FAILURE_OBSERVED') else 3
    except BaseException as exc:
        startup.update(native_execution='FAILED',error_type=type(exc).__name__,error=str(exc))
        write_json(output/'startup_diagnostics.json',startup)
        error={'status':'ERROR','engine':'Isaac Sim / PhysX','case':case,'error_type':type(exc).__name__,
               'error':str(exc),'traceback':traceback.format_exc(),'completed':False,'certified_ready':False,
               'native_audit':native_audit,'failed_phase':startup['phase'],'startup_diagnostics':startup,'wall_elapsed_s':time.perf_counter()-start}
        write_json(output/'error.json',error);write_json(output/'worker_status.json',error)
        if control_trace is not None:
            control_trace.save(output,completed=False)
        if recorder is not None and recorder.samples['time']:
            np.savez_compressed(output/'partial_native_trace.npz',**recorder.arrays())
        print(traceback.format_exc(),flush=True)
        return 2
    finally:
        if app is not None:
            try:app.close()
            except Exception:pass
