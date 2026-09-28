"""Effort-driven full-body articulated-track excavator mission.

The integrator owns every moving state after initialization. Saved states are
native states; the PACDM state is an independent control shadow only.
"""
from project import *
import argparse
import time
import numpy as np
import mujoco
from benchmark_model import context, PORTS, GRAVITY
from hydraulic_actuators import HydraulicBank
from digging_export import bucket_inside
from mobile_mission import Mission, TravelController, build_scene, STAGES
from tracked_arm import TrackedArmController
from track_drive import TrackDriveBank
from tracked_audit import CompiledTreeAudit
from native_step import validate_model, checked_forward, advance_after_forward
from native_solve import accurate_forward, FORCE_TOLERANCE

POWER_KEYS=['arm_supply','arm_throttle','arm_leakage','arm_relief','arm_friction',
            'arm_compressibility_geometry','arm_rotary','arm_mechanical',
            'drive_supply','drive_throttle','drive_leakage','drive_relief','drive_friction',
            'drive_numerical_storage_loss','drive_mechanical','equality','contact','passive']

def constraint_powers(m,d):
    jv=np.zeros(d.nefc)
    mujoco.mj_mulJacVec(m,d,jv,d.qvel)
    equality=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    contact=d.efc_type>=int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
    return float(d.efc_force[equality]@jv[equality]),float(d.efc_force[contact]@jv[contact]),float(d.qfrc_passive@d.qvel)

def run_case(name='development', *, dt=.0005, duration=None, soil=True,
             bucket_contact=True, traction=.8, density_scale=1., drive_enabled=True,
             approach=.5, distance=2., yaw=.3, hold=False, audit_interval=4.,
             save=True):
    started=time.perf_counter()
    if dt<=0 or abs(round(.001/dt)*dt-.001)>1e-12:
        raise ValueError('Physics timestep must divide the fixed 1 ms travel-servo period')
    cmg,mapping,e,a,r,inverse=context()
    mission=Mission(e,r,approach=approach,distance=distance,yaw=yaw)
    if duration is None:duration=mission.duration
    xml=RESULTS/'assets'/f'{name}.xml'
    meta=build_scene(xml,mission,dt=dt,soil=soil,bucket_contact=bucket_contact,
                     traction=traction,density_scale=density_scale)
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);validate_model(m)
    arm_jids=np.array([m.joint(k).id for k in e.tree_ids])
    qi=m.jnt_qposadr[arm_jids];vi=m.jnt_dofadr[arm_jids]
    pi=np.array([e.tree_ids.index(k) for k in PORTS])
    arm_ai=np.array([m.actuator('effort_'+k).id for k in PORTS])
    d.qpos[qi]=mission.initial_tree
    d.qvel[:]=0.
    mujoco.mj_forward(m,d)
    arm=TrackedArmController(cmg,mapping,e,a,r,inverse,mission.initial_tree,m,d)
    travel=TravelController(m,d,meta)
    desired,target,rate=mission.reference(0.)
    command=arm.evaluate(*desired,d)
    bank=HydraulicBank(r.nominal[pi],command)
    drive=TrackDriveBank()
    audit=CompiledTreeAudit(m)
    root_id=m.body('body_53').id;bucket=m.body('body_56').id
    body_ids=np.arange(1,m.nbody,dtype=int)
    grains=np.array([m.body(f'grain_{i:03d}').id for i in range(meta['particle_count'])],int)
    gm=m.body_mass[grains]
    carried=np.zeros(len(grains),bool)
    loaded_at_departure=carried.copy()
    source_bodies=[m.body(b['id']).id for b in cmg['bodies'] if b['kind']=='rigid_body']
    eq_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_EQUALITY,i) or '' for i in range(m.neq)]
    track_eqids=np.array([i for i,n in enumerate(eq_names) if n.startswith('track_') and 'closure' in n])
    arm_eqids=np.array([i for i,n in enumerate(eq_names) if not n.startswith('track_')])
    rotor_eqids=np.array([i for i,n in enumerate(eq_names) if n.startswith('track_') and 'gear' in n])
    work=np.zeros(len(POWER_KEYS));absolute=np.zeros_like(work)
    midwork=np.zeros(5);midabs=np.zeros(5)
    port_displacement_work=np.zeros(10)
    solver_stats=dict(policy='residual_controlled_native_warmstart_retry',target_relative_residual=FORCE_TOLERANCE,
        maximum_initial_residual=0.,maximum_final_residual=0.,retried_steps=0,total_native_restarts=0,total_cold_restarts=0,
        physical_input_projection=False,solver_option_mutation=False)
    trace={k:[] for k in ['time','qpos','qvel','ctrl','independent','desired','arm_effort','arm_command',
        'arm_pressure','arm_flow','drive_effort','drive_command','drive_pressure','drive_pressure_applied','drive_speed','drive_desired_speed',
        'pose','target','energy','arm_fluid_energy','drive_fluid_energy','work','power','captured_mass',
        'mechanical_midpoint_work','force_balance_initial','force_balance_final','solver_restarts',
        'carried_remaining_mass','delivered_mass','loop_gap','track_gap','gear_gap','stage','contact_count',
        'source_body_wrenches','body_contact_wrenches','base_velocity','equality_force']}
    snapshots=[]
    peak=dict(arm_gap=0.,track_gap=0.,gear_gap=0.,tracking=0.,force_balance=0.,arm_fluid_identity=0.,
              drive_fluid_identity=0.,port_power_identity=0.,drive_torque=0.,tilt=0.,pressure_bound=0.)
    initial_energy=float(d.energy.sum());initial_arm_fluid=bank.stored_energy(d.qpos[qi][pi]);initial_drive_fluid=drive.stored_energy()
    initial_pose=travel.pose(d)
    last_command=np.zeros(2)
    warnings0=np.array([x.number for x in d.warning])
    steps=round(duration/dt);control_every=max(1,round(.01/dt));servo_every=max(1,round(.001/dt));sample_every=max(1,round(.02/dt))
    audit_every=max(1,round(audit_interval/dt));print_every=max(1,round(2/dt))
    initial_grain_xyz=d.xpos[grains].copy()
    initial_physical_arrays=None
    for step in range(steps+1):
        t=step*dt
        desired,target,rate=mission.reference(0 if hold else t)
        if hold:target=np.array([-approach,0.,0.]);rate=np.zeros(3)
        q=d.qpos[qi].copy();v=d.qvel[vi].copy()
        if step%control_every==0:
            command=arm.evaluate(*desired,d)
        if step%servo_every==0:
            last_command=travel.evaluate(d,target,rate,servo_every*dt,update_outer=step%control_every==0)
        item=bank.evaluate(q[pi],v[pi],command,dt)
        di=drive.evaluate(d.qvel[travel.vi],last_command if drive_enabled else np.zeros(2),dt)
        d.ctrl[arm_ai]=item['effort'];d.ctrl[travel.ai]=di['effort'] if drive_enabled else 0.
        # All external applied wrenches remain identically zero. Contact and
        # equality forces are generated inside the native mechanics.
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            raise ValueError('Unexpected external applied force')
        warmstart_before=d.qacc_warmstart.copy()
        solve=accurate_forward(m,d)
        if solve['initial_residual']>max(FORCE_TOLERANCE,solver_stats['maximum_initial_residual']):
            save_npz(RESULTS/f'{name}_solver_retry_worst.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),
                qacc_warmstart=warmstart_before,time=t,residual_history=np.asarray(solve['residual_history']))
        solver_stats['maximum_initial_residual']=max(solver_stats['maximum_initial_residual'],solve['initial_residual'])
        solver_stats['maximum_final_residual']=max(solver_stats['maximum_final_residual'],solve['final_residual'])
        solver_stats['retried_steps']+=int(solve['retry_count']>0)
        solver_stats['total_native_restarts']+=solve['retry_count']
        solver_stats['total_cold_restarts']+=solve['cold_count']
        ep,cp,pp=constraint_powers(m,d)
        power=np.array([item['power'][k[4:]] if k.startswith('arm_') else
                        di['power'][k[6:]] if k.startswith('drive_') else
                        {'equality':ep,'contact':cp,'passive':pp}[k] for k in POWER_KEYS])
        if not drive_enabled:
            # Disconnected drive bank consumes no source flow or shaft work.
            power[[i for i,k in enumerate(POWER_KEYS) if k.startswith('drive_')]]=0.
        eqmask=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        arm_mask=eqmask&np.isin(d.efc_id,arm_eqids)
        belt_mask=eqmask&np.isin(d.efc_id,track_eqids)
        gear_mask=eqmask&np.isin(d.efc_id,rotor_eqids)
        ag=float(np.max(abs(d.efc_pos[arm_mask]),initial=0));tg=float(np.max(abs(d.efc_pos[belt_mask]),initial=0));gg=float(np.max(abs(d.efc_pos[gear_mask]),initial=0))
        err=float(np.max(abs(q[e.active[:6]]-desired[0][:6])))
        ma=np.zeros(m.nv);mujoco.mj_mulM(m,d,ma,d.qacc)
        force_res=float(np.max(abs(ma-d.qfrc_smooth-d.qfrc_constraint)))/max(1.,float(np.max(abs(ma))),float(np.max(abs(d.qfrc_bias))))
        if force_res>1e-6 and force_res>peak['force_balance']:
            np.savez_compressed(RESULTS/f'{name}_force_worst.npz',
                qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc=d.qacc.copy(),
                qacc_warmstart=warmstart_before,qfrc_smooth=d.qfrc_smooth.copy(),
                qfrc_constraint=d.qfrc_constraint.copy(),qfrc_bias=d.qfrc_bias.copy(),
                solver_niter=d.solver_niter.copy(),solver_improvement=np.asarray(d.solver.improvement).copy(),
                solver_gradient=np.asarray(d.solver.gradient).copy(),time=t,relative_residual=force_res)
        power_res=abs(float(d.qfrc_actuator@d.qvel)-item['power']['mechanical']-(di['power']['mechanical'] if drive_enabled else 0.))
        tilt=np.arccos(np.clip(d.xmat[root_id].reshape(3,3)[2,2],-1,1))
        for key,val in dict(arm_gap=ag,track_gap=tg,gear_gap=gg,tracking=err,force_balance=force_res,
            arm_fluid_identity=abs(item['fluid_identity_W']),drive_fluid_identity=abs(di['fluid_identity_W']) if drive_enabled else 0.,
            port_power_identity=power_res,drive_torque=np.max(abs(di['effort'])) if drive_enabled else 0.,tilt=tilt,
            pressure_bound=max(0.,np.max(bank.pressure)-bank.ps,-np.min(bank.pressure),np.max(drive.pressure)-drive.ps,-np.min(drive.pressure))).items():
            peak[key]=max(peak[key],float(val))
        captured=remaining=delivered=0.
        if len(grains):
            local=(d.xpos[grains]-d.xpos[bucket])@d.xmat[bucket].reshape(3,3)
            inside=bucket_inside(local);lifted=d.xpos[grains,2]>.9
            if mission.departure-1<=t<=mission.loaded_arrival:carried|=inside&lifted
            if abs(t-mission.departure)<dt/2:loaded_at_departure|=inside&lifted
            captured=float(gm[inside].sum());remaining=float(gm[inside&carried].sum())
            in_depot=np.all(abs(d.xpos[grains,:2]-mission.depot_center[:2])<=mission.depot_half_size,axis=1)&(d.xpos[grains,2]<.8)
            delivered=float(gm[in_depot&carried].sum())
        if step%audit_every==0 or step==steps:
            snap=audit.audit_snapshot(d);snap['time_s']=t;snapshots.append(snap)
        if step%sample_every==0 or step==steps:
            mujoco.mj_rnePostConstraint(m,d)
            # Wrenches are calculated explicitly below to keep body order and
            # force-first convention stable across report versions.
            bw=np.zeros((m.nbody,6));raw=np.zeros(6)
            for ci in range(d.ncon):
                con=d.contact[ci];mujoco.mj_contactForce(m,d,ci,raw);R=con.frame.reshape(3,3)
                for bid,sgn in [(m.geom_bodyid[con.geom1],-1),(m.geom_bodyid[con.geom2],1)]:
                    f=sgn*(R.T@raw[:3]);tau=sgn*(R.T@raw[3:])
                    bw[bid,:3]+=f;bw[bid,3:]+=tau+np.cross(con.pos-d.xipos[bid],f)
            pose=travel.pose(d)
            vals=dict(time=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),independent=q[e.active],desired=desired[0][:6],
                arm_effort=item['effort'],arm_command=command.copy(),arm_pressure=bank.pressure.copy(),arm_flow=item['flow'],
                drive_effort=di['effort'] if drive_enabled else np.zeros(2),drive_command=last_command.copy(),drive_pressure=drive.pressure.copy(),
                drive_pressure_applied=di['pressure_next'] if drive_enabled else drive.pressure.copy(),
                drive_speed=d.qvel[travel.vi].copy(),drive_desired_speed=travel.last.get('desired_speed',np.zeros(2)),
                pose=pose,target=target,energy=d.energy.copy(),arm_fluid_energy=bank.stored_energy(q[pi]),drive_fluid_energy=drive.stored_energy(),
                work=work.copy(),power=power,mechanical_midpoint_work=midwork.copy(),captured_mass=captured,carried_remaining_mass=remaining,delivered_mass=delivered,
                force_balance_initial=solve['initial_residual'],force_balance_final=solve['final_residual'],solver_restarts=solve['retry_count'],
                loop_gap=ag,track_gap=tg,gear_gap=gg,stage=mission.stage(t),contact_count=d.ncon,
                source_body_wrenches=d.cfrc_int[source_bodies].copy(),body_contact_wrenches=bw[body_ids],
                base_velocity=d.qvel[:6].copy(),equality_force=d.efc_force[eqmask].copy())
            for key,value in vals.items():trace[key].append(value)
        if step%print_every==0:
            p=travel.pose(d)
            print(f'{name}: t={t:5.1f} pose=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) arm={err:.3g} belt={tg:.2e} load={captured:.1f}kg contacts={d.ncon} wall={time.perf_counter()-started:.1f}s',flush=True)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            raise ValueError('Nonfinite native state')
        if np.any(np.array([x.number for x in d.warning])!=warnings0):
            raise ValueError('Native solver warning')
        if step==steps:break
        work+=dt*power;absolute+=dt*abs(power)
        port_q0=np.r_[q[pi],d.qpos[m.jnt_qposadr[travel.jids]]]
        effort=np.r_[item['effort'],di['effort'] if drive_enabled else np.zeros(2)]
        old_velocity=d.qvel.copy()
        # Constraint rows and generalized forces refer to this pre-step state.
        # Their work uses a midpoint velocity quadrature; the original held-
        # speed hydraulic ledger is retained separately rather than overwritten.
        equality_force=d.efc_force[eqmask].copy()
        contact_mask=d.efc_type>=int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        contact_force=d.efc_force[contact_mask].copy()
        passive_force=d.qfrc_passive.copy()
        advance_after_forward(m,d)
        midpoint_velocity=.5*(old_velocity+d.qvel)
        jmid=np.zeros(d.nefc);mujoco.mj_mulJacVec(m,d,jmid,midpoint_velocity)
        mids=np.array([item['effort']@midpoint_velocity[vi][pi],
                      (di['effort']@midpoint_velocity[travel.vi]) if drive_enabled else 0.,
                      equality_force@jmid[eqmask],contact_force@jmid[contact_mask],passive_force@midpoint_velocity])
        midwork+=dt*mids;midabs+=dt*abs(mids)
        port_q1=np.r_[d.qpos[qi][pi],d.qpos[m.jnt_qposadr[travel.jids]]]
        port_displacement_work+=effort*(port_q1-port_q0)
        bank.advance(item,dt)
        if drive_enabled:drive.advance(di,dt)
    result={k:np.asarray(v) for k,v in trace.items()}
    wi=dict(zip(POWER_KEYS,map(float,work)));aw=dict(zip(POWER_KEYS,map(float,absolute)))
    delta_mechanical=float(d.energy.sum())-initial_energy
    rectangle_rhs=wi['arm_mechanical']+wi['drive_mechanical']+wi['equality']+wi['contact']+wi['passive']
    mech_rhs=float(midwork.sum());mech_defect=delta_mechanical-mech_rhs
    arm_rhs=wi['arm_supply']-wi['arm_throttle']-wi['arm_leakage']-wi['arm_relief']-wi['arm_friction']+wi['arm_compressibility_geometry']+wi['arm_rotary']
    drive_rhs=wi['drive_supply']-wi['drive_throttle']-wi['drive_leakage']-wi['drive_relief']-wi['drive_friction']-wi['drive_numerical_storage_loss']
    fluid_delta=bank.stored_energy(d.qpos[qi][pi])-initial_arm_fluid+drive.stored_energy()-initial_drive_fluid
    total_defect=delta_mechanical+fluid_delta-arm_rhs-drive_rhs-wi['equality']-wi['contact']-wi['passive']
    pose=travel.pose(d);terminal_target=mission.reference(0 if hold else duration)[1]
    loaded_window=(result['time']>=mission.departure)&(result['time']<=mission.loaded_arrival)
    loaded_travel=float(np.linalg.norm(result['pose'][loaded_window][-1,:2]-result['pose'][loaded_window][0,:2])) if np.sum(loaded_window)>1 else 0.
    report=dict(name=name,completed=True,dt_s=dt,duration_s=duration,elapsed_s=time.perf_counter()-started,scene=meta,
        bodies=m.nbody-1,coordinates=m.nv,actuators=m.nu,source_mass_kg=sum(b['mass_kg'] for b in cmg['bodies']),
        peak=peak,work_J=wi,absolute_work_J=aw,mechanical_energy_change_J=delta_mechanical,fluid_energy_change_J=fluid_delta,
        mechanical_balance_defect_J=mech_defect,mechanical_balance_relative=abs(mech_defect)/max(1.,float(midabs.sum())),
        mechanical_midpoint_work_J=dict(zip(['arm_mechanical','drive_mechanical','equality','contact','passive'],map(float,midwork))),
        mechanical_midpoint_absolute_work_J=dict(zip(['arm_mechanical','drive_mechanical','equality','contact','passive'],map(float,midabs))),
        mechanical_rectangle_defect_J=delta_mechanical-rectangle_rhs,
        mechanical_work_convention='Native pre-step generalized forces times average of pre/post-step velocities; held-speed hydraulic and displacement ledgers separately retained.',
        total_balance_defect_J=total_defect,total_balance_relative=abs(total_defect)/max(1.,sum(v for k,v in aw.items() if k not in ['arm_mechanical','drive_mechanical'])),
        port_displacement_work_J=port_displacement_work,drive_partition_coupling_defect_J=float(port_displacement_work[8:].sum()-wi['drive_mechanical']),
        initial_pose=initial_pose,final_pose=pose,final_target=terminal_target,
        final_position_error_m=float(np.linalg.norm(pose[:2]-terminal_target[:2])),final_heading_error_rad=abs(float(np.arctan2(np.sin(pose[2]-terminal_target[2]),np.cos(pose[2]-terminal_target[2])))),
        peak_captured_mass_kg=float(result['captured_mass'].max()),loaded_at_departure_mass_kg=float(gm[loaded_at_departure].sum()),
        carried_mass_kg=float(gm[carried].sum()),delivered_mass_kg=delivered,loaded_base_displacement_m=loaded_travel,
        final_particle_positions_m=d.xpos[grains].copy(),initial_particle_positions_m=initial_grain_xyz,
        carried_particle_ids=np.where(carried)[0],arm_controller=arm.diagnostics(),warning_count=0,
        numerical_solver=solver_stats,
        native_state_projection=False,base_propulsion_wrench=False,q22_actuated=False,p0_command_N=0.,
        body_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) for b in body_ids],
        source_wrench_body_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) for b in source_bodies],
        power_order=POWER_KEYS,stage_names=STAGES,audit_passed=all(x['passed'] for x in snapshots),
        scope='Explicit-track synthetic mobile excavator; simulation verification, uncalibrated drivetrain and coarse rigid aggregate')
    if save:
        save_npz(RESULTS/f'{name}.npz',**result)
        save_json(RESULTS/f'{name}.json',report)
        save_json(RESULTS/f'{name}_audit.json',dict(passed=report['audit_passed'],snapshots=snapshots,frame_covariance=audit.frame_covariance(d.qpos,d.qvel)))
    return report,result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='development');p.add_argument('--dt',type=float,default=.0005);p.add_argument('--duration',type=float)
    p.add_argument('--empty',action='store_true');p.add_argument('--hold',action='store_true');p.add_argument('--no-drive',action='store_true')
    p.add_argument('--traction',type=float,default=.8);p.add_argument('--density-scale',type=float,default=1.)
    args=p.parse_args()
    report,_=run_case(args.name,dt=args.dt,duration=args.duration,soil=not args.empty,hold=args.hold,drive_enabled=not args.no_drive,traction=args.traction,density_scale=args.density_scale)
    print(json.dumps({k:v for k,v in report.items() if k in ['name','peak','final_pose','audit_passed','mechanical_balance_relative','carried_mass_kg','delivered_mass_kg']},default=lambda v:v.tolist(),indent=2))
