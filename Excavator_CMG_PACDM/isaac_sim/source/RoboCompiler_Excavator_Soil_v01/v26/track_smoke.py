"""Independent isolated articulated track contact and propulsion experiment."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
from track_model import isolated_scene


def run(output='outputs/track_smoke', duration=5., dt=.0005, case='straight', payload_mass=9425.452625161175, friction=.8, torque_limit=8000.):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta=isolated_scene(out/'model.xml', config={'torque_limit':torque_limit}, dt=dt,payload_mass=payload_mass,friction=friction)
    m=mujoco.MjModel.from_xml_path(str(out/'model.xml'));d=mujoco.MjData(m)
    if case=='no_teeth':
        for gid in range(m.ngeom):
            if '_tooth_' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gid) or ''):
                m.geom_contype[gid]=0;m.geom_conaffinity[gid]=0
    mujoco.mj_forward(m,d)
    aid=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in meta['drive_actuator_names']])
    jid=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in meta['drive_joint_names']])
    dof=m.jnt_dofadr[jid]
    base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'body_53')
    start=d.xpos[base].copy(); center_local=np.array([-.4525,2.1034,.344]);start_center=d.xpos[base]+d.xmat[base].reshape(3,3)@center_local
    samples=[];gapmax=0;vmax=0;contactmax=0;toothcontactmax=0;work=0.;lateralmax=0.;axismax=0.;gearmax=0.;startclock=time.perf_counter()
    shoe_groups=[(side['center_x_m'],np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in side['shoe_bodies']])) for side in meta['sides']]
    rotj=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'track_{s}_motor_rotor_axle') for s in ('left','right')])
    gear_q=m.jnt_qposadr[rotj];sprocket_q=m.jnt_qposadr[jid]
    closure=[(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'track_{side}_closure_first_{j}'),mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'track_{side}_closure_last_{j}')) for side in ('left','right') for j in (0,1)]
    warnings=[];integral=np.zeros(2)
    for k in range(round(duration/dt)):
        t=d.time
        target=max(0.,min((t-1)/1.,1.))*.5
        if case=='unpowered':targets=np.array([0.,0.])
        elif case=='turn':targets=np.array([-.5,.5])*max(0.,min((t-1),1.))
        else:targets=np.array([target,target])
        # Test speed controller applies bounded torque; it does not impose motion.
        error=targets-d.qvel[dof]
        integral=np.clip(integral+error*dt,-torque_limit/18000,torque_limit/18000)
        torque=np.clip(6000*error+18000*integral,-torque_limit,torque_limit) if case!='unpowered' else np.zeros(2)
        d.ctrl[aid]=torque
        work+=float(torque@d.qvel[dof])*dt
        mujoco.mj_step(m,d)
        if not np.all(np.isfinite(d.qpos)):raise RuntimeError('Nonfinite state')
        if k%20==0:
            gap=max(np.linalg.norm(d.site_xpos[a]-d.site_xpos[b]) for a,b in closure)
            gapmax=max(gapmax,gap);vmax=max(vmax,float(np.max(np.abs(d.qvel))));contactmax=max(contactmax,d.ncon)
            tooth=sum(('tooth_' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom1)) or '') or 'tooth_' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom2)) or '')) for con in d.contact)
            toothcontactmax=max(toothcontactmax,tooth)
            R=d.xmat[base].reshape(3,3)
            for sx,bodies in shoe_groups:
                local=(d.xpos[bodies]-d.xpos[base])@R
                lateralmax=max(lateralmax,float(np.max(abs(local[:,0]-sx))))
                axismax=max(axismax,float(np.arccos(np.clip(R[:,0]@d.xmat[bodies[0]].reshape(3,3)[:,0],-1,1))))
            gearmax=max(gearmax,float(np.max(abs(d.qpos[gear_q]-45*d.qpos[sprocket_q]))))
            samples.append([float(d.time),*d.xpos[base].tolist(),float(np.arctan2(R[1,0],R[0,0]))-np.pi/2,*d.qvel[dof].tolist(),*torque.tolist(),gap,d.ncon,tooth,*d.energy.tolist()])
        if k%2000==0:print(f'{case} t={d.time:.3f} dx={d.xpos[base,0]-start[0]:.4f} z={d.xpos[base,2]:.4f} ncon={d.ncon} vmax={np.max(abs(d.qvel)):.3f}',flush=True)
    mujoco.mj_forward(m,d)
    np.savez_compressed(out/'final_state.npz',qpos=d.qpos,qvel=d.qvel)
    np.savetxt(out/'samples.csv',np.array(samples),delimiter=',',header='time,x,y,z,yaw,speed_left,speed_right,torque_left,torque_right,loopgap,contacts,toothcontacts,PE,KE',comments='')
    result=dict(case=case,duration_s=duration,dt_s=dt,wall_s=time.perf_counter()-startclock,
        displacement_m=(d.xpos[base]-start).tolist(),yaw_change_rad=float(np.arctan2(d.xmat[base].reshape(3,3)[1,0],d.xmat[base].reshape(3,3)[0,0]))-np.pi/2,
        max_belt_loop_gap_m=gapmax,max_track_lateral_deviation_m=lateralmax,max_track_axis_error_rad=axismax,max_gear_angle_residual_rad=gearmax,
        chassis_center_displacement_m=(d.xpos[base]+d.xmat[base].reshape(3,3)@center_local-start_center).tolist(),max_abs_qvel=vmax,max_contacts=contactmax,max_tooth_contacts=toothcontactmax,
        drive_work_J=work,final_sprocket_speeds_rad_s=d.qvel[dof].tolist(),
        warnings={str(i):int(w.number) for i,w in enumerate(d.warning) if w.number},
        reference_mass_kg=float(np.sum(m.body_mass)),positive_drive_contact=bool(toothcontactmax>0),metadata=meta)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='metadata'},indent=2),flush=True)
    return result

def assess_suite(output='outputs/track_validated'):
    """Assess a completed, identical-model isolated validation suite.

    Direct bounded motor torques isolate track mechanics from the separate hydraulic
    model. Limits are engineering acceptance criteria, not hardware certification.
    """
    out=Path(output)
    cases={n:json.loads((out/n/'result.json').read_text()) for n in
           ('coarse','nominal','fine','turn','no_teeth','unpowered')}
    gates=[]
    def gate(name,value,limit,passed):
        gates.append(dict(name=name,value=value,limit=limit,passed=bool(passed)))
    for name,r in cases.items():
        gate(name+'_no_native_warnings',len(r['warnings']),0,not r['warnings'])
        gate(name+'_belt_closure_m',r['max_belt_loop_gap_m'],.001,r['max_belt_loop_gap_m']<.001)
        gate(name+'_lateral_retention_m',r['max_track_lateral_deviation_m'],.025,r['max_track_lateral_deviation_m']<.025)
        gate(name+'_gear_constraint_rad',r['max_gear_angle_residual_rad'],.001,r['max_gear_angle_residual_rad']<.001)
        for metric,limit in [('mass_reconstruction_error_kg',1e-9),('com_reconstruction_error_m',1e-12),('inertia_reconstruction_error_kg_m2',1e-8)]:
            value=r['metadata'][metric];gate(name+'_'+metric,value,limit,value<limit)
    for name in ('coarse','nominal','fine'):
        r=cases[name];move=r['chassis_center_displacement_m']
        gate(name+'_forward_progress_m',move[0],.4,move[0]>.4)
        gate(name+'_lateral_error_m',abs(move[1]),.01,abs(move[1])<.01)
        gate(name+'_yaw_error_rad',abs(r['yaw_change_rad']),.01,abs(r['yaw_change_rad'])<.01)
        error=float(max(abs(np.array(r['final_sprocket_speeds_rad_s'])-.5)))
        gate(name+'_final_drive_speed_error_rad_s',error,.1,error<.1)
    position_spread=float(np.ptp([cases[n]['chassis_center_displacement_m'][0] for n in ('coarse','nominal','fine')]))
    gate('three_timestep_forward_position_spread_m',position_spread,.02,position_spread<.02)
    yaw=cases['turn']['yaw_change_rad'];gate('differential_drive_turn_rad',yaw,.2,yaw>.2)
    gate('positive_sprocket_engagement_contacts',cases['nominal']['max_tooth_contacts'],0,cases['nominal']['max_tooth_contacts']>0)
    gate('disabled_teeth_zero_engagement_contacts',cases['no_teeth']['max_tooth_contacts'],0,cases['no_teeth']['max_tooth_contacts']==0)
    nominal_progress=cases['nominal']['chassis_center_displacement_m'][0]
    for name in ('unpowered','no_teeth'):
        ratio=abs(cases[name]['chassis_center_displacement_m'][0])/nominal_progress
        gate(name+'_travel_fraction_of_driven',ratio,.25,ratio<.25)
    work_ratio=abs(cases['no_teeth']['drive_work_J'])/abs(cases['nominal']['drive_work_J'])
    gate('disabled_teeth_work_fraction_of_driven',work_ratio,.1,work_ratio<.1)
    numerical_settings={}
    for name in cases:
        compiled=mujoco.MjModel.from_xml_path(str(out/name/'model.xml'))
        numerical_settings[name]=dict(mujoco_version=mujoco.__version__,
            timestep_s=float(compiled.opt.timestep),newton_tolerance=float(compiled.opt.tolerance),
            iterations=int(compiled.opt.iterations),line_search_iterations=int(compiled.opt.ls_iterations),
            integrator='implicitfast',solver='Newton',cone='elliptic')
    result=dict(status='PASS' if all(g['passed'] for g in gates) else 'FAIL',
                passed=sum(g['passed'] for g in gates),total=len(gates),gates=gates,
                numerical_settings=numerical_settings,
                scope='Isolated rigid ballast with full original machine supported mass; direct torque propulsion; hydraulic and articulated arm validation are separate.',
                methodology='Native free base, free articulated belt roots, native hinge chains, two-site seam closure, explicit geared motor rotors, physical roller/flange and tooth/pin contact. No trajectory assignment or external base wrench.',
                assumptions=cases['nominal']['metadata']['assumptions'])
    (out/'assessment.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/track_smoke');p.add_argument('--assess',action='store_true');p.add_argument('--duration',type=float,default=5.);p.add_argument('--dt',type=float,default=.0005);p.add_argument('--case',choices=['straight','turn','unpowered','no_teeth'],default='straight');p.add_argument('--payload-mass',type=float,default=9425.452625161175);p.add_argument('--friction',type=float,default=.8);p.add_argument('--torque-limit',type=float,default=8000.)
    a=p.parse_args()
    if a.assess: print(json.dumps(assess_suite(a.output),indent=2))
    else: run(a.output,a.duration,a.dt,a.case,a.payload_mass,a.friction,a.torque_limit)
