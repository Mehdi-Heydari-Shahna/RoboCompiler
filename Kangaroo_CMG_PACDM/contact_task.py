"""Floating-base landing, crouch, weight shift and push-recovery benchmark.

No state projection, pelvis weld, mocap drive, or generalized control force.
One explicit disturbance acts at the torso COM. All other control enters the
twelve source linear force ports, with assumed finite force response.
"""
from pathlib import Path
import argparse, json, time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from reconstructed_model import whole_body
from native_model import export
from whole_body_motion import NativeMetrics
from contact_reference import DURATION, FOOT_NAMES

ROOT=Path(__file__).resolve().parent
DEFAULTS=dict(drop_height_m=.05,push_force_N=50.,push_start_s=6.7,push_duration_s=.25,
              friction=.8,actuator_time_constant_s=.002,command_slew_N_s=1500000.,
              control_period_s=.001,kp_N_m=300000.,kd_N_s_m=4000.,
              loop_time_constant_s=.00005,contact_time_constant_s=.003)
HISTORY_COLUMNS=['time_s','potential_J','kinetic_J','motor_W','passive_W','loop_W',
                 'contact_W','limit_W','disturbance_W','ground_Fx_N','ground_Fy_N',
                 'ground_Fz_N','push_Fx_N','push_Fy_N','push_Fz_N','loop_gap_m',
                 'universal_dot','slide_margin_m','hinge_margin_rad','max_motor_force_N',
                 'max_motor_error_m','penetration_m','tilt_rad','com_x_m','com_y_m','com_z_m']

def make_model(name,dt,config,visual=False,no_contact=False,no_loops=False):
    c=whole_body();path=export(c,ROOT/'models'/f'{name}.xml',floating=True,drive=True,
                              visual=visual,contact=True,timestep=dt,solref=config['loop_time_constant_s'])
    tree=ET.parse(path);root=tree.getroot();root.find('option').set('cone','elliptic')
    root.find('option').set('iterations','100')
    for geom in root.iter('geom'):
        if geom.get('contype')=='1':
            geom.set('friction',f"{config['friction']} .005 .0001")
            geom.set('solref',f"{config['contact_time_constant_s']} 1")
            geom.set('solimp','.95 .99 .001')
        if geom.get('type')=='box' and geom.get('contype')=='1':
            geom.set('name',geom.get('name','') or 'foot_'+str(len(list(root.iter('geom')))))
    # Assign stable unique names while preserving the source collision boxes.
    for body in root.iter('body'):
        for geom in body.findall('geom'):
            if geom.get('type')=='box' and geom.get('contype')=='1':geom.set('name',body.get('name')+'_sole')
    for actuator in root.find('actuator'):
        actuator.tag='general';actuator.set('dyntype','filterexact')
        actuator.set('dynprm',str(config['actuator_time_constant_s']))
        actuator.set('gaintype','fixed');actuator.set('gainprm','1');actuator.set('biastype','none')
        actuator.set('actlimited','true');actuator.set('actrange','-5000 5000')
    if no_contact:root.find('option/flag').set('contact','disable')
    if no_loops:
        for eq in root.find('equality'):eq.set('active','false')
    ET.indent(root);tree.write(path,encoding='utf-8',xml_declaration=True)
    return c,path

def external_push(t,cfg):
    a=(t-cfg['push_start_s'])/cfg['push_duration_s']
    # Smooth finite-duration load, peak specified, impulse F_peak*T/2.
    return np.array([0.,cfg['push_force_N']*np.sin(np.pi*a)**2 if 0<a<1 else 0.,0.])

def run(name='landing_nominal',dt=.000025,config=None,duration=DURATION,no_contact=False,no_loops=False,passive=False):
    cfg={**DEFAULTS,**(config or {})};c,path=make_model(name,dt,cfg,no_contact=no_contact,no_loops=no_loops)
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);metrics=NativeMetrics(c,m,d)
    ref=dict(np.load(ROOT/'data/contact_reference.npz'));us=CubicSpline(ref['t'],ref['u'],axis=0);ff=CubicSpline(ref['t'],ref['force'],axis=0)
    jids=np.array([m.joint(x).id for x in c['coordinate_ids']]);qo=m.jnt_qposadr[jids];vo=m.jnt_dofadr[jids]
    ai=np.array([c['coordinate_ids'].index(x) for x in c['independent_ids']]);mq=qo[ai];mv=vo[ai]
    jmap={j['id']:j for j in c['joints']};lo=np.array([jmap[x]['limits']['lower'] for x in c['coordinate_ids']]);hi=np.array([jmap[x]['limits']['upper'] for x in c['coordinate_ids']]);slide=np.array([jmap[x]['type']=='prismatic' for x in c['coordinate_ids']])
    torso=m.body('torso').id;rootid=m.body('base_link').id;footids=[m.body(b).id for b in FOOT_NAMES]
    d.qpos[:3]=ref['base'][0]+[0,0,cfg['drop_height_m']];d.qpos[3:7]=[1,0,0,0];d.qpos[qo]=ref['q'][0]
    d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d);E0=float(d.energy.sum())
    steps=round(duration/dt);stride=max(1,round(.005/dt));control_stride=max(1,round(cfg['control_period_s']/dt))
    if abs(control_stride*dt-cfg['control_period_s'])>1e-12:raise ValueError('Controller period must be integral steps')
    H=np.empty((steps+1,len(HISTORY_COLUMNS)));samples={k:[] for k in ['time','q','v','a','act','command','motor_velocity','motor_power','motor_work','motor_positive_work','motor_negative_work','work','ledger','foot_force','foot_moment','foot_position','com','push','cop','contact_count']}
    work=np.zeros(6);mw=np.zeros(12);positive=mw.copy();negative=mw.copy();previous=None;previous_motor=None
    force_command=np.zeros(12);saturated=0;warnings0=sum(w.number for w in d.warning);touchdown=None
    maxpowererror=0.;maxfriction=0.;maxbasectrl=0.;maxforce_rate=0.;lastforce=None;contactneg=0.;start=time.time()
    maxslip=np.zeros(2);initialfeet=None;max_generalized_applied=0.;max_constraint_decomposition=0.
    for k in range(steps+1):
        t=k*dt;target=us(t);veltarget=us(t,1)
        if k%control_stride==0:
            raw=(ff(t) if d.ncon else np.zeros(12))+cfg['kp_N_m']*(target-d.qpos[mq])+cfg['kd_N_s_m']*(veltarget-d.qvel[mv])
            if passive:raw[:]=0
            saturated+=int(np.any(abs(raw)>5000));bounded=np.clip(raw,-5000,5000)
            change=cfg['command_slew_N_s']*cfg['control_period_s']
            force_command+=np.clip(bounded-force_command,-change,change);d.ctrl[:]=force_command
        push=external_push(t,cfg);d.xfrc_applied[:]=0;d.xfrc_applied[torso,:3]=push
        mujoco.mj_checkPos(m,d);mujoco.mj_checkVel(m,d);mujoco.mj_forward(m,d)
        force=d.actuator_force.copy();motorP=force*d.qvel[mv]
        types=d.efc_type;ep=d.efc_force*d.efc_vel
        loopP=float(np.sum(ep[types==0]));contactP=float(np.sum(ep[types>=5]));limitP=float(np.sum(ep[(types==3)|(types==4)]))
        Jp=np.zeros((3,m.nv));Jr=np.zeros_like(Jp);mujoco.mj_jacBodyCom(m,d,Jp,Jr,torso)
        extP=float(push@(Jp@d.qvel));motor=float(d.qfrc_actuator@d.qvel);passiveP=float(d.qfrc_passive@d.qvel)
        power=np.array([motor,passiveP,loopP,contactP,limitP,extP])
        max_constraint_decomposition=max(max_constraint_decomposition,abs(float(d.qfrc_constraint@d.qvel)-loopP-contactP-limitP))
        if previous is not None:
            work+=.5*dt*(previous+power);mw+=.5*dt*(previous_motor+motorP)
            positive+=.5*dt*(np.maximum(previous_motor,0)+np.maximum(motorP,0))
            negative+=.5*dt*(np.minimum(previous_motor,0)+np.minimum(motorP,0))
        previous=power;previous_motor=motorP.copy();ledger=float(d.energy.sum()-E0-work.sum())
        feet=np.zeros((2,3));moments=np.zeros((2,3));copweight=np.zeros(2);normal_sum=0.;penetration=0.
        for i in range(d.ncon):
            con=d.contact[i];raw=np.zeros(6);mujoco.mj_contactForce(m,d,i,raw)
            b1=m.geom_bodyid[con.geom1];b2=m.geom_bodyid[con.geom2];world=con.frame.reshape(3,3).T@raw[:3];torque=con.frame.reshape(3,3).T@raw[3:]
            for j,bid in enumerate(footids):
                sign=1 if b2==bid else (-1 if b1==bid else 0)
                if sign:feet[j]+=sign*world;moments[j]+=sign*(torque+np.cross(con.pos-d.xipos[bid],world))
            if raw[0]>1e-6:maxfriction=max(maxfriction,float(np.linalg.norm(raw[1:3])/raw[0]))
            contactneg=min(contactneg,float(raw[0]));penetration=max(penetration,max(0.,-float(con.dist)))
            copweight+=con.pos[:2]*abs(world[2]);normal_sum+=abs(world[2])
        ground=feet.sum(axis=0)
        if touchdown is None and ground[2]>1:touchdown=t
        if t>.8:
            if initialfeet is None:initialfeet=d.xpos[footids,:2].copy()
            maxslip=np.maximum(maxslip,np.linalg.norm(d.xpos[footids,:2]-initialfeet,axis=1))
        gap,angle=metrics.closure();margin=np.minimum(d.qpos[qo]-lo,hi-d.qpos[qo]);tilt=float(np.arccos(np.clip(d.xmat[rootid].reshape(3,3)[2,2],-1,1)))
        com=d.subtree_com[rootid].copy()
        H[k]=np.r_[t,d.energy,power,ground,push,gap,angle,min(margin[slide]),min(margin[~slide]),max(abs(force)),max(abs(target-d.qpos[mq])),penetration,tilt,com]
        maxpowererror=max(maxpowererror,abs(motor-motorP.sum()));maxbasectrl=max(maxbasectrl,float(max(abs(d.qfrc_actuator[:6]))))
        max_generalized_applied=max(max_generalized_applied,float(max(abs(d.qfrc_applied))))
        if lastforce is not None:maxforce_rate=max(maxforce_rate,float(max(abs(force-lastforce))/dt))
        lastforce=force
        if k%stride==0:
            vals=[t,d.qpos.copy(),d.qvel.copy(),d.qacc.copy(),force,force_command.copy(),d.qvel[mv].copy(),motorP,mw.copy(),positive.copy(),negative.copy(),work.copy(),ledger,feet,moments,d.xpos[footids].copy(),com,push,copweight/normal_sum if normal_sum>1 else np.full(2,np.nan),d.ncon]
            for key,value in zip(samples,vals):samples[key].append(value)
        if k%max(1,round(1./dt))==0:print(name,f't={t:.1f}s z={d.qpos[2]:.3f} tilt={np.degrees(tilt):.2f}deg force={max(abs(force)):.1f}N ledger={ledger:+.4f}J',flush=True)
        if k<steps:
            # Same native implicitfast pipeline as mj_step, with logging between
            # forward dynamics and integration. Avoid a duplicate forward solve.
            mujoco.mj_checkAcc(m,d);mujoco.mj_implicit(m,d)
    samples={key:np.array(value) for key,value in samples.items()};times=H[:,0];tail=times>duration-.5
    totals=np.cumsum(.5*dt*(H[1:,3:9]+H[:-1,3:9]),axis=0);balance=H[:,1:3].sum(axis=1)-E0-np.r_[0,totals.sum(axis=1)]
    mask=(samples['time']>duration-.5);ss=(samples['time']>1)&(samples['time']<6.5)
    planned=CubicSpline(ref['t'],ref['base'],axis=0)(samples['time'])
    r=dict(name=name,configuration=cfg,timestep_s=dt,duration_s=duration,steps=steps,
           no_contact=no_contact,no_loops=no_loops,passive_control=passive,
           mass_kg=float(m.body_mass.sum()),body_count=m.nbody-1,motor_count=m.nu,
           touchdown_s=touchdown,maximum_loop_gap_m=float(max(H[:,15])),maximum_universal_dot=float(max(H[:,16])),
           minimum_slide_margin_m=float(min(H[:,17])),minimum_hinge_margin_rad=float(min(H[:,18])),
           maximum_motor_force_N=float(max(H[:,19])),maximum_motor_error_m=float(max(H[:,20])),
           maximum_penetration_m=float(max(H[:,21])),maximum_tilt_deg=float(np.degrees(max(H[:,22]))),
           final_tilt_deg=float(np.degrees(H[-1,22])),maximum_foot_drift_after_landing_m=maxslip.tolist(),
           maximum_ground_normal_N=float(max(H[:,11])),mean_final_ground_normal_N=float(H[tail,11].mean()),
           achieved_crouch_m=float(np.max(samples['q'][ss,2])-np.min(samples['q'][ss,2])) if ss.any() else 0.,
           achieved_lateral_excursion_m=float(np.ptp(samples['q'][ss,1])) if ss.any() else 0.,
           maximum_final_base_speed_m_s=float(np.max(np.linalg.norm(samples['v'][mask,:3],axis=1))),
           final_base_position_m=samples['q'][-1,:3].tolist(),final_position_error_m=float(np.linalg.norm(samples['q'][-1,:3]-planned[-1])),
           final_motor_work_J=mw.tolist(),positive_motor_work_J=positive.tolist(),negative_motor_work_J=negative.tolist(),
           work_J=dict(zip(['motor','passive','loop','contact','limit','disturbance'],work.tolist())),
           energy_change_J=float(H[-1,1:3].sum()-E0),maximum_energy_ledger_error_J=float(max(abs(balance))),final_energy_ledger_error_J=float(balance[-1]),
           maximum_actuator_virtual_work_error_W=maxpowererror,maximum_constraint_power_decomposition_error_W=max_constraint_decomposition,
           maximum_contact_friction_ratio=maxfriction,minimum_contact_normal_N=contactneg,
           peak_force_rate_N_s=maxforce_rate,saturated_control_updates=saturated,
           maximum_base_actuator_generalized_force=maxbasectrl,maximum_generalized_applied_force=max_generalized_applied,
           warning_count=sum(w.number for w in d.warning)-warnings0,elapsed_seconds=time.time()-start)
    np.savez_compressed(ROOT/'results'/f'{name}.npz',history=H,**samples)
    (ROOT/'results'/f'{name}.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2),flush=True);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='landing_nominal');p.add_argument('--dt',type=float,default=.000025);p.add_argument('--duration',type=float,default=DURATION);p.add_argument('--drop',type=float,default=.05);p.add_argument('--push',type=float,default=50.);p.add_argument('--friction',type=float,default=.8);p.add_argument('--no-contact',action='store_true');p.add_argument('--no-loops',action='store_true');p.add_argument('--passive',action='store_true')
    a=p.parse_args();run(a.name,a.dt,dict(drop_height_m=a.drop,push_force_N=a.push,friction=a.friction),a.duration,a.no_contact,a.no_loops,a.passive)
