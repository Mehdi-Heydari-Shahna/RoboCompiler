"""Underactuated whole-body inverse dynamics with native unilateral contacts.

Only the twelve authored joint motors receive commands. Base forces are
zero except for the explicitly logged disturbance pulse. Contact forces in
the QP are predictions; MuJoCo computes all actual ground reactions.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from scipy.interpolate import BPoly
import osqp
import mujoco
from .model import load_model,chart_from_native,chart_velocity_from_native,native_qpos_from_chart
from .pin_backend import PinBackend
from .contact import FootKinematics
from .task import HURDLES,target_trajectory

def build_scene(root,name='nominal',dt=.001,friction=.8,payload=0.,terrain=True):
    root=Path(root);tree=ET.parse(root/'upstream/unitree_go2/go2.xml');e=tree.getroot()
    e.set('model','Go2 CMG PACDM torque-driven contact course')
    e.find('compiler').set('meshdir','../upstream/unitree_go2/assets')
    e.find('option').attrib.update(timestep=str(dt),integrator='implicitfast',iterations='80',tolerance='1e-10')
    for x in e.findall('keyframe'):e.remove(x)
    # Benchmark contact stiffness: avoid the source rubber approximation's
    # centimetre-scale compression at rail edges. Original XML is preserved.
    e.find(".//default[@class='foot']/geom").attrib.update(
        friction=f'{friction} .02 .01',solref='.008 1',solimp='.95 .99 .001')
    vis=ET.SubElement(e,'visual');ET.SubElement(vis,'global',offwidth='1280',offheight='800')
    ET.SubElement(vis,'quality',shadowsize='4096');ET.SubElement(vis,'headlight',ambient='.35 .35 .35',diffuse='.7 .7 .7')
    base=e.find(".//body[@name='base']")
    if payload:
        inert=base.find('inertial');inert.set('mass',str(float(inert.get('mass'))+payload))
        # Added mass at the source COM, source body inertia unchanged: explicit
        # point-mass payload, not a separately articulated object.
    for leg in ['FL','FR','RL','RR']:
        ET.SubElement(e.find(f".//body[@name='{leg}_calf']"),'site',name=leg+'_toe',pos='-.002 0 -.213',size='.004',rgba='0 .8 1 1',group='5')
    w=e.find('worldbody')
    ET.SubElement(w,'light',pos='1 -1 3',dir='-.2 .2 -1',diffuse='.8 .8 .8')
    ET.SubElement(w,'light',pos='-1 2 2',dir='.3 -.3 -1',diffuse='.5 .6 .8')
    ET.SubElement(w,'geom',name='floor',type='plane',size='4 3 .02',rgba='.10 .14 .20 1',friction=f'{friction} .02 .01')
    if terrain:
        for k,(x,h) in enumerate(HURDLES):
            ET.SubElement(w,'geom',name=f'rail_{k}',type='box',pos=f'{x} 0 {h/2}',size=f'.025 .6 {h/2}',rgba='.95 .56 .10 1',friction=f'{friction} .02 .01')
        ET.SubElement(w,'geom',name='low_gate',type='box',pos='1.65 0 .38',size='.025 .575 .02',rgba='.20 .62 .70 1')
        for y in [-.55,.55]:
            ET.SubElement(w,'geom',name=f'gate_post_{y}',type='box',pos=f'1.65 {y} .18',size='.025 .025 .18',rgba='.20 .62 .70 1')
    # Painted landing zone and path marks are visual-only.
    ET.SubElement(w,'geom',name='dock',type='box',pos='2.20 .30 .0003',size='.34 .28 .0003',euler='0 0 .35',rgba='.08 .35 .38 1',contype='0',conaffinity='0')
    for x in np.arange(-.2,2.6,.15):
        ET.SubElement(w,'geom',type='box',pos=f'{x} -.64 .0005',size='.04 .007 .0005',rgba='.25 .6 .7 1',contype='0',conaffinity='0')
    path=root/'results'/f'{name}.xml';path.parent.mkdir(exist_ok=True);ET.indent(tree);tree.write(path,encoding='unicode')
    return path

class WholeBodyController:
    def __init__(self,cmg,friction=.5,feedforward=True):
        self.backend=PinBackend(cmg);self.kin=FootKinematics(cmg,cmg['q_reference'])
        self.armature=np.asarray(cmg['armature']);self.damping=np.asarray(cmg['damping']);self.friction=np.asarray(cmg['frictionloss'])
        self.limits=np.tile([23.7,23.7,45.43],4);self.mu=friction;self.feedforward=feedforward
        self.previous=None

    def command(self,q,v,qref,vref,aref,active,av,aa,stance):
        points,Jfeet=self.kin.points_and_jacobians(q);J=Jfeet.reshape(12,18)
        eps=1e-5
        Jp=self.kin.points_and_jacobians(q+eps*v)[1].reshape(12,18)
        Jm=self.kin.points_and_jacobians(q-eps*v)[1].reshape(12,18)
        jdv=(Jp-Jm)@v/(2*eps)
        M=self.backend.mass(q)+np.diag(self.armature);h=self.backend.bias(q,v)
        passive_comp=self.damping*v+self.friction*np.tanh(v/.02)
        desired_base=aa[:6]+np.array([100,100,160,120,120,120])*(active[:6]-q[:6])+np.array([20,20,25,22,22,22])*(av[:6]-v[:6])
        desired_feet=aa[6:]+160*(active[6:]-points.ravel())+25*(av[6:]-J@v)-jdv
        desired_joint=aref[6:]+60*(qref[6:]-q[6:])+12*(vref[6:]-v[6:])
        # Weighted acceleration tracking; soft foot tasks tolerate native
        # compliant/rolling contact without silently imposing a physical weld.
        L=np.zeros((36,30));target=np.r_[desired_base,desired_feet,desired_joint,np.zeros(6)]
        L[:6,:6]=np.eye(6);L[6:18,:18]=J;L[18:30,6:18]=np.eye(12)
        weights=np.r_[np.full(3,30.),np.full(3,20.),np.full(12,5.),np.full(12,.03),np.zeros(6)]
        P=L.T@(weights[:,None]*L)+np.diag(np.r_[np.full(18,1e-5),np.full(12,2e-5)])
        c=-L.T@(weights*target)
        dynamics=np.c_[M[:6],-J[:,:6].T]
        torquemap=np.c_[M[6:],-J[:,6:].T];offset=h[6:]+passive_comp[6:]
        constraints=[dynamics,torquemap];lower=[-h[:6],-self.limits-offset];upper=[-h[:6],self.limits-offset]
        for leg in range(4):
            rows=np.zeros((5,30));ix=18+3*leg;rows[0,ix+2]=1
            mu=self.mu/np.sqrt(2)
            rows[1,ix]=1;rows[1,ix+2]=-mu;rows[2,ix]=-1;rows[2,ix+2]=-mu
            rows[3,ix+1]=1;rows[3,ix+2]=-mu;rows[4,ix+1]=-1;rows[4,ix+2]=-mu
            constraints.append(rows);lower.append(np.array([0,-np.inf,-np.inf,-np.inf,-np.inf]))
            upper.append(np.array([180 if stance[leg] else 0,0,0,0,0]))
        A=np.vstack(constraints);lo=np.concatenate(lower);hi=np.concatenate(upper)
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(np.triu(P)),q=c,A=sparse.csc_matrix(A),l=lo,u=hi,
                                       verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=4000,polishing=True)
        if self.previous is not None:solver.warm_start(x=self.previous)
        answer=solver.solve(raise_error=False)
        if answer.info.status_val not in (1,2):raise RuntimeError('Whole-body QP: '+answer.info.status)
        self.previous=answer.x.copy();acc=answer.x[:18];forces=answer.x[18:].reshape(4,3)
        tau=torquemap@answer.x+offset
        if not self.feedforward:tau=60*(qref[6:]-q[6:])+3*(vref[6:]-v[6:])
        violation=max(float(np.max(lo-A@answer.x)),float(np.max(A@answer.x-hi)),0.)
        return np.clip(tau,-self.limits,self.limits),forces,violation,answer.info.iter

def run_case(root,name='nominal',dt=.001,friction=.8,payload=0.,push=1.,feedforward=True,terrain=True,duration=None,actuation=True):
    root=Path(root);cmg=load_model();ref=dict(np.load(root/'data/reference.npz'))
    interp=BPoly.from_derivatives(ref['time'],np.stack([ref['q'],ref['v'],ref['a']],axis=1))
    path=build_scene(root,name,dt,friction,payload,terrain);m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    d.qpos[:]=native_qpos_from_chart(ref['q'][0]);mujoco.mj_forward(m,d)
    ctrl=WholeBodyController(cmg,feedforward=feedforward);base=m.body('base').id
    footids=[m.geom(x).id for x in ['FL','FR','RL','RR']]
    siteids=[m.site(x+'_toe').id for x in ['FL','FR','RL','RR']]
    limits=np.tile([23.7,23.7,45.43],4);joints={j['id']:j for j in cmg['joints']}
    lower=np.array([joints[j]['limits']['lower'] for j in cmg['coordinate_ids'][6:]])
    upper=np.array([joints[j]['limits']['upper'] for j in cmg['coordinate_ids'][6:]])
    duration=float(ref['time'][-1]) if duration is None else duration
    control_every=round(.004/dt);log_every=round(.01/dt);steps=round(duration/dt)
    control_targets=target_trajectory(np.arange(steps//control_every+1)*control_every*dt)
    names=['time','qpos','qvel','q','v','q_ref','feet','foot_ref','stance','normal_force','support_force','predicted_force','torque','push','qp_violation','bad_contacts','body_error','angle_error']
    log={k:[] for k in names};bad=0;peakbad=0;maxqp=0.;minheight=1.;minmargin=1.;maxratio=0.;qpiter=0;early=False
    forces=np.zeros((4,3));violation=0.;contact_events=[];impulses=np.zeros((2,3))
    for k in range(steps+1):
        t=k*dt;q=chart_from_native(d.qpos);v=chart_velocity_from_native(q,d.qvel)
        if k%control_every==0:
            active,av,aa,stance=[arr[k//control_every] for arr in control_targets]
            qref=interp(t);vref=interp(t,nu=1);aref=interp(t,nu=2)
            d.ctrl[:],forces,violation,nit=ctrl.command(q,v,qref,vref,aref,active,av,aa,stance)
            if not actuation:d.ctrl[:]=0.
            maxqp=max(maxqp,violation);qpiter=max(qpiter,nit)
        pulse=np.zeros(6)
        if 1.2<=t<1.35:pulse[:3]=push*np.array([0.,32.,0.])
        if 24.3<=t<24.5:pulse[:3]=push*np.array([-25.,0.,0.])
        d.xfrc_applied[base]=pulse
        mujoco.mj_forward(m,d)
        normal=np.zeros(4);support_force=np.zeros(4);unexpected=0
        for ci in range(d.ncon):
            contact=d.contact[ci];g1,g2=contact.geom1,contact.geom2
            world1=m.geom_bodyid[g1]==0;world2=m.geom_bodyid[g2]==0
            if world1 or world2:
                moving=g2 if world1 else g1
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
                if moving in footids:
                    leg=footids.index(moving);normal[leg]+=f[0]
                    world_force=contact.frame.reshape(3,3).T@f[:3]
                    support_force[leg]+=(1 if moving==g2 else -1)*world_force[2]
                elif f[0]>1.:
                    unexpected+=1
                    if len(contact_events)<100:
                        contact_events.append(dict(time_s=t,body=m.body(m.geom_bodyid[moving]).name,obstacle=m.geom(g1 if world1 else g2).name,normal_force_N=float(f[0])))
            else:
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
                if f[0]>1.:
                    unexpected+=1
                    if len(contact_events)<100:contact_events.append(dict(time_s=t,body=m.body(m.geom_bodyid[g1]).name,obstacle=m.body(m.geom_bodyid[g2]).name,normal_force_N=float(f[0])))
        bad+=unexpected;peakbad=max(peakbad,unexpected)
        minheight=min(minheight,q[2]);minmargin=min(minmargin,float(np.min(np.minimum(q[6:]-lower,upper-q[6:]))))
        maxratio=max(maxratio,float(np.max(abs(d.ctrl)/limits)))
        if k%log_every==0:
            sample=round(t/.01);qr=ref['q'][sample];ar=ref['active'][sample]
            row=[t,d.qpos.copy(),d.qvel.copy(),q,v,qr,d.site_xpos[siteids].copy(),ar[6:].reshape(4,3),ref['stance'][sample],normal,support_force,forces,d.ctrl.copy(),pulse,violation,unexpected,np.linalg.norm(q[:3]-qr[:3]),np.linalg.norm(q[3:6]-qr[3:6])]
            for key,value in zip(names,row):log[key].append(value)
        if q[2]<.18 or np.linalg.norm(q[4:6])>.65:
            early=True;break
        if k<steps:
            impulses[int(t>=20)] += pulse[:3]*dt
            mujoco.mj_step(m,d)
    log={k:np.asarray(v) for k,v in log.items()};np.savez_compressed(root/'results'/f'{name}.npz',**log)
    final=q;target=interp(duration);moving=log['time']>=2
    metrics=dict(name=name,dt_s=dt,friction=friction,point_payload_kg=payload,push_scale=push,feedforward=feedforward,
                 actuation=actuation,completed=not early,simulated_s=float(d.time),final_position_error_m=float(np.linalg.norm(final[:3]-target[:3])),
                 final_yaw_error_rad=float(abs(final[3]-target[3])),min_base_height_m=minheight,max_tilt_rad=float(np.max(np.linalg.norm(log['q'][:,4:6],axis=1))),
                 rms_body_error_m=float(np.sqrt(np.mean(log['body_error'][moving]**2))) if np.any(moving) else 0.,
                 peak_body_error_m=float(np.max(log['body_error'])),unexpected_contact_instances=bad,peak_unexpected_contacts=peakbad,
                 min_joint_margin_rad=minmargin,peak_torque_limit_fraction=maxratio,max_qp_violation=maxqp,max_qp_iterations=qpiter,
                 final_speed_m_s=float(np.linalg.norm(v[:3])),measured_contact_modes=sorted(set(np.sum(log['support_force']>2,axis=1).tolist())),
                 external_push_impulse_Ns=impulses[0].tolist(),recovery_push_impulse_Ns=impulses[1].tolist(),native_nq=m.nq,native_nv=m.nv,motor_count=m.nu,
                 unexpected_contact_events=contact_events,numerical_warnings=sum(int(w.number) for w in d.warning))
    (root/'results'/f'{name}.json').write_text(json.dumps(metrics,indent=2)+'\n');return metrics
