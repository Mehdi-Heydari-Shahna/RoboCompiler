"""Native free-body conservation and ground-supported full-body validation."""
from pathlib import Path
import json,sys,time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from whole_body_dynamics import FloatingSource
from reconstructed_model import whole_body
from native_model import export
from whole_body_motion import NativeMetrics
from validate_whole_body import Checks
ROOT=Path(__file__).resolve().parent

def setup(c,name,dt,drive,contact):
    path=export(c,ROOT/'models'/f'{name}.xml',floating=True,drive=drive,contact=contact,timestep=dt,solref=.0001 if contact else .001)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);j=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in c['coordinate_ids']]
    qo=np.array([model.jnt_qposadr[x] for x in j]);vo=np.array([model.jnt_dofadr[x] for x in j]);return model,data,qo,vo,NativeMetrics(c,model,data)

def coast(c,dt):
    name=f'free_coast_{dt:g}';m,d,qo,vo,metrics=setup(c,name,dt,False,False);f=FloatingSource(c);q=np.load(ROOT/'data/configurations.npz')['home'][:76]
    ud=np.r_[[.10,-.03,.05,.02,.03,-.01],np.array([.001,.001,-.001,.002,.001,-.001]*2)]
    s=f.state(q,np.array([0,0,1.2]),np.eye(3),ud,[0,0,0]);d.qpos[:3]=[0,0,1.2];d.qpos[3:7]=[1,0,0,0];d.qpos[qo]=q;d.qvel[:6]=s['velocity'][:6];d.qvel[vo]=s['velocity'][6:];m.opt.gravity[:]=0
    mujoco.mj_forward(m,d);steps=round(.4/dt);H=[];Q=[];V=[];W=0.;lastpower=None;E0=d.energy.sum();moment0=np.r_[s['linear_momentum'],s['angular_momentum']];moment_error=0.;warnings0=sum(w.number for w in d.warning)
    for i in range(steps+1):
        mujoco.mj_forward(m,d);power=d.qfrc_constraint@d.qvel
        if lastpower is not None:W+=.5*dt*(lastpower+power)
        lastpower=power;gap,angle=metrics.closure();H.append([d.time,*d.energy,power,gap,angle,d.energy.sum()-E0-W])
        if i%max(1,round(.004/dt))==0:
            R=Rotation.from_quat(d.qpos[3:7][[1,2,3,0]]).as_matrix();v=np.r_[d.qvel[:3],R@d.qvel[3:6],d.qvel[vo]];e=f.evaluate(d.qpos[qo],d.qpos[:3],R,v,[0,0,0]);moment_error=max(moment_error,float(np.max(abs(np.r_[e['linear_momentum'],e['angular_momentum']]-moment0))));Q.append(d.qpos.copy());V.append(d.qvel.copy())
        if i<steps:mujoco.mj_step(m,d)
    H=np.array(H);r=dict(timestep_s=dt,duration_s=.4,maximum_point_gap_m=float(max(H[:,4])),maximum_universal_dot=float(max(H[:,5])),initial_kinetic_energy_J=float(E0),maximum_energy_drift_J=float(np.max(abs(H[:,1:3].sum(axis=1)-E0))),maximum_energy_ledger_error_J=float(np.max(abs(H[:,6]))),maximum_momentum_error_mixed_SI=moment_error,maximum_force_N=float(max(abs(d.actuator_force))),warning_count=sum(w.number for w in d.warning)-warnings0)
    np.savez_compressed(ROOT/'results'/f'{name}.npz',history=H,q=Q,v=V);print(name,r,flush=True);return r

def standing(c,dt=.0001,duration=2.):
    name=f'standing_{dt:g}';m,d,qo,vo,metrics=setup(c,name,dt,True,True);f=FloatingSource(c);q=np.load(ROOT/'data/configurations.npz')['home'][:76];s=f.state(q,np.zeros(3),np.eye(3),np.zeros(18),[0,0,-9.81]);active=np.array(f.internal.active);corners=[];jac=[]
    # Use all eight source foot-box bottom corners and a nonnegative vertical
    # load distribution matching total mass and horizontal COM. No root actuator.
    for side in ['left','right']:
        body=side+'_ankle_roll';T=s['poses'][body];Rg=Rotation.from_quat([-.173648,0,0,.984808]).as_matrix()
        for a in [-.045,.045]:
            for b in [-.105,.105]:
                local=np.array([0,-.005,.04])+Rg@np.array([a,-.0125,b]);p,J,_=f.internal.source._point(s,body,local);corners.append(p);jac.append(J)
    corners=np.array(corners);mass=s['total_mass'];A=np.vstack([np.ones(8),corners[:,0],corners[:,1]]);target=mass*9.81*np.r_[1,s['center_of_mass'][:2]];normal=A.T@np.linalg.solve(A@A.T,target)
    if np.min(normal)<=0:raise RuntimeError('COM outside chosen vertical support distribution')
    load=sum(J.T@np.array([0,0,F]) for J,F in zip(jac,normal));red=s['tangent_map'];effort=np.linalg.solve(red[:,6:].T@f.B,red[:,6:].T@(s['bias_forces']-load))
    baseheight=-min(corners[:,2])+.00005;d.qpos[:3]=[0,0,baseheight];d.qpos[3:7]=[1,0,0,0];d.qpos[qo]=q;mujoco.mj_forward(m,d);steps=round(duration/dt);H=[];states=[];times=[];forces=[];maxgap=0.;maxangle=0.;saturation=0;minmargin=float('inf');records={j['id']:j for j in c['joints']};lo=np.array([records[x]['limits']['lower'] for x in c['coordinate_ids']]);hi=np.array([records[x]['limits']['upper'] for x in c['coordinate_ids']]);warning0=sum(w.number for w in d.warning);work=0;lastpower=None;E0=d.energy.sum();start=time.time()
    for i in range(steps+1):
        command=effort+300000*(q[active]-d.qpos[qo[active]])+4000*(0-d.qvel[vo[active]]);d.ctrl[:]=np.clip(command,-5000,5000);saturation+=int(np.any(abs(command)>5000));mujoco.mj_forward(m,d);gap,angle=metrics.closure();maxgap=max(maxgap,gap);maxangle=max(maxangle,angle);minmargin=min(minmargin,float(np.min(np.minimum(d.qpos[qo]-lo,hi-d.qpos[qo]))))
        contact=np.zeros(3)
        for k in range(d.ncon):
            con=d.contact[k];force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force);world=con.frame.reshape(3,3).T@force[:3]
            # The floor is geom 0; MuJoCo's contact force acts on geom2.
            if con.geom1==0:contact+=world
            elif con.geom2==0:contact-=world
        Pm=d.qfrc_actuator@d.qvel;Pd=d.qfrc_passive@d.qvel;Pc=d.qfrc_constraint@d.qvel;power=Pm+Pd+Pc
        if lastpower is not None:work+=.5*dt*(power+lastpower)
        lastpower=power
        H.append([d.time,*d.energy,Pm,Pd,Pc,*contact,gap,angle,np.max(abs(d.qpos[qo[active]]-q[active])),np.max(abs(d.actuator_force)),d.energy.sum()-E0-work,*d.qpos[:3],*d.qvel[:6]])
        if i%max(1,round(.005/dt))==0:states.append(d.qpos.copy());times.append(d.time);forces.append(d.actuator_force.copy())
        if i<steps:mujoco.mj_step(m,d)
    H=np.array(H);tail=H[:,0]>duration-.2;r=dict(duration_s=duration,timestep_s=dt,total_mass_kg=mass,weight_N=mass*9.81,static_motor_force_N=effort.tolist(),static_corner_normal_force_N=normal.tolist(),static_base_equilibrium_error=float(max(abs((s['bias_forces']-load)[:6]))),maximum_point_gap_m=maxgap,maximum_universal_dot=maxangle,minimum_joint_margin_mixed_SI=minmargin,maximum_motor_force_N=float(max(H[:,12])),maximum_motor_position_error_m=float(max(H[:,11])),mean_final_ground_force_N=H[tail,6:9].mean(axis=0).tolist(),maximum_final_root_speed_m_s=float(np.max(np.linalg.norm(H[tail,17:20],axis=1))),maximum_root_translation_m=float(np.max(np.linalg.norm(H[:,14:17]-H[0,14:17],axis=1))),maximum_energy_ledger_error_J=float(max(abs(H[:,13]))),saturation_samples=saturation,warning_count=sum(w.number for w in d.warning)-warning0,maximum_generalized_applied_force=float(max(abs(d.qfrc_applied))),elapsed_seconds=time.time()-start)
    np.savez_compressed(ROOT/'results'/f'{name}.npz',history=H,sample_t=times,q=states,force=forces);(ROOT/'results'/f'{name}.json').write_text(json.dumps(r,indent=2)+'\n');print(name,r,flush=True);return r

def run():
    c=whole_body();checks=Checks();coasts=[coast(c,dt) for dt in [.0002,.0001]];stands=[standing(c,dt) for dt in [.00005,.000025]]
    for i,r in enumerate(coasts):
        for key,lim in [('maximum_point_gap_m',1e-5),('maximum_universal_dot',1e-4),('maximum_energy_drift_J',.001),('maximum_energy_ledger_error_J',.001),('maximum_momentum_error_mixed_SI',.001)]:checks.check(f'coast_{i}_'+key,r[key],lim)
        checks.flag(f'coast_{i}_unactuated_no_warnings',r['maximum_force_N']==0 and r['warning_count']==0)
    for i,r in enumerate(stands):
        for key,lim in [('static_base_equilibrium_error',1e-8),('maximum_point_gap_m',.0002),('maximum_universal_dot',.002),('maximum_motor_force_N',5000),('maximum_motor_position_error_m',.001),('maximum_final_root_speed_m_s',.01),('maximum_root_translation_m',.02),('maximum_energy_ledger_error_J',.03)]:checks.check(f'standing_{i}_'+key,r[key],lim)
        checks.check(f'standing_{i}_vertical_force_balance_N',abs(r['mean_final_ground_force_N'][2]-r['weight_N']),2)
        checks.flag(f'standing_{i}_limits_no_saturation_warnings_external_force',r['minimum_joint_margin_mixed_SI']>=0 and r['saturation_samples']==0 and r['warning_count']==0 and r['maximum_generalized_applied_force']==0)
    checks.flag('coast_energy_refines',coasts[-1]['maximum_energy_drift_J']<coasts[0]['maximum_energy_drift_J']);checks.flag('standing_energy_refines',stands[-1]['maximum_energy_ledger_error_J']<stands[0]['maximum_energy_ledger_error_J'])
    r=checks.save(ROOT/'results/floating_motion_summary.json',coast=coasts,standing=stands);print(r['status'],sum(x['passed'] for x in checks.rows),'/',len(checks.rows),flush=True);return r

if __name__=='__main__':sys.exit(0 if run()['status']=='PASS_RECONSTRUCTED_MODEL' else 1)
