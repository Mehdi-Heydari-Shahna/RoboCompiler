"""Physical 12-motor native runs and every-step mechanical energy ledger."""
from pathlib import Path
import json,time,sys
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from reconstructed_model import whole_body,UniversalMechanism,CutGraph,polish
from pacdm import PACDM
from whole_body_dynamics import FloatingSource,add_drive_terms
from native_model import export
from validate_whole_body import Checks
ROOT=Path(__file__).resolve().parent
DURATION=4.;REFERENCE_DT=.005


def requested(t):
    t=np.asarray(t);base=np.tile([0,0,0,.04,0,0],2);amp=np.tile([.003,.004,.004,.018,.003,.003],2);phase=np.r_[[0,.5,1,0,.5,-.5],[.3,1.3,-.5,1,.2,.7]]
    def ramp(x):
        y=np.clip(x,0,1);s=10*y**3-15*y**4+6*y**5;d=(30*y**2-60*y**3+30*y**4)*((x>0)&(x<1));dd=(60*y-180*y**2+120*y**3)*((x>0)&(x<1));return s,d,dd
    a,ad,add=ramp(t);b,bd,bdd=ramp(DURATION-t);r=a*b;rd=ad*b-a*bd;rdd=add*b-2*ad*bd+a*bdd
    angle=np.pi*t[...,None]+phase;sn=np.sin(angle);cs=np.cos(angle)
    return base+amp*r[...,None]*sn,amp*(rd[...,None]*sn+np.pi*r[...,None]*cs),amp*(rdd[...,None]*sn+2*np.pi*rd[...,None]*cs-np.pi**2*r[...,None]*sn)


def reference(c):
    ts=np.arange(round(DURATION/REFERENCE_DT)+1)*REFERENCE_DT;U,V,A=requested(ts);m=UniversalMechanism(c);g=CutGraph(c,np.array(c['initial_seed']));solver=PACDM(g)
    home=np.load(ROOT/'data/configurations.npz')['home'];Q,info=solver.track(U,home);Q=np.array([polish(g,x) for x in Q])
    if info['fallback_count']:raise RuntimeError('Reference needed reacquisition')
    forces=[];vel=[];energy=[];support=[];gap=0.;f=FloatingSource(c)
    for i,(qaug,uv,ua) in enumerate(zip(Q,V,A)):
        q=qaug[:76];s=m.state(q,uv,[0,0,-9.81]);s=add_drive_terms(s,c);F=m.inverse(s,ua);forces.append(F);vel.append(s['velocity']);energy.append([s['potential_energy'],s['kinetic_energy']]);gap=max(gap,max(abs(s['residual'])))
        # Full body root reaction when the physical support holds the pelvis fixed.
        ef=f.evaluate(q,np.zeros(3),np.eye(3),np.r_[np.zeros(6),s['velocity']],[0,0,-9.81]);acc=np.r_[np.zeros(6),s['tangent_map']@ua+s['curvature']];support.append((ef['mass_matrix']@acc+ef['bias_forces'])[:6])
        if i%160==0:print('  reference',i,'/',len(ts),flush=True)
    result=dict(t=ts,q=Q[:,:76],qaug=Q,qd=np.array(vel),u=U,ud=V,udd=A,force=np.array(forces),energy=np.array(energy),support_wrench=np.array(support))
    np.savez_compressed(ROOT/'data/motion_reference.npz',**result)
    (ROOT/'results/reference.json').write_text(json.dumps(dict(duration_s=DURATION,samples=len(ts),maximum_closure=gap,maximum_force_N=float(np.max(np.abs(forces))),pacdm_tracking=info),indent=2)+'\n');return result


class NativeMetrics:
    def __init__(self,c,m,d):
        self.c=c;self.m=m;self.d=d;self.a=[];self.b=[];self.ua=[];self.ub=[];self.uca=[];self.ucb=[]
        def sid(name):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,name)
        for cut in c['closures']:
            self.a.append(sid(cut['id']+'_a'));self.b.append(sid(cut['id']+'_b'))
            if cut['type']=='universal':
                self.ua.append(sid(cut['id']+'_ua'));self.ub.append(sid(cut['id']+'_ub'));self.uca.append(sid(cut['id']+'_a'));self.ucb.append(sid(cut['id']+'_b'))
    def closure(self):
        x=self.d.site_xpos;gap=np.max(np.linalg.norm(x[self.a]-x[self.b],axis=1));a=(x[self.ua]-x[self.uca])/.1;b=(x[self.ub]-x[self.ucb])/.1
        return gap,np.max(abs(np.sum(a*b,axis=1)))


def native_run(c,ref,dt,name,feedforward=True,no_loop=False,duration=DURATION):
    path=export(c,ROOT/'models'/f'{name}.xml',floating=False,drive=True,timestep=dt,solref=.0004)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in c['coordinate_ids']];qo=np.array([model.jnt_qposadr[x] for x in jids]);vo=np.array([model.jnt_dofadr[x] for x in jids]);active=np.array([c['coordinate_ids'].index(x) for x in c['independent_ids']]);motorv=vo[active]
    records={j['id']:j for j in c['joints']};lower=np.array([records[x]['limits']['lower'] for x in c['coordinate_ids']]);upper=np.array([records[x]['limits']['upper'] for x in c['coordinate_ids']]);slide=np.array([records[x]['type']=='prismatic' for x in c['coordinate_ids']])
    data.qpos[qo]=ref['q'][0];data.qvel[vo]=ref['qd'][0]
    if no_loop:data.eq_active[:]=0
    interpol=CubicSpline(ref['t'],ref['force'],axis=0);mujoco.mj_forward(model,data);metrics=NativeMetrics(c,model,data)
    # Per-port source force bounds (ctrlrange): +-2000 N, or +-5000 N for the leg-length ports.
    if not np.all(model.actuator_ctrllimited):raise ValueError('Every force port needs its source ctrlrange')
    ctrl_lo=model.actuator_ctrlrange[:,0].copy();ctrl_hi=model.actuator_ctrlrange[:,1].copy();force_limit=np.maximum(-ctrl_lo,ctrl_hi)
    steps=round(duration/dt);ts=np.arange(steps+1)*dt;U,V,_=requested(ts);FF=interpol(ts)
    # Every step: U,T, P_motor,P_damping,P_constraint,gap,angular, min slide margin,min hinge margin, F12,v_motor12,error12,saturation count,applied force norm.
    history=np.empty((steps+1,48));snap_t=[];snap_q=[];snap_v=[];snap_F=[];stride=max(1,round(.005/dt));warning0=np.array([w.number for w in data.warning]);start=time.time()
    for k in range(steps+1):
        q=data.qpos[qo];v=data.qvel[vo];error=U[k]-q[active];command=(FF[k] if feedforward else 0)+40000*error+600*(V[k]-v[active]);bounded=np.clip(command,ctrl_lo,ctrl_hi);data.ctrl[:]=bounded
        # Synchronize forces, energies and site poses to this actual state before logging.
        mujoco.mj_forward(model,data);gap,angular=metrics.closure();margin=np.minimum(q-lower,upper-q)
        row=np.r_[data.energy,data.qfrc_actuator@data.qvel,data.qfrc_passive@data.qvel,data.qfrc_constraint@data.qvel,gap,angular,np.min(margin[slide]),np.min(margin[~slide]),data.actuator_force,v[active],error,np.sum(command!=bounded),np.max(abs(data.qfrc_applied)),np.max(abs(data.qfrc_constraint))]
        history[k]=row
        if k%stride==0:snap_t.append(data.time);snap_q.append(q.copy());snap_v.append(v.copy());snap_F.append(data.actuator_force.copy())
        if k<steps:mujoco.mj_step(model,data)
    power=history[:,2]+history[:,3]+history[:,4];work=np.r_[0,np.cumsum((power[1:]+power[:-1])*.5*dt)];E=history[:,:2].sum(axis=1);balance=E-E[0]-work
    def workof(col):return float(np.trapezoid(history[:,col],ts))
    summary=dict(name=name,timestep_s=dt,duration_s=duration,steps=steps,maximum_point_gap_m=float(max(history[:,5])),maximum_universal_dot=float(max(history[:,6])),minimum_prismatic_margin_m=float(min(history[:,7])),minimum_revolute_margin_rad=float(min(history[:,8])),maximum_motor_force_N=float(np.max(abs(history[:,9:21]))),maximum_motor_force_fraction=float(np.max(abs(history[:,9:21])/force_limit)),tracking_rms_m=float(np.sqrt(np.mean(history[:,33:45]**2))),tracking_max_m=float(np.max(abs(history[:,33:45]))),maximum_mechanical_power_W=float(np.max(abs(history[:,2]))),motor_work_J=workof(2),damping_work_J=workof(3),constraint_work_J=workof(4),energy_change_J=float(E[-1]-E[0]),energy_ledger_max_error_J=float(max(abs(balance))),energy_ledger_final_error_J=float(balance[-1]),saturation_samples=int(sum(history[:,45])),maximum_generalized_applied_force=float(max(history[:,46])),warning_count=int(sum(np.array([w.number for w in data.warning])-warning0)),elapsed_seconds=time.time()-start,control='source force motors with PACDM inverse feedforward + PD' if feedforward else 'same-gain PD only',constraint_model='disabled negative control' if no_loop else 'connect + universal orthogonality tendon')
    np.savez_compressed(ROOT/'results'/f'{name}.npz',time=ts,history=history,balance=balance,sample_t=snap_t,q=snap_q,qd=snap_v,force=snap_F)
    (ROOT/'results'/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n');print(name,summary,flush=True);return summary


def run(ref=None):
    c=whole_body();checks=Checks();ref=reference(c) if ref is None else ref;runs=[]
    for dt in [.0002,.0001,.00005]:runs.append(native_run(c,ref,dt,f'motion_dt_{dt:g}'))
    pd=native_run(c,ref,.0001,'motion_pd_only',feedforward=False)
    neg=native_run(c,ref,.0001,'negative_control_no_loops',no_loop=True,duration=.3)
    for r in runs:
        n=r['name']+'_';checks.check(n+'point_closure',r['maximum_point_gap_m'],2e-4);checks.check(n+'universal_angular',r['maximum_universal_dot'],.002);checks.check(n+'tracking_rms',r['tracking_rms_m'],.001);checks.check(n+'force_fraction',r['maximum_motor_force_fraction'],1.);checks.check(n+'energy_ledger',r['energy_ledger_max_error_J'],.02);checks.flag(n+'limits',r['minimum_prismatic_margin_m']>=0 and r['minimum_revolute_margin_rad']>=0);checks.flag(n+'no_saturation_warnings_external_tree_force',r['saturation_samples']==0 and r['warning_count']==0 and r['maximum_generalized_applied_force']==0)
    checks.flag('time_step_refinement_energy',runs[-1]['energy_ledger_max_error_J']<runs[0]['energy_ledger_max_error_J'])
    checks.flag('negative_control_detects_missing_loops',neg['maximum_point_gap_m']>.01)
    checks.flag('every_actuator_moves',bool(np.all(np.ptp(ref['u'],axis=0)>.004)))
    report=checks.save(ROOT/'results/motion_summary.json',runs=runs,pd_control=pd,negative_control=neg,energy_scope='Mechanical energy, motor mechanical work, viscous loss and compliant constraint work; no electrical efficiency or battery model.')
    print(report['status'],sum(x['passed'] for x in checks.rows),'/',len(checks.rows),flush=True);return report

if __name__=='__main__':sys.exit(0 if run()['status']=='PASS_RECONSTRUCTED_MODEL' else 1)
