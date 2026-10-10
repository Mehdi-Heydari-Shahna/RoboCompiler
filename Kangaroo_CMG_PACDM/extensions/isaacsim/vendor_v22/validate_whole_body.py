"""Reproducible whole-body static and instantaneous benchmark, with strict gates."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np
from scipy.spatial.transform import Rotation
from reconstructed_model import whole_body,UniversalMechanism,CutGraph,polish
from pacdm import PACDM,exp
from whole_body_dynamics import FloatingSource,PinFloating,MJFloating,add_drive_terms
from native_model import export
from constraint_solvers import solve_reduced,solve_kkt
from urdf_roundtrip import write as write_urdf,read as read_urdf
ROOT=Path(__file__).resolve().parent

class Checks:
    def __init__(self):self.rows=[]
    def check(self,name,value,limit):
        passed=bool(np.isfinite(value) and value<=limit);self.rows.append(dict(name=name,passed=passed,value=float(value),limit=float(limit)))
        if not passed:print('FAIL',self.rows[-1],flush=True)
    def flag(self,name,passed):self.rows.append(dict(name=name,passed=bool(passed)))
    def save(self,path,**metadata):
        obj=dict(status='PASS_RECONSTRUCTED_MODEL' if all(x['passed'] for x in self.rows) else 'FAIL',gates=self.rows,**metadata)
        Path(path).write_text(json.dumps(obj,indent=2)+'\n');return obj

def err(a,b):return float(np.max(abs(np.asarray(a)-b)))

def run():
    t0=time.time();checks=Checks();c=whole_body();m=UniversalMechanism(c);f=FloatingSource(c);g=CutGraph(c,np.array(c['initial_seed']));p=PACDM(g)
    (ROOT/'data/whole_body_cmg.json').write_text(json.dumps(c,indent=2)+'\n')
    checks.flag('source_to_cmg_deterministic_rebuild',c==whole_body())
    write_urdf(c,ROOT/'models/kangaroo_full.urdf');rebuilt=read_urdf(ROOT/'models/kangaroo_full.urdf')
    (ROOT/'data/urdf_rebuilt_cmg.json').write_text(json.dumps(rebuilt,indent=2)+'\n')
    original={j['id']:j for j in c['joints']};roundtrip=max(err(j['T_BJ'],original[j['id']]['T_BJ']) for j in rebuilt['joints'])
    checks.check('urdf_to_cmg_joint_frame_roundtrip',roundtrip,1e-12)
    checks.check('urdf_to_cmg_inertia_roundtrip',max(err(b['inertia_com_kg_m2'],a['inertia_com_kg_m2']) for a,b in zip(c['bodies'],rebuilt['bodies'])),1e-15)
    checks.flag('urdf_to_cmg_bodies_loops_actuators_order_and_limits',all({k:v for k,v in a.items() if k!='inertia_com_kg_m2'}=={k:v for k,v in b.items() if k!='inertia_com_kg_m2'} for a,b in zip(c['bodies'],rebuilt['bodies'])) and rebuilt['closures']==c['closures'] and rebuilt['actuators']==c['actuators'] and rebuilt['coordinate_ids']==c['coordinate_ids'] and all(j.get('limits')==original[j['id']].get('limits') for j in rebuilt['joints']))
    checks.flag('all_78_source_rigid_bodies_76_internal_coordinates_12_force_ports',len(c['bodies'])==78 and m.n==76 and len(c['actuators'])==12)
    base=np.tile([0,0,0,.04,0,0],2);width=np.tile([.003,.004,.004,.012,.003,.003],2)
    rng=np.random.default_rng(202603);U=np.vstack([base,base+np.diag(width),base-np.diag(width),base+rng.uniform(-1,1,(12,12))*width])
    print('Acquiring full-body PACDM and checking',len(U),'bounded configurations...',flush=True)
    q,acq=p.acquire(base,g.lift(np.array(c['initial_seed'])));checks.flag('pacdm_full_body_acquisition',acq['success'])
    if not acq['success']:raise RuntimeError(acq)
    Q,track=p.track(U,q);Q=np.array([polish(g,x) for x in Q]);q=polish(g,q);np.savez_compressed(ROOT/'data/configurations.npz',home=q,Q=Q,U=U)
    checks.flag('pacdm_37_states_no_fallback',track['fallback_count']==0)
    maxima={k:0. for k in ['closure','se3_residual','map_difference','graph_fd','fixed_defect_fd','closure_fd','jdot_fd','floating_jdot_fd','mass_pin','mass_mj','bias_pin','bias_mj','potential_pin','potential_mj','kinetic_pin','kinetic_mj','poses_pin','poses_mj','jacobians_pin','jacobians_mj','native_residual','native_tangent','energy_rate','reaction_power','kkt_acceleration','kkt_reaction','forward_pin','forward_mj','fixed_inverse','fixed_support_balance','motor_force']}
    def update(k,value):maxima[k]=max(maxima[k],float(value))
    pi=PinFloating(c);mj=MJFloating(c,export(c,ROOT/'models/floating_tree.xml'))
    # Off-manifold derivatives, including the fixed-defect SE(3) transport.
    off=Q[5]+rng.normal(0,.001,g.n);direction=rng.normal(0,1,g.n);eps=1e-7
    for key,defect in [('graph_fd',None),('fixed_defect_fd',np.array([exp(rng.normal(0,.005,6)) for _ in c['closures']]))]:
        r,J,_=g.residual(off,defect);fd=(g.residual(off+eps*direction,defect)[0]-g.residual(off-eps*direction,defect)[0])/(2*eps);update(key,err(fd,J@direction))
    body_total_mass=sum(b['mass_kg'] for b in c['bodies']);margin=float('inf');case_count=0
    force_cases=[];reaction_cases=[];acceleration_cases=[];fixed_forces=[];fixed_support=[];states=[]
    for i,aug in enumerate(Q):
        q=aug[:m.n];N,info=p.mapping(aug);checks.flag(f'pacdm_rank_state_{i}',info['success'] and info['rank_full']==128)
        update('se3_residual',info['residual_inf']);margin=min(margin,float(np.min(np.minimum(q-m.lower,m.upper-q))))
        ud=rng.uniform(-.008,.008,12);s=m.state(q,ud,[0,0,-9.81]);update('closure',max(abs(s['residual'])));update('map_difference',err(N[:76],s['tangent_map']))
        if i in [0,5,24]:
            v=s['velocity'];e=m.source.evaluate(q+rng.normal(0,.001,76),v,[0,0,-9.81]);qq=e['configuration'];r,J,D=m.closure(e)
            ep=m.source.evaluate(qq+eps*v,v,[0,0,-9.81]);em=m.source.evaluate(qq-eps*v,v,[0,0,-9.81]);rp,Jp,_=m.closure(ep);rm,Jm,_=m.closure(em)
            update('closure_fd',err((rp-rm)/(2*eps),J@v));update('jdot_fd',err((Jp-Jm)/(2*eps),D))
        for gravity in [[0,0,-9.81],[1.1,-.7,-8.9],[0,0,0]]:
            position=rng.uniform(-.2,.2,3)+[0,0,1.1];R=Rotation.from_rotvec(rng.uniform(-.3,.3,3)).as_matrix();uv=np.r_[rng.uniform(-.2,.2,6),ud]
            sf=f.state(q,position,R,uv,gravity);v=sf['velocity'];a_f=add_drive_terms(sf,c)
            engines={'source':sf,'pin':pi.evaluate(q,position,R,v,gravity),'mj':mj.evaluate(q,position,R,v,gravity)}
            for label in ['pin','mj']:
                e=engines[label]
                for key,name in [('mass_matrix','mass'),('bias_forces','bias'),('potential_energy','potential'),('kinetic_energy','kinetic')]:update(name+'_'+label,err(e[key],sf[key]))
                for name in sf['poses']:
                    update('poses_'+label,err(e['poses'][name],sf['poses'][name]));update('jacobians_'+label,err(e['jacobians'][name],sf['jacobians'][name]))
            update('native_residual',max(abs(mj.data.efc_pos)))
            nativeJ=mj.data.efc_J.reshape(-1,mj.model.nv)[:,mj.order];T=np.eye(f.n);T[3:6,3:6]=R.T
            update('native_tangent',np.max(abs(nativeJ@T@sf['tangent_map'])))
            for _ in range(2):
                force=rng.uniform(-100,100,12);tau=f.B@force;rr=solve_reduced(a_f,tau);kk=solve_kkt(a_f,tau)
                force_cases.append(force);reaction_cases.append(rr['generalized_reaction']);acceleration_cases.append(rr['acceleration'])
                states.append(np.r_[i,gravity,position,Rotation.from_matrix(R).as_quat(),v])
                update('kkt_acceleration',err(rr['acceleration'],kk['acceleration']));update('kkt_reaction',err(rr['generalized_reaction'],kk['generalized_reaction']));update('reaction_power',abs(v@rr['generalized_reaction']))
                for label in ['pin','mj']:
                    e=add_drive_terms({**sf,**engines[label]},c);out=solve_reduced(e,tau);update('forward_'+label,err(rr['acceleration'],out['acceleration']))
                case_count+=1
            # Independent energy directional derivative, with changing pose and v.
            if i in [0,5,24]:
                acceleration=rng.uniform(-.1,.1,f.n)
                def energy(sign):
                    dt=sign*1e-6;e=f.evaluate(q+dt*v[6:],position+dt*v[:3],Rotation.from_rotvec(dt*v[3:6]).as_matrix()@R,v+dt*acceleration,gravity)
                    return e['potential_energy']+e['kinetic_energy']
                rate=(energy(1)-energy(-1))/(2e-6);update('energy_rate',abs(rate-v@(sf['mass_matrix']@acceleration+sf['bias_forces'])))
                ep=f.evaluate(q+eps*v[6:],position+eps*v[:3],Rotation.from_rotvec(eps*v[3:6]).as_matrix()@R,v,gravity)
                em=f.evaluate(q-eps*v[6:],position-eps*v[:3],Rotation.from_rotvec(-eps*v[3:6]).as_matrix()@R,v,gravity)
                for name in sf['poses']:update('floating_jdot_fd',err((ep['jacobians'][name]-em['jacobians'][name])/(2*eps),sf['body_jacobian_dots'][name]))
            # Fixed-pelvis inverse dynamics and the separate unactuated support wrench.
            sfixed=f.state(q,position,R,np.r_[np.zeros(6),ud],gravity);ef=add_drive_terms(sfixed,c);Nf=ef['tangent_map'][:,6:];udd=rng.uniform(-.05,.05,12);acc=Nf@udd+ef['curvature'];load=ef['mass_matrix']@acc+ef['bias_forces'];force=np.linalg.solve(Nf.T@f.B,Nf.T@load)
            update('motor_force',max(abs(force)));fixed={**ef,'mass_matrix':ef['mass_matrix'][6:,6:],'bias_forces':ef['bias_forces'][6:],'tangent_map':Nf[6:],'curvature':ef['curvature'][6:]}
            recovered=solve_reduced(fixed,m.B@force);update('fixed_inverse',err(recovered['acceleration'],acc[6:]));reaction=ef['jacobian'][:,6:].T@np.linalg.lstsq(ef['jacobian'][:,6:].T,load[6:]-m.B@force,rcond=1e-10)[0];update('fixed_support_balance',err(reaction,load[6:]-m.B@force))
            fixed_forces.append(force);fixed_support.append(load[:6])
        if i%8==0:print('  validated configuration',i+1,'/',len(Q),flush=True)
    limits={'closure':1e-8,'se3_residual':1e-8,'map_difference':2e-6,'graph_fd':1e-7,'fixed_defect_fd':1e-7,'closure_fd':1e-7,'jdot_fd':1e-7,'floating_jdot_fd':1e-7,'mass_pin':1e-10,'mass_mj':1e-7,'bias_pin':1e-9,'bias_mj':1e-6,'potential_pin':1e-9,'potential_mj':1e-6,'kinetic_pin':1e-10,'kinetic_mj':1e-7,'poses_pin':1e-10,'poses_mj':1e-10,'jacobians_pin':1e-10,'jacobians_mj':1e-10,'native_residual':1e-8,'native_tangent':1e-7,'energy_rate':1e-6,'reaction_power':1e-7,'kkt_acceleration':1e-5,'kkt_reaction':1e-6,'forward_pin':1e-5,'forward_mj':1e-3,'fixed_inverse':1e-7,'fixed_support_balance':1e-7,'motor_force':5000}
    for key,limit in limits.items():checks.check(key,maxima[key],limit)
    checks.flag('all_source_joint_limits',margin>=0);checks.flag('source_force_ports_prismatic',all(next(j for j in c['joints'] if j['id']==a['joint'])['type']=='prismatic' for a in c['actuators']))
    checks.check('total_mass_against_body_sum',abs(body_total_mass-sf['total_mass']),1e-10)
    np.savez_compressed(ROOT/'results/force_cases.npz',case_state=states,motor_force=force_cases,generalized_loop_reaction=reaction_cases,acceleration=acceleration_cases,fixed_base_motor_force=fixed_forces,fixed_base_support_wrench=fixed_support)
    report=checks.save(ROOT/'results/instantaneous.json',model_scope='Explicit reconstruction, not author-verified hardware model',maxima=maxima,configurations=len(Q),gravity_vectors=3,floating_force_cases=case_count,total_mass_kg=body_total_mass,minimum_joint_margin_mixed_units=margin,elapsed_seconds=time.time()-t0,acquisition=acq,tracking=track)
    print(report['status'],sum(x['passed'] for x in checks.rows),'/',len(checks.rows),'seconds',time.time()-t0,flush=True)
    return report

if __name__=='__main__':sys.exit(0 if run()['status']=='PASS_RECONSTRUCTED_MODEL' else 1)
