"""Motor-only Pinocchio rollout with an independent compliant Coulomb solver.

No MuJoCo imports or calls. CRBA/RNEA, frame Jacobians and integration are
Pinocchio. Contact forces are solved from the plant state, not QP predictions.
Only sphere feet resolve world contacts; forbidden body contacts are audited.
"""
from pathlib import Path
import json, time, sys
from copy import deepcopy
import numpy as np
import pinocchio as pin
from scipy.interpolate import BPoly
from .model import load_model
from .pin_backend import PinBackend
from .pin_controller import WholeBodyController
from .task import target_trajectory, HURDLES

RADIUS=.022
KN=50000.0
DN=100.0
KT=10000.0
DTAN=60.0


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

class Plant:
    def __init__(self, cmg, payload=0.):
        physical=deepcopy(cmg)
        if payload:
            next(b for b in physical['bodies'] if b['id']=='base')['mass_kg']+=payload
        self.backend=PinBackend(physical)
        self.b=self.backend
        self.feet=cmg['feet']
        self.armature=np.asarray(cmg['armature'])
        self.damping=np.asarray(cmg['damping'])
        self.dry=np.asarray(cmg['frictionloss'])
        self.base_com=np.asarray(next(b for b in cmg['bodies'] if b['id']=='base')['com_m'])
        self.history={}

    def kinematics(self,q):
        b=self.b
        pin.computeJointJacobians(b.model,b.data,b._native_q(q))
        pin.updateFramePlacements(b.model,b.data)
        points=[];linear=[];angular=[]
        for f in self.feet:
            fid=b.body_frame_ids[f['body']]
            pose=b.data.oMf[fid]
            J=np.asarray(pin.getFrameJacobian(b.model,b.data,fid,pin.LOCAL_WORLD_ALIGNED))[:,b._v_indices]
            offset=pose.rotation@np.asarray(f['point_m'])
            points.append(pose.translation+offset)
            linear.append(J[:3]-skew(offset)@J[3:])
            angular.append(J[3:].copy())
        fid=b.body_frame_ids['base'];pose=b.data.oMf[fid]
        J=np.asarray(pin.getFrameJacobian(b.model,b.data,fid,pin.LOCAL_WORLD_ALIGNED))[:,b._v_indices]
        Jpush=J[:3]-skew(pose.rotation@self.base_com)@J[3:]
        return np.array(points),np.array(linear),np.array(angular),Jpush

    def contacts(self, points, linear, angular):
        contacts=[]
        for leg,p in enumerate(points):
            candidates=[(p[2]-RADIUS,np.array([0.,0.,1.]),'floor')]
            for ri,(x,h) in enumerate(HURDLES):
                lo=np.array([x-.025,-.6,0.]);hi=np.array([x+.025,.6,h])
                closest=np.clip(p,lo,hi);delta=p-closest;dist=np.linalg.norm(delta)
                if dist>1e-12:
                    candidates.append((dist-RADIUS,delta/dist,f'rail{ri}'))
                else:
                    ds=np.r_[p-lo,hi-p];axis=int(np.argmin(ds));n=np.zeros(3);n[axis%3]=-1 if axis<3 else 1
                    candidates.append((-float(np.min(ds))-RADIUS,n,f'rail{ri}'))
            gap,n,surface=min(candidates,key=lambda v:v[0])
            if gap>0.:continue
            axis=np.array([1.,0.,0.]) if abs(n[0])<.8 else np.array([0.,1.,0.])
            t1=axis-n*np.dot(n,axis);t1/=np.linalg.norm(t1);t2=np.cross(n,t1)
            basis=np.stack([t1,t2,n])
            J=basis@(linear[leg]-skew(-RADIUS*n)@angular[leg])
            key=(leg,surface)
            # History is accumulated contact-point tangential slip, in world axes.
            xi=basis[:2]@self.history.get(key,np.zeros(3))
            contacts.append((leg,key,float(gap),basis,J,xi))
        return contacts

    def step(self,q,v,tau,push,dt,mu):
        M=self.b.mass(q)+np.diag(self.armature);h=self.b.bias(q,v)
        points,lin,ang,Jpush=self.kinematics(q)
        contacts=self.contacts(points,lin,ang)
        dry=self.dry*np.tanh(v/.02)
        applied=np.r_[np.zeros(6),tau]+Jpush.T@push
        H=M+dt*np.diag(self.damping)
        vfree=np.linalg.solve(H,M@v+dt*(applied-h-dry))
        forces=np.zeros((4,3));normals=np.zeros(4);min_gap=min([c[2] for c in contacts]+[0.])
        p=np.zeros(3*len(contacts));residual=0.;nit=0;law_residual=0.;cone_excess=0.;minimum_normal=0.
        if contacts:
            J=np.vstack([c[4] for c in contacts]);Wmap=np.linalg.solve(H,J.T);W=J@Wmap
            dn=DN+dt*KN;dtan=DTAN+dt*KT
            compliance=np.tile([1/(dt*dtan),1/(dt*dtan),1/(dt*dn)],len(contacts))
            A=W+np.diag(compliance)
            targets=np.concatenate([np.r_[KT*c[5]/dtan,KN*c[2]/dn] for c in contacts])
            b=J@vfree+targets
            # Block projected Gauss-Seidel: unilateral normal spring/damper,
            # circular Coulomb disk, tangential spring with plastic slip return.
            for nit in range(1,121):
                old=p.copy()
                for i in range(len(contacts)):
                    j=3*i
                    p[j+2]=max(0.,p[j+2]-(A[j+2]@p+b[j+2])/A[j+2,j+2])
                    block=A[j:j+2,j:j+2]
                    alpha=1./np.linalg.eigvalsh(block)[-1]
                    pt=p[j:j+2]-alpha*(A[j:j+2]@p+b[j:j+2])
                    cap=mu*p[j+2];length=np.linalg.norm(pt)
                    p[j:j+2]=pt*min(1.,cap/max(length,1e-30))
                residual=float(np.max(np.abs(p-old)))
                if residual<1e-10:break
            vn=vfree+Wmap@p
            history={}
            for i,c in enumerate(contacts):
                leg,key,gap,basis,jac,xi=c;impulse=p[3*i:3*i+3];force=impulse/dt
                forces[leg]=basis.T@force;normals[leg]=force[2]
                contact_velocity=jac@vn
                slip=contact_velocity[:2]
                required_n=max(0.,-KN*gap-dn*contact_velocity[2])
                required_t=-KT*xi-dtan*slip
                required_t*=min(1.,mu*force[2]/max(np.linalg.norm(required_t),1e-30))
                law_residual=max(law_residual,abs(force[2]-required_n),float(np.max(abs(force[:2]-required_t))))
                cone_excess=max(cone_excess,float(np.linalg.norm(force[:2])-mu*force[2]))
                minimum_normal=min(minimum_normal,float(force[2]))
                elastic=xi+dt*slip
                if np.linalg.norm(force[:2])>=mu*force[2]-1e-8:
                    elastic=-(force[:2]+DTAN*slip)/KT
                history[key]=basis[:2].T@elastic
            self.history=history
            reaction=J.T@p/dt
        else:
            vn=vfree;reaction=np.zeros(18);self.history={}
        balance=M@((vn-v)/dt)+h+self.damping*vn+dry-applied-reaction
        momentum_residual=float(np.max(np.abs(balance)))
        qn_native=pin.integrate(self.b.model,self.b._native_q(q),self.b._native_v(vn,'v')*dt)
        qn=np.asarray(qn_native)[self.b._q_indices].copy()
        return qn,vn,dict(points=points,forces=forces,normal=normals,min_gap=min_gap,
                         contact_residual=residual,contact_iterations=nit,dynamics_residual=momentum_residual,law_residual=law_residual,
                         cone_excess=cone_excess,minimum_normal=minimum_normal)


def run_case(root,name='nominal',dt=.001,friction=.8,payload=0.,push_scale=1.,duration=26.,actuation=True):
    root=Path(root);out=root/'results_pinocchio';out.mkdir(exist_ok=True)
    cmg=load_model();ref=dict(np.load(root/'data/reference.npz',allow_pickle=False))
    interp=BPoly.from_derivatives(ref['time'],np.stack([ref['q'],ref['v'],ref['a']],axis=1))
    q=ref['q'][0].copy();v=np.zeros(18);plant=Plant(cmg,payload);ctrl=WholeBodyController(cmg)
    control_every=round(.004/dt);log_every=round(.01/dt);steps=round(duration/dt)
    assert control_every>=1 and abs(control_every*dt-.004)<1e-12 and abs(log_every*dt-.01)<1e-12
    targets=target_trajectory(np.arange(steps//control_every+1)*control_every*dt)
    limits=np.tile([23.7,23.7,45.43],4)
    joints={j['id']:j for j in cmg['joints']}
    lower=np.array([joints[j]['limits']['lower'] for j in cmg['coordinate_ids'][6:]])
    upper=np.array([joints[j]['limits']['upper'] for j in cmg['coordinate_ids'][6:]])
    logs={key:[] for key in ['time','q','v','q_ref','feet','foot_ref','stance','normal_force','support_force','contact_force','predicted_force','torque','push','body_error','angle_error','contact_residual']}
    peak_qp=0.;peak_residual=0.;peak_dyn=0.;peak_iter=0;min_gap=0.;min_margin=1.;peak_torque=0.;min_height=1.;peak_tilt=0.
    dense_q=[];dense_time=[];peak_law=0.;peak_cone=0.;min_normal=0.
    impulses=np.zeros((2,3));complete=True;failure=None;start=time.monotonic()
    for k in range(steps+1):
        t=k*dt
        dense_q.append(q.copy());dense_time.append(t)
        if k%control_every==0:
            active,av,aa,stance=[x[k//control_every] for x in targets]
            qr=interp(t);vr=interp(t,nu=1);ar=interp(t,nu=2)
            if actuation:
                tau,pred,viol,iters=ctrl.command(q,v,qr,vr,ar,active,av,aa,stance)
                peak_qp=max(peak_qp,viol)
            else:tau=np.zeros(12);pred=np.zeros((4,3))
        pulse=np.zeros(3)
        if 1.2<=t<1.35:pulse=push_scale*np.array([0.,32.,0.])
        if 24.3<=t<24.5:pulse=push_scale*np.array([-25.,0.,0.])
        # Forces logged at t are impulses solved over [t,t+dt)/dt.
        qn,vn,diag=plant.step(q,v,tau,pulse,dt,friction)
        peak_residual=max(peak_residual,diag['contact_residual']);peak_dyn=max(peak_dyn,diag['dynamics_residual'])
        peak_law=max(peak_law,diag['law_residual']);peak_cone=max(peak_cone,diag['cone_excess']);min_normal=min(min_normal,diag['minimum_normal'])
        peak_iter=max(peak_iter,diag['contact_iterations']);min_gap=min(min_gap,diag['min_gap'])
        min_margin=min(min_margin,float(np.min(np.minimum(q[6:]-lower,upper-q[6:]))))
        peak_torque=max(peak_torque,float(np.max(abs(tau)/limits)));min_height=min(min_height,float(q[2]));peak_tilt=max(peak_tilt,float(np.linalg.norm(q[4:6])))
        if k%log_every==0:
            qr=interp(t);sample=min(round(t/.01),len(ref['time'])-1)
            row=[t,q.copy(),v.copy(),qr,diag['points'],ref['feet'][sample],ref['stance'][sample],diag['normal'],diag['forces'][:,2],diag['forces'],pred,tau,np.r_[pulse,np.zeros(3)],np.linalg.norm(q[:3]-qr[:3]),np.linalg.norm(q[3:6]-qr[3:6]),diag['contact_residual']]
            for key,value in zip(logs,row):logs[key].append(value)
        if q[2]<.18 or np.linalg.norm(q[4:6])>.65 or not np.all(np.isfinite(q)):
            complete=False;failure='fall/posture guard';break
        if k==steps:break
        impulses[int(t>=20)]+=pulse*dt
        q,v=qn,vn
        if k and k%round(5/dt)==0:print(f'{name}: {t:.1f}/{duration:.1f}s',flush=True)
    logs={key:np.asarray(value) for key,value in logs.items()}
    np.savez_compressed(out/f'{name}.npz',**logs,time_dense=dense_time,q_dense=dense_q)
    metrics=dict(name=name,engine='Pinocchio '+pin.__version__,dt_s=dt,friction=friction,point_payload_kg=payload,push_scale=push_scale,
        terrain=True,completed=complete,simulated_s=float(t),failure=failure,actuation=actuation,
        final_position_error_m=float(np.linalg.norm(q[:3]-interp(duration)[:3])),final_yaw_error_rad=float(abs(q[3]-interp(duration)[3])),
        rms_body_error_m=float(np.sqrt(np.mean(logs['body_error']**2))),peak_body_error_m=float(np.max(logs['body_error'])),
        peak_joint_tracking_error_rad=float(np.max(abs(logs['q'][:,6:]-logs['q_ref'][:,6:]))),
        min_base_height_m=min_height,max_tilt_rad=peak_tilt,min_joint_margin_rad=min_margin,peak_torque_limit_fraction=peak_torque,
        max_qp_violation=peak_qp,max_contact_fixed_point_residual_Ns=peak_residual,max_contact_iterations=peak_iter,
        max_discrete_dynamics_residual_N_or_Nm=peak_dyn,max_contact_law_residual_N=peak_law,max_friction_cone_excess_N=peak_cone,min_contact_normal_N=min_normal,max_foot_penetration_m=-min_gap,
        final_speed_m_s=float(np.linalg.norm(v[:3])),measured_contact_modes=sorted(set(np.sum(logs['support_force']>2,axis=1).tolist())),
        external_push_impulse_Ns=impulses[0].tolist(),recovery_push_impulse_Ns=impulses[1].tolist(),
        wall_seconds=time.monotonic()-start,mujoco_imported='mujoco' in sys.modules,physical_dofs=18,motors=12,
        contact_model=dict(normal_stiffness_N_m=KN,normal_damping_Ns_m=DN,tangent_stiffness_N_m=KT,tangent_damping_Ns_m=DTAN,
        radius_m=RADIUS,normal='unilateral implicit spring damper',friction='circular Coulomb cap on implicit tangential spring/damper; projected iteration; elastic memory/plastic slip',
        contact_point='physical sphere surface with angular Jacobian',torsional_and_rolling_resistance=False))
    (out/f'{name}.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics
