"""Whole-body world-aligned floating-base dynamics; no fictitious root masses."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
import pinocchio as pin
import mujoco
from source_dynamics import _skew
from source_bias import SourceBiasDynamics
from reconstructed_model import UniversalMechanism
from constraint_solvers import reduce_kinematics

class FloatingSource:
    def __init__(self,cmg):
        self.internal=UniversalMechanism(cmg);self.cmg=cmg;self.n=self.internal.n+6
        self.B=np.vstack([np.zeros((6,12)),self.internal.B])
        self.active=list(range(6))+[6+i for i in self.internal.active]
    def evaluate(self,q,p,R,v,gravity):
        m=self.internal;v=np.asarray(v);g=np.asarray(gravity);omega=v[3:6]
        local=m.source.evaluate(q,v[6:],[0,0,0]);poses={};js={};ds={};vel={};cj={};cd={}
        M=np.zeros((self.n,self.n));h=np.zeros(self.n);gg=h.copy();U=0.;linear=np.zeros(3);angular=np.zeros(3);mass=0.;com=np.zeros(3)
        for b in self.cmg['bodies']:
            name=b['id'];T0=local['poses'][name];offset=R@T0[:3,3]
            T=np.eye(4);T[:3,:3]=R@T0[:3,:3];T[:3,3]=p+offset;poses[name]=T
            J0=local['jacobians'][name];D0=local['body_jacobian_dots'][name]
            J=np.zeros((6,self.n));J[:3,:3]=np.eye(3);J[:3,3:6]=-_skew(offset);J[3:,3:6]=np.eye(3)
            J[:3,6:]=R@J0[:3];J[3:,6:]=R@J0[3:]
            od=np.cross(omega,offset)+R@(J0[:3]@v[6:]);D=np.zeros_like(J);D[:3,3:6]=-_skew(od)
            D[:3,6:]=_skew(omega)@R@J0[:3]+R@D0[:3];D[3:,6:]=_skew(omega)@R@J0[3:]+R@D0[3:]
            js[name]=J;ds[name]=D;vel[name]=J@v
            r=T[:3,:3]@b['com_m'];rd=np.cross(vel[name][3:],r)
            Jc=J[:3]-_skew(r)@J[3:];Dc=D[:3]-_skew(rd)@J[3:]-_skew(r)@D[3:]
            cj[name]=Jc;cd[name]=Dc
            mb=b['mass_kg'];I=T[:3,:3]@np.array(b['inertia_com_kg_m2'])@T[:3,:3].T;w=vel[name][3:]
            M+=mb*Jc.T@Jc+J[3:].T@I@J[3:]
            grav=-mb*Jc.T@g;gg+=grav
            h+=mb*Jc.T@(Dc@v)+J[3:].T@(I@(D[3:]@v)+np.cross(w,I@w))+grav
            pc=T[:3,3]+r;U-=mb*g@pc;mom=mb*Jc@v;linear+=mom;angular+=np.cross(pc,mom)+I@w
            mass+=mb;com+=mb*pc
        return dict(configuration=q,velocity=v,poses=poses,jacobians=js,body_jacobian_dots=ds,
                    body_velocities=vel,body_com_jacobians=cj,body_com_jacobian_dots=cd,
                    mass_matrix=M,bias_forces=h,gravity_compensation=gg,potential_energy=float(U),
                    kinetic_energy=float(.5*v@M@v),linear_momentum=linear,angular_momentum=angular,
                    total_mass=mass,center_of_mass=com/mass)
    def state(self,q,p,R,ud,gravity):
        zero=self.evaluate(q,p,R,np.zeros(self.n),gravity)
        r,J,D=self.internal.closure(zero);k=reduce_kinematics(r,J,D,np.zeros(self.n),self.active)
        v=k['tangent_map']@ud;e=self.evaluate(q,p,R,v,gravity)
        r,J,D=self.internal.closure(e);k=reduce_kinematics(r,J,D,v,self.active)
        return {**e,**k}

class PinFloating:
    def __init__(self,cmg):
        self.cmg=cmg;self.model=pin.Model();bodies={b['id']:b for b in cmg['bodies']};self.frames={}
        root=cmg['root_body'];support=self.model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3.Identity(),'floating_base')
        self.supports={root:support};self.placements={root:pin.SE3.Identity()};self.joints={}
        def addbody(name,support,relative,parentframe):
            b=bodies[name];self.model.appendBodyToJoint(support,pin.Inertia(b['mass_kg'],np.array(b['com_m']),np.array(b['inertia_com_kg_m2'])),relative)
            self.frames[name]=self.model.addFrame(pin.Frame(name,support,parentframe,relative,pin.FrameType.BODY),False)
        addbody(root,support,pin.SE3.Identity(),0)
        def se3(t):t=np.asarray(t);return pin.SE3(t[:3,:3].copy(),t[:3,3].copy())
        def visit(parent):
            for j in sorted((j for j in cmg['joints'] if j['base_body']==parent),key=lambda j:j['id']):
                child=j['follower_body'];support=self.supports[parent];placement=self.placements[parent]*se3(j['T_BJ']);relative=se3(j['T_FJ']).inverse()
                if j['type']=='fixed':relative=placement*relative
                else:
                    axis=np.array(j['axis']);kind=pin.JointModelRevoluteUnaligned(axis) if j['type']=='revolute' else pin.JointModelPrismaticUnaligned(axis)
                    support=self.model.addJoint(support,kind,placement,j['id']);self.joints[j['id']]=support
                self.supports[child]=support;self.placements[child]=relative;addbody(child,support,relative,self.frames[parent]);visit(child)
        visit(root);self.vo=np.array([self.model.joints[self.joints[x]].idx_v for x in cmg['coordinate_ids']]);self.qo=self.vo+1
        self.order=np.r_[np.arange(6),self.vo];self.data=self.model.createData()
    def evaluate(self,q,p,R,v,gravity):
        n=len(v);T=np.eye(n);T[:3,:3]=R.T;T[3:6,3:6]=R.T
        nq=pin.neutral(self.model);nq[:3]=p;nq[3:7]=Rotation.from_matrix(R).as_quat();nq[self.qo]=q
        nv=np.zeros(n);nv[self.order]=T@v;self.model.gravity.linear=np.array(gravity);self.model.gravity.angular=np.zeros(3)
        M=pin.crba(self.model,self.data,nq).copy()[np.ix_(self.order,self.order)]
        h=pin.nonLinearEffects(self.model,self.data,nq,nv).copy()[self.order]
        td=np.zeros(n);td[:3]=-R.T@np.cross(v[3:6],v[:3]);h=T.T@(h+M@td);M=T.T@M@T
        U=float(pin.computePotentialEnergy(self.model,self.data,nq));pin.computeJointJacobians(self.model,self.data,nq);pin.updateFramePlacements(self.model,self.data)
        P={name:self.data.oMf[fid].homogeneous.copy() for name,fid in self.frames.items()}
        J={name:pin.getFrameJacobian(self.model,self.data,fid,pin.LOCAL_WORLD_ALIGNED)[:,self.order]@T for name,fid in self.frames.items()}
        return dict(mass_matrix=M,bias_forces=h,potential_energy=U,kinetic_energy=float(.5*v@M@v),poses=P,jacobians=J)

class MJFloating:
    def __init__(self,cmg,path):
        self.cmg=cmg;self.model=mujoco.MjModel.from_xml_path(str(path));self.data=mujoco.MjData(self.model)
        jids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in cmg['coordinate_ids']]
        self.vo=np.array([self.model.jnt_dofadr[x] for x in jids]);self.qo=np.array([self.model.jnt_qposadr[x] for x in jids]);self.order=np.r_[np.arange(6),self.vo]
        self.bids={b['id']:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,b['id']) for b in cmg['bodies']}
    def evaluate(self,q,p,R,v,gravity):
        n=len(v);T=np.eye(n);T[3:6,3:6]=R.T
        d=self.data;m=self.model;d.qpos[:3]=p;d.qpos[3:7]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]];d.qpos[self.qo]=q;d.qvel[self.order]=T@v
        m.opt.gravity[:]=gravity;d.ctrl[:]=0;d.qfrc_applied[:]=0;mujoco.mj_forward(m,d)
        M=np.zeros((n,n));mujoco.mj_fullM(m,M,d.qM);M=T.T@M[np.ix_(self.order,self.order)]@T;h=T.T@d.qfrc_bias[self.order]
        P={};J={}
        for name,bid in self.bids.items():
            tt=np.eye(4);tt[:3,:3]=d.xmat[bid].reshape(3,3);tt[:3,3]=d.xpos[bid];P[name]=tt
            jp=np.zeros((3,n));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);J[name]=np.vstack([jp,jr])[:,self.order]@T
        return dict(mass_matrix=M,bias_forces=h,potential_energy=float(d.energy[0]),kinetic_energy=float(d.energy[1]),poses=P,jacobians=J)


def add_drive_terms(e,cmg):
    """Optional source joint armature and viscous loss; no electrical model."""
    out=dict(e);offset=e['mass_matrix'].shape[0]-len(cmg['coordinate_ids'])
    arm=np.r_[np.zeros(offset),[cmg['armature'][x] for x in cmg['coordinate_ids']]]
    damp=np.r_[np.zeros(offset),[cmg['joint_dissipation'][x]['damping'] for x in cmg['coordinate_ids']]]
    out['mass_matrix']=e['mass_matrix']+np.diag(arm);out['bias_forces']=e['bias_forces']+damp*e['velocity']
    out['kinetic_energy']=e['kinetic_energy']+float(.5*np.dot(arm,e['velocity']**2))
    out['dissipation_power']=float(np.dot(damp,e['velocity']**2));return out
