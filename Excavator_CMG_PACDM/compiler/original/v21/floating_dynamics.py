"""Independent source body sums and Pinocchio free-body recursion.
World-coordinate formulas adapted from accepted Kangaroo v20.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import pinocchio as pin
import mujoco
from source_dynamics import _skew
class FloatingSource:
    def __init__(self,cmg,source):
        self.cmg=cmg;self.source=source;self.n=source.nv+6
    def evaluate(self,q,p,R,v,gravity):
        v=np.asarray(v);g=np.asarray(gravity);omega=v[3:6]
        local=self.source.evaluate(q,v[6:],[0,0,0]);poses={};js={};ds={};vel={};cj={};cd={}
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


def directed_tree(cmg,mapping,tree_ids):
 from copy import deepcopy
 out=deepcopy(cmg);records={j['id']:j for j in out['joints']};joints=[]
 for edge in mapping['tree']:
  j=records[edge['id']]
  if edge['direction']==-1:
   j['base_body'],j['follower_body']=j['follower_body'],j['base_body'];j['T_BJ'],j['T_FJ']=j['T_FJ'],j['T_BJ'];j['axis']=(-np.asarray(j['axis'])).tolist()
  joints.append(j)
 out['joints']=joints;out['coordinate_ids']=list(tree_ids);return out
class MJFloating:
 def __init__(self,cmg,tree_ids,path):
  self.model=mujoco.MjModel.from_xml_path(str(path));self.data=mujoco.MjData(self.model);self.ids=tree_ids;self.bodies=[b['id'] for b in cmg['bodies'] if b['kind']=='rigid_body']
  js=[self.model.joint(x).id for x in tree_ids];self.qi=self.model.jnt_qposadr[js];self.vi=self.model.jnt_dofadr[js];self.order=np.r_[np.arange(6),self.vi]
  j=next(j for j in cmg['joints'] if j['id']=='fixed_world_base');self.T0=np.asarray(j['T_BJ'])@np.linalg.inv(np.asarray(j['T_FJ']))
  if self.model.nv!=29:raise ValueError('Robot-only floating oracle required')
 def evaluate(self,q,p,R,v,gravity):
  m,d=self.model,self.data;offset=R@self.T0[:3,3];RB=R@self.T0[:3,:3];B=np.eye(29);B[:3,3:6]=-_skew(offset);B[3:6,3:6]=RB.T
  d.qpos[:3]=p+offset;d.qpos[3:7]=Rotation.from_matrix(RB).as_quat()[[3,0,1,2]];d.qpos[self.qi]=q;d.qvel[self.order]=B@v;m.opt.gravity[:]=gravity;d.ctrl[:]=0;mujoco.mj_forward(m,d)
  mn=np.zeros((29,29));mujoco.mj_fullM(m,mn,d.qM);mn=mn[np.ix_(self.order,self.order)];hn=d.qfrc_bias[self.order].copy();bd=np.zeros(29);bd[:3]=np.cross(v[3:6],np.cross(v[3:6],offset));M=B.T@mn@B;h=B.T@(hn+mn@bd);P={}
  for name in self.bodies:
   bid=m.body(name).id;T=np.eye(4);T[:3,3]=d.xpos[bid];T[:3,:3]=d.xmat[bid].reshape(3,3);P[name]=T
  return dict(mass_matrix=M,bias_forces=h,potential_energy=float(d.energy[0]),kinetic_energy=float(d.energy[1]),poses=P)
