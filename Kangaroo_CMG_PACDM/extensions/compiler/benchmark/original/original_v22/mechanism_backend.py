"""Generic scalar-joint CMG evaluators and point-closure adapter.

No Kangaroo joint names or hand-derived leg equations occur in this module.
The accepted source body-sum/bias implementations are reused unchanged.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from source_bias import SourceBiasDynamics
from source_dynamics import _skew
from constraint_solvers import reduce_kinematics, solve_reduced, solve_kkt


def array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name+' must be real')
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(name+' has invalid shape or nonfinite values')
    return value.copy()


def fmt(value):
    return ' '.join(format(float(x), '.17g') for x in np.ravel(value))


class Mechanism:
    def __init__(self, cmg):
        self.cmg = deepcopy(cmg)
        if cmg.get('schema') != 'roboir.cmg.mechanism/0.16':
            raise ValueError('Unsupported mechanism schema')
        self.ids = list(cmg['coordinate_ids'])
        self.n = len(self.ids)
        self.active = [self.ids.index(j) for j in cmg['independent_ids']]
        if len(set(self.active)) != len(self.active):
            raise ValueError('Repeated independent coordinate')
        self.passive = [i for i in range(self.n) if i not in self.active]
        self.source = SourceBiasDynamics(cmg, self.ids)
        self.bodies = {b['id']: b for b in cmg['bodies']}
        self.joints = {j['id']: j for j in cmg['joints']}
        self.lower = np.array([self.joints[j]['limits']['lower'] for j in self.ids])
        self.upper = np.array([self.joints[j]['limits']['upper'] for j in self.ids])
        self.B = np.zeros((self.n, len(cmg['actuators'])))
        for column, motor in enumerate(cmg['actuators']):
            self.B[self.ids.index(motor['joint']), column] = 1.
        for body in cmg['bodies']:
            I = np.asarray(body['inertia_com_kg_m2'])
            eig = np.linalg.eigvalsh(I)
            if body['kind'] == 'rigid_body' and (eig[0] <= 0 or eig[-1] > sum(eig[:2])+1e-10):
                raise ValueError('Invalid physical principal inertias')
        for c in cmg['closures']:
            if c['type'] != 'point_coincidence' or c['body1'] not in self.bodies or c['body2'] not in self.bodies:
                raise ValueError('Invalid point constraint')
            array(c['point1_m'], (3,), 'first anchor')
            array(c['point2_m'], (3,), 'second anchor')

    def closure(self, evaluation):
        residual, jac, dot = [], [], []
        for c in self.cmg['closures']:
            a, aj, ad = self.source._point(evaluation, c['body1'], np.array(c['point1_m']))
            b, bj, bd = self.source._point(evaluation, c['body2'], np.array(c['point2_m']))
            residual.append(a-b); jac.append(aj-bj); dot.append(ad-bd)
        return np.concatenate(residual), np.vstack(jac), np.vstack(dot)

    def project(self, independent, seed):
        u = array(independent, (len(self.active),), 'independent position')
        q = array(seed, (self.n,), 'seed')
        if np.any(u < self.lower[self.active]) or np.any(u > self.upper[self.active]):
            raise ValueError('Independent position outside source bounds')
        q[self.active] = u
        def calculate(x):
            q[self.passive] = x
            return self.closure(self.source.evaluate(q, np.zeros(self.n), [0,0,0]))
        result = least_squares(lambda x:calculate(x)[0], q[self.passive],
                               jac=lambda x:calculate(x)[1][:,self.passive],
                               bounds=(self.lower[self.passive], self.upper[self.passive]),
                               xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=60)
        q[self.passive] = result.x
        r,j,_ = calculate(result.x)
        if not result.success or np.max(abs(r)) > 1e-9 or np.linalg.matrix_rank(j[:,self.passive],tol=1e-9) != len(self.passive):
            raise ValueError('No closed regular solution on the bounded local branch')
        return q.copy()

    def state(self, q, independent_velocity, gravity):
        q = array(q, (self.n,), 'configuration')
        ud = array(independent_velocity, (len(self.active),), 'independent velocity')
        e0 = self.source.evaluate(q, np.zeros(self.n), gravity)
        r,j,jd = self.closure(e0)
        k0 = reduce_kinematics(r,j,jd,np.zeros(self.n),self.active)
        v = k0['tangent_map'] @ ud
        e = self.source.evaluate(q,v,gravity)
        r,j,jd = self.closure(e)
        k = reduce_kinematics(r,j,jd,v,self.active)
        return {**e, **k}

    def inverse(self, state, independent_acceleration):
        u = array(independent_acceleration,(len(self.active),),'requested acceleration')
        n,b = state['tangent_map'],state['curvature']
        tau = state['mass_matrix'] @ (n@u+b) + state['bias_forces']
        return np.linalg.solve(n.T@self.B, n.T@tau)


class PinTree:
    def __init__(self, mechanism):
        self.mech = mechanism
        cmg = mechanism.cmg
        self.model = pin.Model()
        self.frames = {}
        self.supports = {cmg['root_body']: 0}
        self.placements = {cmg['root_body']: pin.SE3.Identity()}
        self.joint_index = {}
        self.frames[cmg['root_body']] = self.model.addFrame(pin.Frame(
            cmg['root_body'],0,0,pin.SE3.Identity(),pin.FrameType.BODY),False)
        def se3(t):
            t=np.asarray(t);return pin.SE3(t[:3,:3].copy(),t[:3,3].copy())
        def visit(parent):
            children=sorted((j for j in cmg['joints'] if j['base_body']==parent),key=lambda j:j['id'])
            for j in children:
                child=j['follower_body'];support=self.supports[parent]
                placement=self.placements[parent]*se3(j['T_BJ'])
                relative=se3(j['T_FJ']).inverse()
                if j['type']=='fixed':
                    relative=placement*relative
                else:
                    axis=np.asarray(j['axis'],dtype=float)
                    kind=(pin.JointModelRevoluteUnaligned(axis) if j['type']=='revolute'
                          else pin.JointModelPrismaticUnaligned(axis))
                    support=self.model.addJoint(support,kind,placement,j['id'])
                    self.joint_index[j['id']]=support
                self.supports[child]=support;self.placements[child]=relative
                self.frames[child]=self.model.addFrame(pin.Frame(child,support,self.frames[parent],relative,pin.FrameType.BODY),False)
                b=mechanism.bodies[child]
                self.model.appendBodyToJoint(support,pin.Inertia(b['mass_kg'],np.array(b['com_m']),np.array(b['inertia_com_kg_m2'])),relative)
                visit(child)
        visit(cmg['root_body'])
        self.order=np.array([self.model.joints[self.joint_index[x]].idx_v for x in mechanism.ids])
        self.data=self.model.createData()

    def evaluate(self,q,v,gravity):
        nq=np.zeros(self.model.nq);nv=np.zeros(self.model.nv)
        nq[self.order]=q;nv[self.order]=v
        self.model.gravity.linear=np.asarray(gravity,dtype=float)
        self.model.gravity.angular=np.zeros(3)
        mass=pin.crba(self.model,self.data,nq).copy()
        h=pin.nonLinearEffects(self.model,self.data,nq,nv).copy()
        g=pin.computeGeneralizedGravity(self.model,self.data,nq).copy()
        potential=float(pin.computePotentialEnergy(self.model,self.data,nq))
        pin.computeJointJacobiansTimeVariation(self.model,self.data,nq,nv)
        pin.updateFramePlacements(self.model,self.data)
        poses={};jac={};dots={};vel={}
        for body,fid in self.frames.items():
            poses[body]=self.data.oMf[fid].homogeneous.copy()
            jac[body]=pin.getFrameJacobian(self.model,self.data,fid,pin.LOCAL_WORLD_ALIGNED)[:,self.order].copy()
            dots[body]=pin.getFrameJacobianTimeVariation(self.model,self.data,fid,pin.LOCAL_WORLD_ALIGNED)[:,self.order].copy()
            vel[body]=jac[body]@v
        mass=mass[np.ix_(self.order,self.order)]
        return {'mass_matrix':mass,'bias_forces':h[self.order],
                'gravity_compensation':g[self.order], 'potential_energy':potential,
                'kinetic_energy':float(.5*v@mass@v),'poses':poses,'jacobians':jac,
                'body_jacobian_dots':dots,'body_velocities':vel}


def export_mjcf(mechanism, path, timestep=.0005, solref=.002):
    cmg=mechanism.cmg
    root=ET.Element('mujoco',model='roboir_generic_mechanism')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false',fusestatic='false')
    opt=ET.SubElement(root,'option',timestep=str(timestep),gravity='0 0 -9.81',
                      integrator='RK4',solver='Newton',iterations='100',tolerance='1e-12')
    ET.SubElement(opt,'flag',contact='disable',limit='disable',energy='enable')
    defaults=ET.SubElement(root,'default')
    ET.SubElement(defaults,'joint',damping='0',armature='0',frictionloss='0',limited='false')
    ET.SubElement(defaults,'geom',contype='0',conaffinity='0',rgba='.25 .55 .7 1')
    world=ET.SubElement(root,'worldbody')
    frame=ET.SubElement(world,'body',name=cmg['root_body'])
    def pose(t):
        t=np.asarray(t);quat=Rotation.from_matrix(t[:3,:3]).as_quat()
        return {'pos':fmt(t[:3,3]),'quat':fmt(quat[[3,0,1,2]])}
    def visit(parent,element):
        for j in sorted((j for j in cmg['joints'] if j['base_body']==parent),key=lambda j:j['id']):
            if not np.allclose(j['T_FJ'],np.eye(4),atol=1e-14,rtol=0):
                raise ValueError('v0.16 URDF exporter requires follower joint origin identity')
            child=j['follower_body'];b=mechanism.bodies[child]
            el=ET.SubElement(element,'body',name=child,**pose(j['T_BJ']))
            if j['type']!='fixed':
                ET.SubElement(el,'joint',name=j['id'],type='hinge' if j['type']=='revolute' else 'slide',axis=fmt(j['axis']))
            I=np.asarray(b['inertia_com_kg_m2'])
            eig,axes=np.linalg.eigh(I)
            if np.linalg.det(axes)<0:
                axes[:,0]*=-1
            quat=Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
            ET.SubElement(el,'inertial',pos=fmt(b['com_m']),mass=str(b['mass_kg']),
                          diaginertia=fmt(eig),quat=fmt(quat))
            for c in cmg['closures']:
                if child==c['body1']:
                    ET.SubElement(el,'site',name=c['id']+'_a',pos=fmt(c['point1_m']),size='.004')
                if child==c['body2']:
                    ET.SubElement(el,'site',name=c['id']+'_b',pos=fmt(c['point2_m']),size='.004')
            visit(child,el)
    visit(cmg['root_body'],frame)
    equality=ET.SubElement(root,'equality')
    for c in cmg['closures']:
        ET.SubElement(equality,'connect',name=c['id'],body1=c['body1'],body2=c['body2'],
                      anchor=fmt(c['point1_m']),solref=f'{solref} 1',solimp='.9999 .9999 .001')
    motors=ET.SubElement(root,'actuator')
    for a in cmg['actuators']:
        ET.SubElement(motors,'motor',name=a['id'],joint=a['joint'],gear='1',ctrllimited='false')
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


class MJTree:
    def __init__(self,mechanism,path):
        self.mech=mechanism
        self.model=mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
        self.data=mujoco.MjData(self.model)
        self.order=np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in mechanism.ids])
        self.bids={b:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,b) for b in mechanism.bodies}

    def set_state(self,q,v,gravity):
        self.data.qpos[self.order]=q;self.data.qvel[self.order]=v
        self.model.opt.gravity[:]=gravity
        self.data.ctrl[:]=0;self.data.qfrc_applied[:]=0
        mujoco.mj_forward(self.model,self.data)

    def evaluate(self,q,v,gravity):
        self.set_state(q,v,gravity)
        mass=np.zeros((self.model.nv,self.model.nv))
        mujoco.mj_fullM(self.model,mass,self.data.qM)
        poses={};jac={}
        for name,bid in self.bids.items():
            t=np.eye(4);t[:3,:3]=self.data.xmat[bid].reshape(3,3);t[:3,3]=self.data.xpos[bid]
            poses[name]=t
            jp=np.zeros((3,self.model.nv));jr=jp.copy()
            mujoco.mj_jacBody(self.model,self.data,jp,jr,bid)
            jac[name]=np.vstack([jp,jr])[:,self.order]
        return {'mass_matrix':mass[np.ix_(self.order,self.order)],
                'bias_forces':self.data.qfrc_bias[self.order].copy(),
                'potential_energy':float(self.data.energy[0]),'kinetic_energy':float(self.data.energy[1]),
                'poses':poses,'jacobians':jac}

    def closure_jacobian(self):
        r=[];j=[]
        for c in self.mech.cmg['closures']:
            ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,c['id']+suffix) for suffix in ['_a','_b']]
            r.append(self.data.site_xpos[ids[0]]-self.data.site_xpos[ids[1]])
            js=[]
            for i in ids:
                jp=np.zeros((3,self.model.nv));jr=jp.copy()
                mujoco.mj_jacSite(self.model,self.data,jp,jr,i);js.append(jp[:,self.order])
            j.append(js[0]-js[1])
        return np.concatenate(r),np.vstack(j)
