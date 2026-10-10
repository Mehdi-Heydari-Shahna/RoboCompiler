"""Engine-independent model, transforms, observables, and force-port audits.

Body velocity convention here is WORLD linear velocity at COM, then WORLD
angular velocity. Native joint positions/velocities are SI (metres/radians).
No routine in this module advances a physics simulation or edits native state.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
FOOT_NAMES = ('left_ankle_roll', 'right_ankle_roll')


def finite(a, shape=None, name='array'):
    a = np.asarray(a, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(f'{name}: invalid shape or nonfinite values: {a.shape}')
    return a


def inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def axis_frame(axis):
    """Proper rotation with local +X mapped to the physical source axis."""
    x = finite(axis, (3,), 'joint axis').copy()
    if abs(np.linalg.norm(x)-1) > 1e-9:
        raise ValueError('Nonunit joint axis')
    basis = np.eye(3)[int(np.argmin(abs(x)))]
    y = basis - x * np.dot(x, basis)
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def poses_from_xyzw(transforms):
    x = finite(transforms, name='PhysX transforms')
    if x.ndim != 2 or x.shape[1] != 7:
        raise ValueError('Expected [bodies, xyz + quaternion xyzw]')
    out = np.repeat(np.eye(4)[None], len(x), axis=0)
    out[:, :3, :3] = Rotation.from_quat(x[:, 3:7]).as_matrix()
    out[:, :3, 3] = x[:, :3]
    return out


class Model:
    def __init__(self, cmg=None):
        self.c = json.loads((ROOT/'data/whole_body_cmg.json').read_text()) if cmg is None else cmg
        c = self.c
        self.names = [b['id'] for b in c['bodies']]
        self.bi = {n:i for i,n in enumerate(self.names)}
        self.ids = list(c['coordinate_ids'])
        self.qi = {n:i for i,n in enumerate(self.ids)}
        self.joints = {j['id']:j for j in c['joints']}
        self.n, self.nb = len(self.ids), len(self.names)
        self.root = self.bi[c['root_body']]
        self.torso = self.bi['torso']
        self.feet = np.array([self.bi[n] for n in FOOT_NAMES])
        self.active = np.array([self.qi[n] for n in c['independent_ids']])
        self.mass = np.array([b['mass_kg'] for b in c['bodies']])
        self.com_local = np.array([b['com_m'] for b in c['bodies']])
        self.I_local = np.array([b['inertia_com_kg_m2'] for b in c['bodies']])
        self.armature = np.array([c['armature'][n] for n in self.ids])
        self.damping = np.array([c['joint_dissipation'][n]['damping'] for n in self.ids])
        self.lower = np.array([self.joints[n]['limits']['lower'] for n in self.ids])
        self.upper = np.array([self.joints[n]['limits']['upper'] for n in self.ids])
        self.slide = np.array([self.joints[n]['type']=='prismatic' for n in self.ids])
        self.bounds = np.array([a['force_bounds_N'] for a in c['actuators']])
        self.edges = []
        done = {c['root_body']}
        pending = list(c['joints'])
        while pending:
            eligible = [j for j in pending if j['base_body'] in done]
            if not eligible:
                raise ValueError('Disconnected or cyclic computational tree')
            for j in eligible:
                if j['follower_body'] in done:
                    raise ValueError('Repeated tree child')
                b, f = self.bi[j['base_body']], self.bi[j['follower_body']]
                qindex = None if j['type']=='fixed' else self.qi[j['id']]
                self.edges.append((b, f, np.array(j['T_BJ']), inverse(np.array(j['T_FJ'])),
                                   qindex, np.array(j['axis']), j['type']))
                done.add(j['follower_body']); pending.remove(j)
        if len(done) != self.nb or len(self.edges) != self.nb-1:
            raise ValueError('Tree does not cover all bodies')
        cuts = c['closures']
        self.ca = np.array([self.bi[x['body1']] for x in cuts])
        self.cb = np.array([self.bi[x['body2']] for x in cuts])
        self.pa = np.array([x['point1_m'] for x in cuts])
        self.pb = np.array([x['point2_m'] for x in cuts])
        self.universal = np.array([i for i,x in enumerate(cuts) if x['type']=='universal'])
        self.ua = np.array([np.array(cuts[i]['frame1_R'])[:,0] for i in self.universal])
        self.ub = np.array([np.array(cuts[i]['frame2_R'])[:,1] for i in self.universal])
        self.visual = json.loads((ROOT/'assets/visuals.json').read_text())
        self.sole_corners = []
        import itertools
        for foot in FOOT_NAMES:
            geom = next(x for x in self.visual['collisions'] if x['body']==foot)
            w,x,y,z=geom['quaternion_wxyz']
            r = Rotation.from_quat([x,y,z,w]).as_matrix()
            points = np.array(list(itertools.product(*[[-h,h] for h in geom['half_extents']])))
            self.sole_corners.append(points @ r.T + geom['position'])
        self.sole_corners = np.array(self.sole_corners)
        self.validate()

    def validate(self):
        if self.nb != 78 or self.n != 76 or len(self.active) != 12:
            raise ValueError('Not the original v22 full-body robot')
        if len(set(self.names)) != self.nb or len(set(self.ids)) != self.n:
            raise ValueError('Duplicate IDs')
        if len(self.ca) != 24 or len(self.universal)!=8:
            raise ValueError('Expected sixteen point and eight universal cuts')
        if not np.all(self.slide[self.active]) or np.any(self.mass<=0):
            raise ValueError('Invalid mass or physical force ports')
        eig=np.linalg.eigvalsh(self.I_local)
        if np.any(eig<=0) or np.any(eig[:,2]>eig[:,0]+eig[:,1]+1e-10):
            raise ValueError('Nonphysical inertia')
        for j in self.joints.values():
            for key in ('T_BJ','T_FJ'):
                T=finite(j[key],(4,4),key); R=T[:3,:3]
                if np.linalg.norm(R.T@R-np.eye(3))>1e-8 or abs(np.linalg.det(R)-1)>1e-8 or not np.allclose(T[3],[0,0,0,1]):
                    raise ValueError('Invalid joint transform')
        for a,n in zip(self.c['actuators'],self.c['independent_ids']):
            if a['joint']!=n or a['gear']!=1 or a['unit']!='N':
                raise ValueError('Actuator-order/gear/unit mismatch')

    def fk(self, q, base=None, rotation=None):
        q=finite(q,(self.n,),'q')
        P=np.repeat(np.eye(4)[None],self.nb,axis=0)
        if base is not None:P[self.root,:3,3]=finite(base,(3,),'base')
        if rotation is not None:P[self.root,:3,:3]=finite(rotation,(3,3),'rotation')
        for b,f,A,B,k,axis,kind in self.edges:
            motion=np.eye(4)
            if k is not None:
                if kind=='prismatic': motion[:3,3]=axis*q[k]
                else:
                    K=skew(axis); motion[:3,:3]=np.eye(3)+np.sin(q[k])*K+(1-np.cos(q[k]))*(K@K)
            P[f]=P[b]@A@motion@B
        return P

    def joint_frames(self,j):
        Q=np.eye(4);Q[:3,:3]=axis_frame(j['axis'])
        return np.array(j['T_BJ'])@Q, np.array(j['T_FJ'])@Q

    def closures(self,P):
        a=P[self.ca,:3,3]+np.einsum('nij,nj->ni',P[self.ca,:3,:3],self.pa)
        b=P[self.cb,:3,3]+np.einsum('nij,nj->ni',P[self.cb,:3,:3],self.pb)
        ia=self.ca[self.universal];ib=self.cb[self.universal]
        va=np.einsum('nij,nj->ni',P[ia,:3,:3],self.ua)
        vb=np.einsum('nij,nj->ni',P[ib,:3,:3],self.ub)
        dots=np.einsum('ni,ni->n',va,vb)
        L=.1
        distance=np.linalg.norm(a[self.universal]+L*va-b[self.universal]-L*vb,axis=1)-np.sqrt(2)*L
        return a-b,dots,distance

    def com_world(self,P):
        return P[:,:3,3]+np.einsum('nij,nj->ni',P[:,:3,:3],self.com_local)

    def energy(self,P,V,qd):
        V=finite(V,(self.nb,6),'COM velocities'); qd=finite(qd,(self.n,),'qd')
        com=self.com_world(P);R=P[:,:3,:3]
        I=np.einsum('nij,njk,nlk->nil',R,self.I_local,R)
        kinetic=.5*self.mass*np.einsum('ni,ni->n',V[:,:3],V[:,:3])+.5*np.einsum('ni,nij,nj->n',V[:,3:],I,V[:,3:])
        potential=self.mass*9.81*com[:,2]
        rotor=.5*self.armature*qd**2
        return kinetic,potential,rotor

    def foot_points(self,P):
        return np.einsum('nij,nkj->nki',P[self.feet,:3,:3],self.sole_corners)+P[self.feet,None,:3,3]

    def port_wrenches(self,P,V,force):
        """Body force pairs implied by the twelve scalar prismatic force ports.

        Output wrenches are world-frame about body COMs. This is an independent
        geometric reconstruction, NOT an independent native force measurement.
        """
        force=finite(force,(12,),'actuator force'); com=self.com_world(P)
        w=np.zeros((self.nb,6));power=0.
        for k,jid in enumerate(self.c['independent_ids']):
            j=self.joints[jid];b=self.bi[j['base_body']];f=self.bi[j['follower_body']]
            A=P[b]@np.array(j['T_BJ']); B=P[f]@np.array(j['T_FJ'])
            F=A[:3,:3]@np.array(j['axis'])*force[k]
            for bid,point,sgn in ((b,A[:3,3],-1.),(f,B[:3,3],1.)):
                Fbody=sgn*F;torque=np.cross(point-com[bid],Fbody)
                w[bid,:3]+=Fbody;w[bid,3:]+=torque
                power+=float(Fbody@V[bid,:3]+torque@V[bid,3:])
        return w,power

    def coordinates_from_poses(self,P):
        q=np.zeros(self.n)
        for name in self.ids:
            j=self.joints[name];a=P[self.bi[j['base_body']]]@np.array(j['T_BJ'])
            b=P[self.bi[j['follower_body']]]@np.array(j['T_FJ']);rel=inverse(a)@b
            axis=np.array(j['axis']);i=self.qi[name]
            q[i]=float(axis@rel[:3,3]) if self.slide[i] else float(axis@Rotation.from_matrix(rel[:3,:3]).as_rotvec())
        return q


def triangular_face_counts(indices):
    """USD face counts for a flattened or [F,3] triangle index buffer."""
    a=np.asarray(indices)
    if not np.issubdtype(a.dtype,np.integer) or a.size==0 or a.size%3:
        raise ValueError('Triangle indices must be nonempty integer triples')
    return np.full(a.size//3,3,dtype=np.int32)


def inertias_match(actual,expected):
    """Per-body matrix-norm comparison allowing float32 principal-axis storage.

    The tolerance scales with EACH inertia tensor, not the heaviest body. Tiny
    off-diagonal entries do not cause false failures after quaternion storage.
    This does not modify any source inertia tensor.
    """
    a=finite(actual,name='actual inertia');b=finite(expected,name='expected inertia')
    if a.shape!=b.shape or a.shape[-2:]!=(3,3):return False
    err=np.max(abs(a-b),axis=(-2,-1));scale=np.max(abs(b),axis=(-2,-1))
    return bool(np.all(err<=1e-6*scale+1e-12))
