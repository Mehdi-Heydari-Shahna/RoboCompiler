"""Independent world-frame rigid-body tree evaluator (no PACDM/Pinocchio calls).

Scalar revolute/prismatic joints, fixed body attachments, SI units. All physical
vectors use CMG coordinate order. Jacobians are linear-first/angular-last.
Bias accelerations use exact classical recursion with qdd=0, not differencing.
This is an independent NumPy reference implementation, NOT Pinocchio or URDF+.
"""
from __future__ import annotations
import numpy as np


def cross_matrix(x):
    x, y, z = x
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def axis_rotation(axis, angle):
    K = cross_matrix(axis)
    return np.eye(3) + np.sin(angle)*K + (1.-np.cos(angle))*(K@K)


class NumpyTree:
    def __init__(self, cmg):
        self.cmg = cmg
        self.ids = list(cmg['coordinate_ids'])
        self.n = len(self.ids)
        self.active = np.array([self.ids.index(k) for k in cmg['independent_ids']])
        self.passive = np.setdiff1d(np.arange(self.n), self.active)
        self.bodies = {b['id']: b for b in cmg['bodies']}
        self.gravity = np.asarray(cmg.get('gravity_m_s2', [0.,0.,-9.81]))
        done = {cmg['root_body']}
        self.edges = []
        while len(self.edges) < len(cmg['joints']):
            before = len(self.edges)
            for j in cmg['joints']:
                if j['base_body'] in done and j['follower_body'] not in done:
                    E = np.array(j['T_BJ'], float)
                    F = np.array(j['T_FJ'], float)
                    L = np.eye(4)
                    L[:3,:3] = F[:3,:3].T
                    L[:3,3] = -F[:3,:3].T@F[:3,3]
                    k = self.ids.index(j['id']) if j['type'] != 'fixed' else None
                    self.edges.append((j, E, L, k))
                    done.add(j['follower_body'])
            if before == len(self.edges):
                raise ValueError('Expected connected, directed CMG tree')
        if len(done) != len(self.bodies):
            raise ValueError('Disconnected body records')

    def forward(self, q, velocity=None, acceleration=False):
        q = np.asarray(q, float)
        if q.shape != (self.n,) or not np.all(np.isfinite(q)):
            raise ValueError('Invalid physical configuration')
        v = np.zeros(self.n) if velocity is None else np.asarray(velocity, float)
        root = dict(R=np.eye(3), p=np.zeros(3), Jv=np.zeros((3,self.n)),
                    Jw=np.zeros((3,self.n)), w=np.zeros(3), alpha=np.zeros(3),
                    a=np.zeros(3))
        states = {self.cmg['root_body']: root}
        for joint, E, L, k in self.edges:
            b = states[joint['base_body']]
            re = b['R']@E[:3,3]
            R = b['R']@E[:3,:3]
            p = b['p'] + re
            Jv = b['Jv'] - cross_matrix(re)@b['Jw']
            Jw = b['Jw'].copy()
            wp = b['w']
            alpha = b['alpha'].copy()
            w = wp.copy()
            a = (b['a'] + np.cross(alpha, re) + np.cross(wp, np.cross(wp,re))
                 if acceleration else np.zeros(3))
            if k is not None:
                axis = np.asarray(joint['axis'], float)
                aw = R@axis
                if joint['type'] == 'revolute':
                    R = R@axis_rotation(axis, q[k])
                    Jw[:,k] += aw
                    w += aw*v[k]
                    if acceleration:
                        alpha += np.cross(wp, aw)*v[k]
                elif joint['type'] == 'prismatic':
                    r = aw*q[k]
                    p += r
                    Jv -= cross_matrix(r)@b['Jw']
                    Jv[:,k] += aw
                    if acceleration:
                        a += (np.cross(alpha,r) + np.cross(wp,np.cross(wp,r))
                              + 2*np.cross(wp,aw)*v[k])
                else:
                    raise ValueError('Only scalar revolute/prismatic joints supported')
            rl = R@L[:3,3]
            p += rl
            Jv -= cross_matrix(rl)@Jw
            if acceleration:
                a += np.cross(alpha,rl) + np.cross(w,np.cross(w,rl))
            states[joint['follower_body']] = dict(R=R@L[:3,:3],p=p,Jv=Jv,Jw=Jw,
                                                  w=w,alpha=alpha,a=a)
        return states

    def geometry(self, q, velocity=None):
        states = self.forward(q, velocity, acceleration=velocity is not None)
        gaps, jac, gamma = [], [], []
        for c in self.cmg['closures']:
            ends = []
            for side in (1,2):
                b = states[c[f'body{side}']]
                r = b['R']@np.asarray(c[f'point{side}_m'])
                ends.append((b['p']+r, b['Jv']-cross_matrix(r)@b['Jw'],
                             b['a']+np.cross(b['alpha'],r)
                             +np.cross(b['w'],np.cross(b['w'],r))))
            gaps.append(ends[0][0]-ends[1][0])
            jac.append(ends[0][1]-ends[1][1])
            gamma.append(ends[0][2]-ends[1][2])
        platform = states['platform']
        return dict(point_difference=np.concatenate(gaps), jacobian=np.vstack(jac),
                    acceleration_bias=np.concatenate(gamma),
                    platform_jacobian=np.vstack((platform['Jv'],platform['Jw'])),
                    states=states)

    def mass_bias(self, q, velocity):
        states = self.forward(q, velocity, acceleration=True)
        M = np.zeros((self.n,self.n))
        b = np.zeros(self.n)
        potential = 0.
        for name, body in self.bodies.items():
            mass = float(body['mass_kg'])
            if mass == 0.:
                continue
            s = states[name]
            r = s['R']@np.asarray(body['com_m'])
            Jv = s['Jv']-cross_matrix(r)@s['Jw']
            Jw = s['Jw']
            I = s['R']@np.asarray(body['inertia_kg_m2'])@s['R'].T
            a = s['a']+np.cross(s['alpha'],r)+np.cross(s['w'],np.cross(s['w'],r))
            M += mass*(Jv.T@Jv) + Jw.T@I@Jw
            b += mass*Jv.T@(a-self.gravity) + Jw.T@(I@s['alpha']+np.cross(s['w'],I@s['w']))
            potential -= mass*self.gravity@(s['p']+r)
        return M,b,float(potential)

    def mapping(self, q):
        geom = self.geometry(q)
        J = geom['jacobian']
        N = np.zeros((self.n,len(self.active)))
        N[self.active] = np.eye(len(self.active))
        N[self.passive] = -np.linalg.solve(J[:,self.passive],J[:,self.active])
        return N, geom
