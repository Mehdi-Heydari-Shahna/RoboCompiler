"""SI rigid-tree kinematics and recursive Newton--Euler dynamics from CMG.

No simulator, Pinocchio, USD, or collision-library imports. This is the
feedforward model, NOT the state integrator in the Isaac tests. Its numerical
outputs are checked against archived Pinocchio and MuJoCo runs. Spatial vectors
inside RNEA are angular-first; public tool wrenches are [Fx,Fy,Fz,Tx,Ty,Tz].
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def skew(x):
    x, y, z = x
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def inverse(T):
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def adjoint(T):
    R = T[:3, :3]
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = skew(T[:3, 3]) @ R
    return X


def cross_motion(v):
    X = np.zeros((6, 6))
    X[:3, :3] = skew(v[:3])
    X[3:, 3:] = X[:3, :3]
    X[3:, :3] = skew(v[3:])
    return X


class RigidTree:
    def __init__(self, cmg):
        self.cmg = cmg
        self.ids = list(cmg['coordinate_ids'])
        self.n = len(self.ids)
        self.gravity = np.asarray(cmg['gravity_m_s2'], dtype=float)
        self.body_names = [cmg['root_body']]
        records = {b['id']: b for b in cmg['bodies']}
        remaining = list(cmg['joints'])
        self.edges = []
        while remaining:
            progressed = False
            for j in remaining[:]:
                if j['base_body'] not in self.body_names:
                    continue
                if j['follower_body'] in self.body_names:
                    raise ValueError('CMG tree contains a duplicate child/cycle')
                parent = self.body_names.index(j['base_body'])
                self.body_names.append(j['follower_body'])
                i = len(self.body_names) - 1
                k = self.ids.index(j['id']) if j['type'] != 'fixed' else -1
                A, F = np.asarray(j['T_BJ'], float), np.asarray(j['T_FJ'], float)
                axis = np.asarray(j['axis'], float)
                if not np.isclose(np.linalg.norm(axis), 1.):
                    raise ValueError('Joint axes must be unit vectors')
                s = np.zeros(6)
                if k >= 0:
                    s[:3] = axis if j['type'] == 'revolute' else 0.
                    s[3:] = axis if j['type'] == 'prismatic' else 0.
                self.edges.append(dict(i=i, parent=parent, k=k, A=A,
                    Finv=inverse(F), S=adjoint(F) @ s, axis=axis,
                    kind=j['type'], id=j['id']))
                remaining.remove(j)
                progressed = True
            if not progressed:
                raise ValueError('CMG tree is disconnected or reversed')
        if set(self.body_names) != set(records):
            raise ValueError('CMG contains unconnected bodies')
        self.inertias, self.masses, self.coms = [], [], []
        for name in self.body_names:
            b = records[name]
            m, c = float(b['mass_kg']), np.asarray(b['com_m'], float)
            Ic = np.asarray(b['inertia_kg_m2'], float)
            if m < 0 or not np.allclose(Ic, Ic.T) or np.linalg.eigvalsh(Ic).min() < -1e-12:
                raise ValueError(f'Invalid inertia: {name}')
            C = skew(c)
            self.inertias.append(np.block([[Ic + m*C@C.T, m*C], [-m*C, m*np.eye(3)]]))
            self.masses.append(m)
            self.coms.append(c)
        self.inertias = np.asarray(self.inertias)
        self.tool_body = self.body_names.index(cmg['tool']['body'])
        self.tool_transform = np.asarray(cmg['tool']['T_body_tool'], float)
        self.parent_edges = {e['i']: e for e in self.edges}

    def _check(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n,) or not np.all(np.isfinite(q)):
            raise ValueError(f'Expected {self.n} finite coordinates')
        return q

    def transforms(self, q):
        q = self._check(q)
        local = [np.eye(4)]
        for e in self.edges:
            motion = np.eye(4)
            if e['kind'] == 'revolute':
                motion[:3, :3] = Rotation.from_rotvec(e['axis'] * q[e['k']]).as_matrix()
            elif e['kind'] == 'prismatic':
                motion[:3, 3] = e['axis'] * q[e['k']]
            local.append(e['A'] @ motion @ e['Finv'])
        return local

    def poses(self, q):
        local = self.transforms(q)
        world = [np.eye(4) for _ in self.body_names]
        for e in self.edges:
            world[e['i']] = world[e['parent']] @ local[e['i']]
        return dict(zip(self.body_names, world))

    def tool(self, q):
        return self.poses(q)[self.body_names[self.tool_body]] @ self.tool_transform

    def jacobian(self, q):
        poses = self.poses(q)
        tool = poses[self.body_names[self.tool_body]] @ self.tool_transform
        J = np.zeros((6, self.n))
        i = self.tool_body
        while i:
            e = self.parent_edges[i]
            if e['k'] >= 0:
                joint = poses[self.body_names[e['parent']]] @ e['A']
                axis = joint[:3, :3] @ e['axis']
                if e['kind'] == 'revolute':
                    J[:3, e['k']] = np.cross(axis, tool[:3, 3] - joint[:3, 3])
                    J[3:, e['k']] = axis
                else:
                    J[:3, e['k']] = axis
            i = e['parent']
        return J

    def rnea(self, q, v, a, gravity=None):
        q, v, a = self._check(q), self._check(v), self._check(a)
        local = self.transforms(q)
        count = len(self.body_names)
        vel, acc, force = np.zeros((count, 6)), np.zeros((count, 6)), np.zeros((count, 6))
        acc[0, 3:] = -(self.gravity if gravity is None else np.asarray(gravity))
        X = [np.eye(6) for _ in self.body_names]
        for e in self.edges:
            i, p, k, S = e['i'], e['parent'], e['k'], e['S']
            X[i] = adjoint(inverse(local[i]))
            vj = S * (v[k] if k >= 0 else 0.)
            vel[i] = X[i] @ vel[p] + vj
            acc[i] = X[i] @ acc[p] + S*(a[k] if k >= 0 else 0.) + cross_motion(vel[i]) @ vj
            Iv = self.inertias[i] @ vel[i]
            force[i] = self.inertias[i] @ acc[i] - cross_motion(vel[i]).T @ Iv
        tau = np.zeros(self.n)
        for e in reversed(self.edges):
            i, p, k = e['i'], e['parent'], e['k']
            if k >= 0:
                tau[k] = e['S'] @ force[i]
            force[p] += X[i].T @ force[i]
        return tau

    def mass(self, q):
        zero = np.zeros(self.n)
        return np.column_stack([self.rnea(q, zero, col, gravity=np.zeros(3)) for col in np.eye(self.n)])

    def energy(self, q, v):
        local = self.transforms(q)
        poses = self.poses(q)
        vel = np.zeros((len(self.body_names), 6))
        kinetic = potential = 0.
        for e in self.edges:
            i, p, k = e['i'], e['parent'], e['k']
            vel[i] = adjoint(inverse(local[i])) @ vel[p] + e['S']*(v[k] if k >= 0 else 0.)
            kinetic += .5*float(vel[i] @ self.inertias[i] @ vel[i])
            T = poses[self.body_names[i]]
            potential -= self.masses[i]*float(self.gravity @ (T[:3, 3] + T[:3, :3] @ self.coms[i]))
        return kinetic + potential
