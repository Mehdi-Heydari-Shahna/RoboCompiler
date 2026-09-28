"""Read physical states and apply power-consistent ideal prismatic forces.

No Isaac imports: SI units, tensors use actor-origin poses (xyzw), COM linear
velocities and world angular velocities. No kinematic state command exists here.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def disturbance(t):
    value = np.zeros(6)
    for start, width, pulse in [(5.5,.35,[80,0,0,0,0,0]),
                                 (11.2,.35,[0,120,0,0,0,0]),
                                 (14.2,.4,[0,0,0,12,0,0])]:
        if start <= t <= start + width:
            value += np.array(pulse) * np.sin(np.pi*(t-start)/width)
    return value


def phase(t):
    if t < 2: return 'Initial hold'
    if t < 9: return 'Helical inspection sweep'
    if t < 17: return 'Figure-eight / disturbance rejection'
    if t < 20: return 'Precision docking'
    return 'Dock hold'


class StateMapping:
    def __init__(self, cmg, names, local_coms):
        self.cmg = cmg
        self.names = list(names)
        self.idx = {n:i for i,n in enumerate(names)}
        self.local_coms = np.asarray(local_coms, dtype=float)
        self.active = np.array([cmg['coordinate_ids'].index(n) for n in cmg['independent_ids']])
        self.base = np.array(cmg['geometry']['base_anchors_m'])
        self.top = np.array(cmg['geometry']['platform_anchors_m'])
        self.R0 = np.array(cmg['geometry']['base_rotations'])

    def read(self, transforms, com_velocities):
        x = np.asarray(transforms, dtype=float)
        v = np.asarray(com_velocities, dtype=float)
        if x.shape != (len(self.names),7) or v.shape != (len(self.names),6):
            raise ValueError('Unexpected rigid-body tensor shape')
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
            raise FloatingPointError('Non-finite PhysX rigid-body state')
        R = Rotation.from_quat(x[:,3:]).as_matrix()
        p = x[:,:3]
        arm = np.einsum('nij,nj->ni',R,self.local_coms)
        c = p + arm
        w = v[:,3:]
        vo = v[:,:3] - np.cross(w,arm)
        q = np.zeros(24); qd = np.zeros(24)
        ip = self.idx['platform']
        q[:3] = p[ip]
        q[3:6] = Rotation.from_matrix(R[ip]).as_euler('ZYX')
        qd[:3] = vo[ip]
        yaw, pitch, roll = q[3:6]
        Rz = Rotation.from_euler('Z',yaw).as_matrix()
        Rzy = Rotation.from_euler('ZY',[yaw,pitch]).as_matrix()
        angular_map = np.column_stack(([0,0,1],Rz[:,1],Rzy[:,0]))
        qd[3:6] = np.linalg.solve(angular_map,w[ip])
        joint_pos = []; joint_ang = []; closure = []
        for i in range(6):
            iy, ib, ir = [self.idx[f'leg_{i}_{n}'] for n in ('yoke','barrel','rod')]
            Ry = self.R0[i].T @ R[iy]
            Rb = R[iy].T @ R[ib]
            qx = np.arctan2(Ry[2,1],Ry[1,1])
            qy = np.arctan2(Rb[0,2],Rb[0,0])
            axis = R[ib,:,2]
            d = p[ir] - p[ib]
            length = axis @ d
            ld = axis @ (vo[ir]-vo[ib]) + np.cross(w[ib],axis) @ d
            j=6+3*i
            q[j:j+3] = [qx,qy,length]
            qd[j:j+3] = [self.R0[i,:,0] @ w[iy],R[iy,:,1] @ (w[ib]-w[iy]),ld]
            joint_pos.extend([np.linalg.norm(p[iy]-self.base[i]),
                              np.linalg.norm(p[ib]-p[iy]),
                              np.linalg.norm(d-length*axis)])
            joint_ang.extend([Rotation.from_matrix(Rotation.from_euler('X',qx).as_matrix().T@Ry).magnitude(),
                              Rotation.from_matrix(Rotation.from_euler('Y',qy).as_matrix().T@Rb).magnitude(),
                              Rotation.from_matrix(R[ib].T@R[ir]).magnitude()])
            closure.append(np.linalg.norm(p[ir]-(p[ip]+R[ip]@self.top[i])))
        return dict(q=q,velocity=qd,p=p,R=R,com=c,com_velocities=v,
                    closure_error_m=float(max(closure)),
                    joint_error_m=float(max(joint_pos)),joint_error_rad=float(max(joint_ang)))

    def forces(self, state, actuator_force, external_wrench):
        """Equal/opposite forces at a shared point; return world COM wrenches.

        Using the rod origin as the common point makes virtual power exactly
        f*d/dt[u_barrel dot (p_rod-p_barrel)], including small joint drift.
        """
        f = np.zeros((len(self.names),3)); tau = np.zeros_like(f)
        for i, effort in enumerate(actuator_force):
            ib, ir = [self.idx[f'leg_{i}_{n}'] for n in ('barrel','rod')]
            applied = float(effort) * state['R'][ib,:,2]
            point = state['p'][ir]
            for index, force in [(ir,applied),(ib,-applied)]:
                f[index] += force
                tau[index] += np.cross(point-state['com'][index],force)
        ip = self.idx['platform']; wrench=np.asarray(external_wrench)
        f[ip] += wrench[:3]
        tau[ip] += wrench[3:] + np.cross(state['p'][ip]-state['com'][ip],wrench[:3])
        return f,tau
