"""Independent, force-driven six-UPS Newton-Euler model, NumPy/SciPy only.

This is a benchmark-specific verification model, NOT Isaac Sim, MuJoCo, or an
alternative CMG/PACDM compiler. Platform position and world angular velocity are
integrated dynamically; exact six-UPS geometry determines the passive links.
Body Jacobians are derived directly from the anchor geometry, independently of
PointGraph/PACDM. All 19 physical rigid bodies (with merged fixed payload) enter
the mass and bias. There is no massless-leg approximation or reference replay.

Generalized velocity is [world platform-origin linear velocity, world angular
velocity]. M = sum(m Jv.T Jv + Jw.T Iworld Jw); bias is the corresponding
Newton-Euler body sum including gravity, Jdot*v and gyroscopic torques. Jdot is
computed by symmetric directional differences. World SO(3) updates use exp.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    v = np.asarray(v, float)
    s = np.zeros(v.shape[:-1] + (3, 3))
    x, y, z = np.moveaxis(v, -1, 0)
    s[..., 0, 1] = -z; s[..., 0, 2] = y
    s[..., 1, 0] = z; s[..., 1, 2] = -x
    s[..., 2, 0] = -y; s[..., 2, 1] = x
    return s


def axis_rotations(angle, axis):
    angle = np.asarray(angle)
    out = np.broadcast_to(np.eye(3), angle.shape+(3,3)).copy()
    i, j = ((1,2) if axis == 'x' else (2,0))
    c, s = np.cos(angle), np.sin(angle)
    out[...,i,i] = c; out[...,j,j] = c
    out[...,i,j] = -s; out[...,j,i] = s
    return out


class RigidStewart:
    def __init__(self, cmg):
        self.cmg = cmg
        geometry = cmg['geometry']
        self.base = np.asarray(geometry['base_anchors_m'], float)
        self.top = np.asarray(geometry['platform_anchors_m'], float)
        self.R0 = np.asarray(geometry['base_rotations'], float)
        self.names = ['platform'] + [f'leg_{i}_{part}' for i in range(6)
                                     for part in ('yoke','barrel','rod')]
        source = {b['id']: b for b in cmg['bodies']}
        props = {name: (float(source[name]['mass_kg']), np.asarray(source[name]['com_m'],float),
                         np.asarray(source[name]['inertia_kg_m2'],float)) for name in self.names}
        joint = next(j for j in cmg['joints'] if j['id'] == 'payload_mount')
        if joint['type'] != 'fixed' or joint['base_body'] != 'platform':
            raise ValueError('CPU verification requires the supplied fixed payload mount')
        T = np.asarray(joint['T_BJ']) @ np.linalg.inv(joint['T_FJ'])
        payload = source['payload']
        m0, c0, I0 = props['platform']
        m1 = float(payload['mass_kg'])
        c1 = T[:3,:3] @ payload['com_m'] + T[:3,3]
        I1 = T[:3,:3] @ np.asarray(payload['inertia_kg_m2']) @ T[:3,:3].T
        m = m0+m1; c = (m0*c0+m1*c1)/m
        I = I0+I1
        for mass, center in ((m0,c0),(m1,c1)):
            d = center-c; I += mass*((d@d)*np.eye(3)-np.outer(d,d))
        props['platform'] = (m,c,I)
        self.mass = np.array([props[n][0] for n in self.names])
        self.com = np.array([props[n][1] for n in self.names])
        self.inertia = np.array([props[n][2] for n in self.names])
        self.gravity = np.asarray(cmg['gravity_m_s2'], float)

    def kinematics(self, p, R):
        arms = self.top @ R.T
        tips = p + arms
        d = tips-self.base
        length = np.linalg.norm(d, axis=1)
        if np.min(length) < 1e-8:
            raise ValueError('Degenerate zero-length leg')
        u = d/length[:,None]
        ul = np.einsum('nji,nj->ni', self.R0, u)
        den = ul[:,1]**2+ul[:,2]**2
        if np.min(den) < 1e-8:
            raise ValueError('Universal joint chart singularity')
        alpha = np.arctan2(-ul[:,1],ul[:,2])
        beta = np.arcsin(np.clip(ul[:,0],-1.,1.))
        Ry = self.R0 @ axis_rotations(alpha,'x')
        Rb = Ry @ axis_rotations(beta,'y')
        Jtip = np.empty((6,3,6))
        Jtip[:,:,:3] = np.eye(3); Jtip[:,:,3:] = -skew(arms)
        U = (np.eye(3)-u[:,:,None]*u[:,None,:])/length[:,None,None]
        local_derivative = self.R0.transpose(0,2,1) @ U @ Jtip
        ga = np.stack((np.zeros(6),-ul[:,2]/den,ul[:,1]/den),axis=1)
        gb = np.zeros((6,3)); gb[:,0] = 1/np.sqrt(den)
        Ja = np.einsum('ni,nij->nj',ga,local_derivative)
        Jb = np.einsum('ni,nij->nj',gb,local_derivative)
        JL = np.einsum('ni,nij->nj',u,Jtip)
        Jwy = self.R0[:,:,0,None]*Ja[:,None,:]
        Jwb = Jwy+Ry[:,:,1,None]*Jb[:,None,:]
        rotations = np.empty((19,3,3)); positions = np.empty((19,3))
        Jw = np.zeros((19,3,6)); Jo = np.zeros_like(Jw)
        rotations[0] = R; positions[0] = p
        Jw[0,:,3:] = np.eye(3); Jo[0,:,:3] = np.eye(3)
        rotations[1::3] = Ry; rotations[2::3] = Rb; rotations[3::3] = Rb
        positions[1::3] = self.base; positions[2::3] = self.base; positions[3::3] = tips
        Jw[1::3] = Jwy; Jw[2::3] = Jwb; Jw[3::3] = Jwb
        Jo[3::3] = Jtip
        com_arm = np.einsum('bij,bj->bi',rotations,self.com)
        Jv = Jo-skew(com_arm) @ Jw
        return dict(p=positions,R=rotations,com=positions+com_arm,Jv=Jv,Jw=Jw,
                    JL=JL,Ja=Ja,Jb=Jb,length=length,alpha=alpha,beta=beta)

    def derivatives(self, p, R, velocity):
        h = 1e-5/max(1.,float(np.linalg.norm(velocity)))
        plus = self.kinematics(p+h*velocity[:3],Rotation.from_rotvec(h*velocity[3:]).as_matrix() @ R)
        minus = self.kinematics(p-h*velocity[:3],Rotation.from_rotvec(-h*velocity[3:]).as_matrix() @ R)
        return {k:(plus[k]-minus[k])/(2*h) for k in ('Jv','Jw','JL')}

    def terms(self, p, R, velocity, kin=None):
        k = self.kinematics(p,R) if kin is None else kin
        Jv,Jw = k['Jv'],k['Jw']
        I = k['R'] @ self.inertia @ k['R'].transpose(0,2,1)
        M = np.einsum('bik,b,bil->kl',Jv,self.mass,Jv) + np.einsum('bik,bij,bjl->kl',Jw,I,Jw)
        if np.linalg.norm(velocity) > 1e-14:
            deriv = self.derivatives(p,R,velocity)
            a0 = deriv['Jv'] @ velocity; alpha0 = deriv['Jw'] @ velocity
        else:
            a0 = np.zeros((19,3)); alpha0 = np.zeros((19,3))
        omega = Jw @ velocity
        force = self.mass[:,None]*(a0-self.gravity)
        torque = (I @ alpha0[...,None])[...,0] + np.cross(omega,(I @ omega[...,None])[...,0])
        bias = np.einsum('bij,bi->j',Jv,force)+np.einsum('bij,bi->j',Jw,torque)
        if not np.all(np.isfinite(M)) or not np.all(np.isfinite(bias)):
            raise FloatingPointError('Nonfinite Newton-Euler dynamics')
        return M,bias

    @staticmethod
    def body_arrays(kin, velocity):
        poses = np.column_stack((kin['p'],Rotation.from_matrix(kin['R']).as_quat()))
        velocities = np.column_stack((kin['Jv']@velocity,kin['Jw']@velocity))
        return poses,velocities
