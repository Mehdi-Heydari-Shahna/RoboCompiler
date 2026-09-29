"""Source-equivalent explicit effort controllers and immutable PACDM reference."""
from pathlib import Path
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from .rigid import RigidTree


class Reference:
    def __init__(self, root, reference_path=None):
        self.path = Path(reference_path).resolve() if reference_path else Path(root)/'data/reference.npz'
        with np.load(self.path, allow_pickle=False) as z:
            self.data = {k:z[k] for k in z.files}
        r = self.data
        self.duration = float(r['time'][-1])
        self.q = BPoly.from_derivatives(r['time'], np.stack([r['q'], r['v'], r['a']], axis=1))
        self.active = BPoly.from_derivatives(r['time'], np.stack([r['active'], r['active_v'], r['active_a']], axis=1))

    def at(self, t):
        if not -1e-10 <= t <= self.duration + 1e-10:
            raise ValueError('Reference time is outside the reference task')
        t = np.clip(t, 0., self.duration)
        return self.q(t), self.q(t, nu=1), self.q(t, nu=2)

    def target(self, t):
        x = self.active(np.clip(t, 0., self.duration))
        R = Rotation.from_euler('x', np.pi).as_matrix() @ Rotation.from_euler('XYZ', x[3:6]).as_matrix()
        return x[:3], R


class Controller:
    def __init__(self, cmg, reference, mode='contact', feedforward=True, grasp=True):
        if mode not in ('contact', 'wrench'):
            raise ValueError(mode)
        self.tree = RigidTree(cmg)
        self.reference = reference
        self.mode, self.feedforward, self.grasp = mode, feedforward, grasp
        a = cmg['actuation']['actuators']
        self.kp = np.array([x['gain'][0] for x in a[:7]])
        self.kd = np.array([-x['bias'][2] for x in a[:7]])
        self.ctrl_ranges = np.array([x['ctrl_range'] for x in a[:7]])
        self.force_ranges = np.array([x['force_range'] for x in a])
        self.B = np.asarray(cmg['actuation']['moment_matrix'])
        self.armature = np.asarray(cmg['armature'])
        self.damping = np.asarray(cmg['damping'])

    def evaluate(self, t, q, v):
        qr, vr, ar = self.reference.at(t)
        ff = self.tree.rnea(q, v, ar) + self.armature*ar + self.damping*v
        if not self.feedforward:
            ff[:7] = 0.  # wrench finger servo remains unchanged, like the source
        raw_ctrl = qr[:7] + (self.kd*vr[:7] + ff[:7])/self.kp
        ctrl = np.clip(raw_ctrl, self.ctrl_ranges[:,0], self.ctrl_ranges[:,1])
        raw_effort = np.zeros(8)
        raw_effort[:7] = self.kp*(ctrl-q[:7]) - self.kd*v[:7]
        if self.mode == 'contact':
            # The source benchmark overrides actuator8 to gain 0.1568627451,
            # bias [0,-1000,-30], tendon length .5*q7+.5*q8, cap +/-100 N.
            desired = np.clip(qr[7], 0., .04) if self.grasp else .04
            raw_effort[7] = 1000.*(desired-.5*(q[7]+q[8])) - 15.*(v[7]+v[8])
        else:
            raw_effort[7] = ff[7:].sum() + 1000.*(qr[7]-q[7]) + 30.*(vr[7]-v[7])
        effort = np.clip(raw_effort, self.force_ranges[:,0], self.force_ranges[:,1])
        tau_actuator = self.B @ effort
        # No USD position/velocity drives. Source passive damping is applied
        # separately; do not clip it as if it were an actuator force.
        tau_physx = tau_actuator - self.damping*v
        return dict(q_ref=qr, v_ref=vr, a_ref=ar, ctrl=ctrl, ff=ff,
            actuator_effort=effort, torque=tau_actuator, applied_effort=tau_physx,
            arm_saturated=bool(np.any(abs(raw_effort[:7]-effort[:7])>1e-9)),
            finger_saturated=bool(abs(raw_effort[7]-effort[7])>1e-9),
            control_clipped=bool(np.any(abs(raw_ctrl-ctrl)>1e-12)))


def smooth(s):
    s = np.clip(s, 0., 1.)
    return s**3*(10.-15.*s+6.*s*s)


def wrench_at(t, mode, load_kg=.15):
    w = np.zeros(6)
    if mode == 'wrench':
        loading = smooth((t-4.5)/.5)*(1.-smooth((t-19.)/.5))
        w[2] = -9.81*load_kg*loading
        if 13. < t < 13.16:
            w[:3] += np.array([2., -1.1, 0.])*np.sin(np.pi*(t-13.)/.16)**2
        if 14.5 < t < 14.62:
            w[3:] += np.array([.008, .012, 0.])*np.sin(np.pi*(t-14.5)/.12)**2
    else:
        if 13. <= t < 13.16:
            w[:3] = [2., -1.1, 0.]
        if 14.5 <= t < 14.62:
            w[3:] = [.008, .012, 0.]
    return w
