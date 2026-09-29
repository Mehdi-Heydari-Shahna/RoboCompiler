"""MuJoCo-backed stand-in for the Isaac/PhysX APIs used by run_isaac.py.

Test harness only. PhysX-like choices: no rolling/torsional friction, no MuJoCo
joint damping (the runner applies it through wrenches), no MuJoCo actuators
(the runner applies efforts as body wrenches). Optional pin-spin drive
(FAKE_PIN_SPIN_RATE) reproduces the PhysX q22 drift.
"""
import json, os
from pathlib import Path
import numpy as np
import mujoco

WORLD = None


class World:
    def __init__(self, scene_usda):
        scene_usda = Path(scene_usda)
        self.manifest = json.loads((scene_usda.parent/'manifest.json').read_text(encoding='utf-8'))
        xml = (scene_usda.parent/self.manifest['source_scene']).resolve()
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        d.qpos[:] = np.asarray(self.manifest['initial_qpos']); d.qvel[:] = 0.
        # PhysX-like: no rolling/torsional friction, no native joint damping on scalar joints.
        m.geom_friction[:, 1:] = 0.
        for j in range(m.njnt):
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                m.dof_damping[m.jnt_dofadr[j]] = 0.
        mujoco.mj_forward(m, d)
        names = [b['name'] for b in self.manifest['bodies']]
        self.body_ids = np.array([m.body(n).id for n in names])
        self.paths = [b['path'] for b in self.manifest['bodies']]
        # Present the view in a permuted order to exercise the runner's reordering.
        self.view_order = np.r_[np.arange(1, len(names)), 0]
        self.started = False
        self.stepped = 0
        self.pin = m.body('body_59').id
        self.pin_joint = m.joint('q22').id
        self.pin_rate = float(os.environ.get('FAKE_PIN_SPIN_RATE', '0') or 0.)
        # Optional test kick: after the first step give grain_0000 this spin (rad/s).
        self.grain_kick = float(os.environ.get('FAKE_GRAIN_SPIN', '0') or 0.)
        self.xfrc = np.zeros((m.nbody, 6))

    # ---- tensor readback in *view* order
    def transforms(self):
        d = self.d
        ids = self.body_ids[self.view_order]
        q = d.xquat[ids]
        return np.c_[d.xpos[ids], q[:, 1:4], q[:, :1]].astype(np.float32)

    def velocities(self):
        out = np.zeros((len(self.body_ids), 6))
        for k, b in enumerate(self.body_ids[self.view_order]):
            v = np.zeros(6)
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, int(b), v, 0)
            out[k, :3] = v[3:]; out[k, 3:] = v[:3]
        return out.astype(np.float32)

    def apply(self, force, torque, position, indices):
        force = np.asarray(force, float); torque = np.asarray(torque, float); position = np.asarray(position, float)
        self.xfrc[:] = 0.
        for k in np.asarray(indices):
            b = self.body_ids[self.view_order[k]]
            arm = position[k]-self.d.xipos[b]
            self.xfrc[b, :3] += force[k]
            self.xfrc[b, 3:] += torque[k]+np.cross(arm, force[k])

    def step(self, dt):
        m, d = self.m, self.d
        if abs(dt-m.opt.timestep) > 1e-12:
            raise RuntimeError('fake world timestep mismatch')
        d.xfrc_applied[:] = self.xfrc
        if self.pin_rate:
            axis = d.xmat[self.pin].reshape(3, 3) @ m.jnt_axis[self.pin_joint]
            d.xfrc_applied[self.pin, 3:] += 0.005033475*(self.pin_rate-d.qvel[m.jnt_dofadr[self.pin_joint]])/0.5*axis
        mujoco.mj_step(m, d)
        if self.grain_kick and self.stepped == 0:
            j = m.body_jntadr[m.body('grain_0000').id]
            d.qvel[m.jnt_dofadr[j]+3:m.jnt_dofadr[j]+6] = [0., self.grain_kick, 0.]
        mujoco.mj_forward(m, d)  # contact forces/positions at the new state, like PhysX readback
        self.xfrc[:] = 0.
        self.stepped += 1
        if not np.all(np.isfinite(d.qpos)):
            raise RuntimeError('fake MuJoCo plant became nonfinite')

    def contact_forces(self):
        m, d = self.m, self.d
        net = np.zeros((m.nbody, 3))
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            frame = c.frame.reshape(3, 3)
            f = frame.T @ f6[:3]   # force on geom2's body from geom1 (MuJoCo convention: on body2? sign handled symmetric)
            b1 = m.geom_bodyid[c.geom1]; b2 = m.geom_bodyid[c.geom2]
            net[b1] -= f; net[b2] += f
        return net[self.body_ids[self.view_order]].astype(np.float32)


def world():
    if WORLD is None:
        raise RuntimeError('Stage not opened')
    return WORLD


def open_world(path):
    global WORLD
    WORLD = World(path)
    return WORLD
