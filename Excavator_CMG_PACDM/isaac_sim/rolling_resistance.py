"""Explicit rolling resistance for soil grains in the PhysX plant (NumPy only).

Why: PhysX rigid bodies have no rolling friction. With angularDamping = 0,
spilled grains roll without slip at a constant 0.3-2.3 m/s, and the phase-8
settling test (95th percentile soil speed < 0.08 m/s) cannot pass. A
MuJoCo-plant emulation with rolling friction removed fails at the same phase
("Phase timeout: wait for deposition"). A viscous angular-damping surrogate
jams the bucket in "draw through soil" and is therefore not used.

What: the MuJoCo source grains use condim 6: Coulomb rolling friction about
the contact tangents (mu_r) and torsional friction about the contact normal
(mu_t), each at most coefficient * normal force. PhysX exposes each body's net
contact force F, so each grain receives, with n = F/|F| as the contact normal
estimate, w_n = (w.n) n and w_t = w - w_n,

    tau = -min(mu_r*|F|/max(|w_t|, w_reg), 0.5*I/dt) * w_t
          -min(mu_t*|F|/max(|w_n|, w_reg), 0.5*I/dt) * w_n

computed from the previous step's reported net contact force and the grain's
current world angular velocity w. mu_r = 0.008 m and mu_t = 0.003 m are the
MuJoCo max-combined grain/terrain coefficients from the manifest. For a grain
resting or rolling on the terrain this equals MuJoCo's contact law. It is an
approximation elsewhere: grain-grain and grain-bucket pairs use 0.006 m rolling
friction in MuJoCo (1.33x smaller), and inside a loaded pile the net force
underestimates the summed contact normal forces. w_reg regularizes the Coulomb
law near rest (0.05 m/s rolling speed) and the 0.5*I/dt cap keeps the explicit
torque from more than halving either component in one step. Only grains receive
torques; robot, terrain and joints are unchanged. Like the joint viscous losses
in wrenches.py, the torques are applied once, externally, and are recorded in
applied_wrenches.npz.
"""
from __future__ import annotations

import math

import numpy as np

OMEGA_REGULARIZATION_RAD_S = 0.5
DEFAULT_GRAIN_PREFIX = 'grain_'


def manifest_friction_coefficients(manifest):
    """MuJoCo max-combined (torsional, rolling) coefficients of grains and terrain."""
    grain_ids = {b['id'] for b in manifest['bodies'] if b['name'].startswith(DEFAULT_GRAIN_PREFIX)}
    geoms = manifest.get('geometries') or []
    grain = [g for g in geoms if g.get('body_id') in grain_ids]
    terrain = [g for g in geoms if g.get('body_id') == 0 and g.get('contype', 0)]
    if not grain:
        return 0., 0.
    torsional = max(float(g['friction'][1]) for g in grain+terrain)
    rolling = max(float(g['friction'][2]) for g in grain+terrain)
    return torsional, rolling


def manifest_rolling_friction(manifest):
    return manifest_friction_coefficients(manifest)[1]


class GrainRollingResistance:
    def __init__(self, manifest, dt, rolling_friction_m=None, omega_regularization=OMEGA_REGULARIZATION_RAD_S):
        bodies = manifest['bodies']
        self.index = np.array([k for k, b in enumerate(bodies) if b['name'].startswith(DEFAULT_GRAIN_PREFIX)],
                              dtype=np.int64)
        self.n_bodies = len(bodies)
        torsional, derived = manifest_friction_coefficients(manifest)
        if rolling_friction_m is not None and (not math.isfinite(rolling_friction_m) or rolling_friction_m < 0):
            raise ValueError('Rolling friction override must be finite and nonnegative')
        self.mu = derived if rolling_friction_m is None else float(rolling_friction_m)
        self.mu_t = torsional
        self.derived_mu = derived
        self.user_override = rolling_friction_m is not None
        self.dt = float(dt)
        self.omega_reg = float(omega_regularization)
        if (not math.isfinite(self.mu) or self.mu < 0 or not math.isfinite(self.dt) or self.dt <= 0 or
                not math.isfinite(self.omega_reg) or self.omega_reg <= 0):
            raise ValueError('Rolling friction, timestep and regularization must be finite; friction nonnegative')
        inertia = np.array([max(bodies[k]['inertia_diagonal']) for k in self.index], dtype=float)
        self.gain_cap = .5*inertia/self.dt
        mass = np.array([bodies[k]['mass'] for k in self.index], dtype=float)
        self.gravity = float(np.linalg.norm(manifest.get('gravity', [0., 0., -9.81])))
        self.weight = mass*self.gravity
        self.force = np.zeros((len(self.index), 3))
        self.enabled = bool(self.mu > 0 and len(self.index))
        self.peak_torque = 0.
        self.work = 0.
        self.steps = 0
        self.capped_steps = 0

    def update_contacts(self, net_contact_forces):
        forces = np.asarray(net_contact_forces, dtype=float)
        if forces.shape != (self.n_bodies, 3) or not np.all(np.isfinite(forces)):
            raise ValueError('Net contact forces must be finite and follow manifest body order')
        self.force = forces[self.index].copy()

    def torques(self, angular_velocities):
        """World torques about each body's COM; zero for every non-grain body."""
        w_all = np.asarray(angular_velocities, dtype=float)
        if w_all.shape != (self.n_bodies, 3) or not np.all(np.isfinite(w_all)):
            raise ValueError('Angular velocities must be finite and follow manifest body order')
        out = np.zeros_like(w_all)
        if not self.enabled:
            return out
        w = w_all[self.index]
        normal_force = np.linalg.norm(self.force, axis=1)
        axis = self.force/np.maximum(normal_force, 1e-300)[:, None]
        w_n = np.einsum('ij,ij->i', w, axis)[:, None]*axis
        w_t = w-w_n
        rolling = self.mu*normal_force/np.maximum(np.linalg.norm(w_t, axis=1), self.omega_reg)
        spin = self.mu_t*normal_force/np.maximum(np.linalg.norm(w_n, axis=1), self.omega_reg)
        self.capped_steps += int(np.any(rolling > self.gain_cap) or np.any(spin > self.gain_cap))
        out[self.index] = (-np.minimum(rolling, self.gain_cap)[:, None]*w_t
                           - np.minimum(spin, self.gain_cap)[:, None]*w_n)
        self.peak_torque = max(self.peak_torque, float(np.max(np.linalg.norm(out[self.index], axis=1), initial=0.)))
        self.steps += 1
        return out

    def account(self, torques, omega_start, omega_end):
        """Midpoint work of the applied grain torques (nonpositive when dissipative)."""
        t = np.asarray(torques, dtype=float)[self.index]
        w0 = np.asarray(omega_start, dtype=float)[self.index]
        w1 = np.asarray(omega_end, dtype=float)[self.index]
        self.work += float(np.sum(t*(.5*(w0+w1))))*self.dt

    def report(self):
        return {'applied': self.enabled,
                'model': 'regularized Coulomb rolling and torsional resistance torques on soil grains only, '
                         'from each grain\'s previous-step PhysX net contact force; robot, terrain and joints unchanged',
                'law': 'n=F/|F|; w_n=(w.n)n; w_t=w-w_n; tau=-min(mu_r|F|/max(|w_t|,w_reg),0.5I/dt)w_t'
                       '-min(mu_t|F|/max(|w_n|,w_reg),0.5I/dt)w_n',
                'rolling_friction_m': self.mu, 'derived_mujoco_rolling_friction_m': self.derived_mu,
                'torsional_friction_m': self.mu_t,
                'user_override': self.user_override, 'omega_regularization_rad_s': self.omega_reg,
                'grain_bodies': int(len(self.index)),
                'equivalent_single_grain_ground_torque_Nm': float(self.mu*np.median(self.weight)) if len(self.index) else 0.,
                'peak_torque_Nm': self.peak_torque, 'applied_torque_work_J': self.work,
                'steps_applied': self.steps, 'steps_with_stability_cap': self.capped_steps,
                'recorded_in': 'applied_wrenches.npz torques_world_about_com (grain bodies)',
                'reason': 'PhysX has no rolling friction; without it spilled grains roll indefinitely',
                'limitation': 'net-force based: equals MuJoCo for a grain on the terrain; grain-grain and '
                              'grain-bucket rolling uses 0.008 m instead of MuJoCo 0.006 m; inside a loaded pile '
                              'the net force underestimates summed normal forces; absolute, not relative, '
                              'grain angular velocity'}
