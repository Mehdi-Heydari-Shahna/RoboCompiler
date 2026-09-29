"""Read-only native-state bridge to the accepted PACDM/hydraulic controller.

The only mechanical integrator is the client (PhysX). This module does not call
mj_forward, mj_step, inverse dynamics or a MuJoCo contact solver. MuJoCo is used
for the compiled model layout and kinematic caches only. Every request contains
CURRENT native body poses and velocities. The joint readout is a measurement
projection; its off-axis errors are returned and never written to PhysX.

Wire convention: bodies 1..nbody-1, world positions, wxyz unit quaternions,
world linear velocities AT BODY ORIGINS (not COM), world angular velocities.
Free-joint angular velocity is rotated into the current body frame. Acceleration
is a backward difference of observed generalized velocity; initial acceleration
is explicitly unknown and initialized to zero. Gravity remains separately in
the source dynamics model; it is NOT added to measured kinematic acceleration.

A tick at t returns effort to HOLD over [t,t+dt). On the next consecutive tick,
the previous hydraulic transaction is advanced exactly once. Duplicate/skipped
ticks are rejected before any hydraulic state is committed.

Passive pin-spin gauge: the seventh
independent coordinate q22 is the spin of the axisymmetric cross pin body_59
about its own axis. It is never commanded, it changes no other tree coordinate,
and the controller's mass matrix, bias forces and PACDM maps are invariant to
its absolute angle. In PhysX this free spin can creep monotonically past the
source's 1.5 rad numerical branch guard (not a hardware limit). The controller
therefore receives q22 at its original
nominal value; the measured angle and velocity are still read from the plant,
the q22 velocity is passed unchanged, nothing is written to PhysX, and the
measured drift is reported. The symmetry is verified numerically at start-up
and the bridge refuses to run if it does not hold.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def rotations(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[:, [1, 2, 3, 0]]).as_matrix()


# Uncommanded independent coordinates that are verified at start-up to be
# ignorable (cyclic) spins of an axisymmetric pin. See module docstring.
PASSIVE_SPIN_COORDINATES = ('q22',)
SPIN_GAUGE_TOLERANCE = 1e-8


class ControllerStateView:
    """Controller-facing copy of the measured state; the plant is never written.

    TrackedArmController reads only qpos, qvel, qacc and time. Verified pin-spin
    coordinates are replaced by their canonical nominal angle in this copy.
    """
    __slots__ = ('qpos', 'qvel', 'qacc', 'time')

    def __init__(self, data, gauge=()):
        self.qpos = np.array(data.qpos, dtype=float, copy=True)
        self.qvel = np.array(data.qvel, dtype=float, copy=True)
        self.qacc = np.array(data.qacc, dtype=float, copy=True)
        self.time = float(data.time)
        for address, value in gauge:
            self.qpos[address] = value


def _relative_difference(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float('inf')
    scale = max(1., float(np.max(np.abs(a), initial=0.)), float(np.max(np.abs(b), initial=0.)))
    return float(np.max(np.abs(a-b), initial=0.))/scale


def kinematic_observation(m, d):
    """Offline/reference utility: return body-origin world velocities correctly."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    # XBODY means body origin; BODY would mean inertial COM and is incorrect here.
    vel = np.empty((m.nbody - 1, 6))
    for b in range(1, m.nbody):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel[b - 1], 0)
    return dict(positions=d.xpos[1:].copy(), quaternions_wxyz=d.xquat[1:].copy(),
                linear_velocities=vel[:, 3:].copy(), angular_velocities=vel[:, :3].copy())


class NativeStateMapper:
    """Single free/hinge/slide joint per body; exact for this source's tree."""
    def __init__(self, model, initial_qpos):
        self.m = model
        self.d = mujoco.MjData(model)
        self.initial_qpos = self._finite(initial_qpos, (model.nq,), 'initial_qpos')
        self.d.qpos[:] = self.initial_qpos
        self.previous_qpos = self.initial_qpos.copy()
        self.previous_qvel = None
        self.previous_t = None
        self.body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                           for i in range(1, model.nbody)]
        if any(name is None for name in self.body_names):
            raise ValueError('All non-world bodies must have names')
        if np.any(model.body_jntnum > 1):
            raise ValueError('Bridge requires at most one joint per body')
        permitted = [int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE),
                     int(mujoco.mjtJoint.mjJNT_SLIDE)]
        if np.any(~np.isin(model.jnt_type, permitted)):
            raise ValueError('Unsupported source joint type; no silent conversion')
        self.R0 = rotations(model.body_quat)
        self.free_bodies = [b for b in range(1, model.nbody) if model.body_jntnum[b] and
                            model.jnt_type[model.body_jntadr[b]] == int(mujoco.mjtJoint.mjJNT_FREE)]
        self.last = {}

    @staticmethod
    def _finite(value, shape, label):
        a = np.asarray(value, dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'{label} requires finite shape {shape}, got {a.shape}')
        return a.copy()

    def update(self, t, positions, quaternions_wxyz, linear_velocities, angular_velocities):
        m, d = self.m, self.d
        if not np.isfinite(t) or (self.previous_t is not None and t <= self.previous_t):
            raise ValueError('Observation time must be finite and strictly increasing')
        n = m.nbody - 1
        p = np.vstack((np.zeros(3), self._finite(positions, (n, 3), 'positions')))
        q = np.vstack(([1., 0., 0., 0.], self._finite(quaternions_wxyz, (n, 4), 'quaternions_wxyz')))
        norms = np.linalg.norm(q, axis=1)
        if np.max(abs(norms - 1.)) > 1e-3:
            raise ValueError('Observed quaternions are not unit length within 1e-3')
        q /= norms[:, None]
        R = rotations(q)
        v = np.vstack((np.zeros(3), self._finite(linear_velocities, (n, 3), 'linear_velocities')))
        w = np.vstack((np.zeros(3), self._finite(angular_velocities, (n, 3), 'angular_velocities')))
        qpos = self.previous_qpos.copy()
        qvel = np.zeros(m.nv)
        dt = 0. if self.previous_t is None else t - self.previous_t
        local_position_residual = local_rotation_residual = 0.
        for b in range(1, m.nbody):
            parent = int(m.body_parentid[b])
            if not m.body_jntnum[b]:
                target_R = R[parent] @ self.R0[b]
                target_p = p[parent] + R[parent] @ m.body_pos[b]
            else:
                j = int(m.body_jntadr[b]); qi = int(m.jnt_qposadr[j]); vi = int(m.jnt_dofadr[j])
                typ = int(m.jnt_type[j]); axis = m.jnt_axis[j]
                if typ == int(mujoco.mjtJoint.mjJNT_FREE):
                    if parent != 0:
                        raise ValueError('Nested free joint is unsupported')
                    quat = q[b].copy()
                    if np.dot(quat, self.previous_qpos[qi+3:qi+7]) < 0:
                        quat = -quat
                    qpos[qi:qi+7] = np.r_[p[b], quat]
                    qvel[vi:vi+6] = np.r_[v[b], R[b].T @ w[b]]
                    continue
                axis_world = R[parent] @ self.R0[b] @ axis
                relative_rotation = self.R0[b].T @ R[parent].T @ R[b]
                if typ == int(mujoco.mjtJoint.mjJNT_HINGE):
                    # Swing-twist decomposition gives the measured hinge twist
                    # even with nonzero native solver off-axis residual.
                    quat = Rotation.from_matrix(relative_rotation).as_quat()
                    angle = 2. * np.arctan2(np.dot(quat[:3], axis), quat[3])
                    raw = m.qpos0[qi] + angle
                    predictor = self.previous_qpos[qi]
                    if dt and self.previous_qvel is not None:
                        predictor += dt * self.previous_qvel[vi]
                    qpos[qi] = raw + 2*np.pi * round((predictor - raw)/(2*np.pi))
                    qvel[vi] = np.dot(axis_world, w[b] - w[parent])
                    delta_R = Rotation.from_rotvec(axis*(qpos[qi]-m.qpos0[qi])).as_matrix()
                    target_R = R[parent] @ self.R0[b] @ delta_R
                    target_p = p[parent] + R[parent] @ (m.body_pos[b] +
                               self.R0[b] @ (m.jnt_pos[j] - delta_R @ m.jnt_pos[j]))
                else:
                    base_position = p[parent] + R[parent] @ m.body_pos[b]
                    displacement = np.dot(axis_world, p[b] - base_position)
                    qpos[qi] = m.qpos0[qi] + displacement
                    # Remove parent-origin translation and parent rigid rotation.
                    qvel[vi] = np.dot(axis_world, v[b]-v[parent]-np.cross(w[parent], p[b]-p[parent]))
                    target_R = R[parent] @ self.R0[b]
                    target_p = base_position + axis_world*displacement
            local_position_residual = max(local_position_residual, float(np.linalg.norm(p[b]-target_p)))
            local_rotation_residual = max(local_rotation_residual,
                                          float(Rotation.from_matrix(target_R.T @ R[b]).magnitude()))
        acceleration = np.zeros(m.nv) if self.previous_qvel is None else (qvel-self.previous_qvel)/dt
        d.qpos[:] = qpos; d.qvel[:] = qvel; d.qacc[:] = acceleration; d.time = float(t)
        # Kinematic calls only. Do not replace this with mj_forward: that would
        # introduce a second contact solver and fabricated controller base qacc.
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d)
        tree_position_error = float(np.max(np.linalg.norm(d.xpos[1:]-p[1:], axis=1), initial=0.))
        tree_rotation_error = float(np.max(Rotation.from_matrix(
            np.einsum('nji,njk->nik', d.xmat[1:].reshape(-1,3,3), R[1:])).magnitude(), initial=0.))
        # The legacy travel controller reads these body observations. Expose the
        # native positions/rotations and angular velocities, not tree projections.
        # cvel linear channels are not used by that controller and remain caches.
        d.xpos[:] = p; d.xquat[:] = q; d.xmat[:] = R.reshape(-1,9); d.cvel[:, :3] = w
        d.qacc[:] = acceleration
        self.previous_qpos = qpos.copy(); self.previous_qvel = qvel.copy(); self.previous_t = float(t)
        self.last = dict(local_joint_position_error_m=local_position_residual,
                         local_joint_rotation_error_rad=local_rotation_residual,
                         tree_position_error_m=tree_position_error, tree_rotation_error_rad=tree_rotation_error,
                         initial_acceleration_unknown=dt == 0., acceleration_method='backward difference of measured generalized velocity',
                         native_state_projection=False)
        self.observation = dict(positions=p, rotations=R, linear_velocities=v, angular_velocities=w)
        return self.last


class ControllerBridge:
    def __init__(self, source_root, scene_xml, initial_qpos, dt=.001, case='soil_final', mode='mission'):
        self.source_root = Path(source_root).resolve()
        self.scene_xml = Path(scene_xml).resolve()
        if case not in ('soil_demo', 'soil_final', 'face_empty', 'face_no_soil_contact') or mode not in ('mission', 'zero_effort'):
            raise ValueError('Unsupported source case or mode; use packaged soil_demo/soil_final/face_empty/face_no_soil_contact and mission/zero_effort')
        self.dt = float(dt)
        if (not np.isfinite(dt) or dt <= 0 or
                abs(round(.001/dt)*dt-.001) > 1e-12):
            raise ValueError('dt must exactly divide the 1 ms inner servo period')
        self.case, self.mode = case, mode
        self.m = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        if self.m.nu != 10:
            raise ValueError('This adapter expects the original ten effort ports')
        self.mapper = NativeStateMapper(self.m, initial_qpos); self.d = self.mapper.d
        self.step = 0; self.pending = None; self.completed_steps = 0
        self.totals = {'arm': {}, 'travel': {}, 'actuator_midpoint_work_J': 0.,
                       'actuator_held_speed_work_J': 0.}
        self.ai_all = np.arange(self.m.nu)
        self.actuator_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in self.ai_all]
        if (np.any(self.m.actuator_trntype != int(mujoco.mjtTrn.mjTRN_JOINT)) or
                not np.allclose(self.m.actuator_gear[:,0], 1.) or
                np.any(self.m.actuator_gear[:,1:])):
            raise ValueError('Expected direct scalar unit-geared joint effort actuators')
        self.port_jids = self.m.actuator_trnid[:, 0].astype(int)
        self.port_qi = self.m.jnt_qposadr[self.port_jids]
        self.port_vi = self.m.jnt_dofadr[self.port_jids]
        self.initial_qpos = np.asarray(initial_qpos).copy()
        self.last_metrics = {}
        self.ready = False
        self.finalized = False
        self.spin_gauge = ()
        self.passive_spin_gauge = None
        if mode == 'mission':
            for path in (self.source_root, self.source_root/'v26', self.source_root/'v26'/'source'/'Excavator_RoboIR_full_body_v21'):
                sys.path.insert(0,str(path))
            from benchmark_model import context, PORTS
            from soil_mission import SoilMission
            from soil_scene import BOUNDS
            from digging_export import bucket_inside
            from digging_path import LIP
            from tracked_arm import TrackedArmController
            from mobile_mission import TravelController
            from hydraulic_actuators import HydraulicBank
            from track_drive import TrackDriveBank
            self.context = context()
            self.cmg, self.mapping, self.engine, self.a, self.reference, self.inverse = self.context
            recipe = json.loads((self.source_root/'RUN_RECIPES.json').read_text())['cases'][case]
            self.mission = SoilMission(self.engine, self.reference, cut_depth=recipe['cut_depth'], cut_policy=recipe['cut_policy'])
            jids = np.array([self.m.joint(k).id for k in self.engine.tree_ids])
            self.qi = self.m.jnt_qposadr[jids]; self.vi = self.m.jnt_dofadr[jids]
            if np.max(abs(self.initial_qpos[self.qi]-self.mission.initial_tree)) > 1e-8:
                raise ValueError('Initial arm state does not match chosen mission initial state')
            self.pi = np.array([self.engine.tree_ids.index(k) for k in PORTS])
            self.ai = np.array([self.m.actuator('effort_'+k).id for k in PORTS])
            self.meta = json.loads((self.scene_xml.parent/'scene_metadata.json').read_text())
            self.arm_class = TrackedArmController; self.travel_class = TravelController
            self.bank_class = HydraulicBank; self.drive = TrackDriveBank()
            self.bounds = BOUNDS; self.bucket_inside = bucket_inside; self.lip_local = LIP
            self.grains = np.array([i for i in range(1,self.m.nbody) if self.m.body(i).name.startswith('grain_')], dtype=int)
            self.masses = self.m.body_mass[self.grains].copy()
            self.ever_lifted = np.zeros(len(self.grains),bool); self.deposited_dwell = np.zeros(len(self.grains))
            self.bucket = self.m.body('body_56').id
            self.sample_every = round(.04/dt); self.arm_every = round(.01/dt); self.servo_every = round(.001/dt)
            self.last_command = np.zeros(2)
            self._configure_passive_spin_gauge(PORTS)

    def _configure_passive_spin_gauge(self, ports):
        """Verify, then gauge-fix, uncommanded pin-spin coordinates.

        Checks, each with tolerance SPIN_GAUGE_TOLERANCE:
        1. the coordinate is not an effort port and is not one of the six
           commanded independent coordinates;
        2. its PACDM/ideal tangent column is the unit vector on itself, i.e. a
           spin changes no other tree coordinate;
        3. the controller's arm-row mass matrix, bias forces, tangent maps and
           energies, and the floating-base oracle, are unchanged by a finite
           spin of half the numerical guard;
        4. in the plant model the spinning body has its COM on the joint axis
           and an inertia tensor that is axisymmetric about that axis.
        Any failure raises before the first native step.
        """
        from floating_dynamics import FloatingSource
        e, r = self.engine, self.reference
        gravity = np.asarray(self.m.opt.gravity, dtype=float).copy()
        floating = FloatingSource(self.cmg, e.source)
        independent = list(e.independent_ids)
        velocity = np.linspace(-.3, .3, len(independent))
        base_velocity = np.array([.05, -.04, .03, .02, -.01, .015])
        records, gauge = [], []
        for name in PASSIVE_SPIN_COORDINATES:
            if name in ports:
                raise ValueError(f'{name} is an effort port and cannot be gauge fixed')
            k = independent.index(name)
            if k < 6:
                raise ValueError(f'{name} is a commanded independent coordinate and cannot be gauge fixed')
            tree = list(e.tree_ids).index(name)
            canonical = float(r.nominal[tree])
            nominal = np.asarray(r.nominal, dtype=float).copy()
            shift = .5*float(r.guards['independent_rad'])
            shifted_q = nominal.copy(); shifted_q[tree] += shift
            first = r.component(nominal, velocity, gravity)
            second = r.component(shifted_q, velocity, gravity)
            unit = np.zeros(len(nominal)); unit[tree] = 1.
            tangent_error = float(np.max(np.abs(np.asarray(first['tangent_map'])[:, k]-unit)))
            compared = [key for key in ('mass_matrix', 'bias_forces', 'tangent_map', 'tangent_map_dot',
                                        'curvature', 'jacobian', 'jacobian_dot', 'residual', 'constraint_rank',
                                        'jdot_velocity', 'velocity', 'kinetic_energy', 'potential_energy')
                        if key in first and key in second]
            component_error = max(_relative_difference(first[key], second[key]) for key in compared)
            fa = floating.evaluate(nominal, np.zeros(3), np.eye(3), np.r_[base_velocity, first['velocity']], gravity)
            fb = floating.evaluate(shifted_q, np.zeros(3), np.eye(3), np.r_[base_velocity, second['velocity']], gravity)
            floating_error = max(_relative_difference(fa[key], fb[key]) for key in
                                 ('mass_matrix', 'bias_forces', 'kinetic_energy', 'potential_energy'))
            jid = self.m.joint(name).id
            body = int(self.m.jnt_bodyid[jid])
            axis = np.asarray(self.m.jnt_axis[jid], dtype=float)
            axis = axis/np.linalg.norm(axis)
            arm = np.asarray(self.m.body_ipos[body])-np.asarray(self.m.jnt_pos[jid])
            com_offset = float(np.linalg.norm(arm-np.dot(arm, axis)*axis))
            principal = rotations(np.asarray(self.m.body_iquat[body])[None])[0]
            inertia = principal @ np.diag(self.m.body_inertia[body]) @ principal.T
            scale = max(float(np.max(self.m.body_inertia[body])), 1e-12)
            axis_not_principal = float(np.linalg.norm(inertia @ axis-np.dot(axis, inertia @ axis)*axis))/scale
            transverse = np.linalg.eigvalsh((np.eye(3)-np.outer(axis, axis)) @ inertia @ (np.eye(3)-np.outer(axis, axis)))
            transverse_asymmetry = float(abs(transverse[2]-transverse[1]))/scale
            checks = dict(tangent_column_unit_error=tangent_error,
                          arm_component_relative_change=component_error,
                          floating_oracle_relative_change=floating_error,
                          plant_com_distance_from_axis_m=com_offset,
                          plant_axis_principal_error=axis_not_principal,
                          plant_transverse_inertia_asymmetry=transverse_asymmetry)
            failed = [key for key, value in checks.items()
                      if not np.isfinite(value) or value > SPIN_GAUGE_TOLERANCE]
            if failed:
                raise ValueError(f'{name} is not a verified ignorable pin spin; failed checks: '
                                 + ', '.join(f'{key}={checks[key]:.3e}' for key in failed))
            address = int(self.m.jnt_qposadr[jid])
            gauge.append((address, canonical))
            records.append(dict(joint=name, body=self.m.body(body).name, independent_index=k,
                                qpos_address=address, controller_angle_rad=canonical,
                                verification_shift_rad=shift, tolerance=SPIN_GAUGE_TOLERANCE,
                                checks=checks, last_measured_offset_rad=0.,
                                max_abs_measured_offset_rad=0.))
        self.spin_gauge = tuple(gauge)
        self.passive_spin_gauge = dict(
            method='controller input uses the original nominal angle of verified ignorable pin spins',
            reason='q22 is the free spin of the axisymmetric cross pin body_59; its absolute angle has no '
                   'kinematic or dynamic effect, but the source numerical guard limits it to 1.5 rad',
            plant_state_written=False, velocity_modified=False, coordinates=records)

    def _controller_state(self):
        """Measured state for the arm controller, with only the verified gauge applied."""
        view = ControllerStateView(self.d, self.spin_gauge)
        if self.passive_spin_gauge is not None:
            for (address, canonical), record in zip(self.spin_gauge, self.passive_spin_gauge['coordinates']):
                offset = float(self.d.qpos[address]-canonical)
                record['last_measured_offset_rad'] = offset
                record['max_abs_measured_offset_rad'] = max(record['max_abs_measured_offset_rad'], abs(offset))
        return view

    def _spin_gauge_report(self):
        if self.passive_spin_gauge is None:
            return {}
        return {record['joint']: dict(measured_rad=float(self.d.qpos[address]), controller_rad=canonical,
                                      measured_offset_rad=float(self.d.qpos[address]-canonical))
                for (address, canonical), record in zip(self.spin_gauge, self.passive_spin_gauge['coordinates'])}

    def metadata(self):
        return dict(protocol='excavator_native_body_v1', body_names=self.mapper.body_names,
                    body_ids=list(range(1,self.m.nbody)),
                    actuator_names=self.actuator_names, joint_names=[self.m.joint(int(i)).name for i in self.port_jids],
                    dt=self.dt, case=self.case, mode=self.mode, coordinate_units='SI; radians; wxyz quaternions',
                    velocity_origin='body origin, world frame', native_plant='client PhysX',
                    mujoco_role='read-only compiled model and kinematics; never stepped',
                    initial_acceleration='unknown, initialized to zero; gravity separate',
                    hydraulic_commit='previous effort interval committed when next tick confirms elapsed dt',
                    passive_spin_gauge=jsonable(self.passive_spin_gauge))

    def _commit(self):
        if self.pending is None:
            return
        old = self.pending
        self.bank.advance(old['arm'],self.dt); self.drive.advance(old['travel'],self.dt)
        for bank_name in ('arm','travel'):
            for name,value in old[bank_name]['power'].items():
                self.totals[bank_name][name] = self.totals[bank_name].get(name,0.) + float(value)*self.dt
        speed = self.d.qvel[self.port_vi]
        self.totals['actuator_midpoint_work_J'] += float(old['effort'] @ (.5*(old['speed']+speed)))*self.dt
        self.totals['actuator_held_speed_work_J'] += float(old['effort'] @ old['speed'])*self.dt
        self.completed_steps += 1
        self.pending = None

    def _metrics(self,t,desired):
        d = self.d; obs=self.mapper.observation; xyz=obs['positions'][self.grains]
        speed=np.linalg.norm(obs['linear_velocities'][self.grains],axis=1)
        bpos=obs['positions'][self.bucket]; R=obs['rotations'][self.bucket]
        inside=self.bucket_inside((xyz-bpos)@R) if len(xyz) else np.zeros(0,bool)
        self.ever_lifted |= inside & (xyz[:,2]>.5)
        receiver=np.all(abs(xyz[:,:2]-self.mission.depot_center[:2])<self.mission.depot_half_size,axis=1)
        receiver &= (xyz[:,2]<.55)&(~inside)&self.ever_lifted&(speed<.12)
        sample_interval = .04 if not self.last_metrics else t-self.last_metrics['time_s']
        self.deposited_dwell=np.where(receiver,self.deposited_dwell+sample_interval,0.)
        deposited=receiver&(self.deposited_dwell>=.5)
        in_source=np.all((xyz>=self.bounds[:,0])&(xyz<=self.bounds[:,1]+[0,0,.12]),axis=1)&~inside&~deposited
        spilled=~(inside|deposited|in_source)
        bmass=float(self.masses[inside].sum()); depmass=float(self.masses[deposited].sum())
        tracking=float(np.max(abs(d.qpos[self.qi][self.engine.active[:6]]-desired[0][:6])))
        lip=bpos+R@self.lip_local
        soil_speed=float(np.quantile(speed,.95)) if len(speed) else 0.
        self.last_metrics=dict(time_s=float(t),phase=int(self.mission.phase), phase_name=self.mission.names[self.mission.phase],
                              bucket_mass_kg=bmass,deposited_mass_kg=depmass,lifted_mass_kg=float(self.masses[self.ever_lifted].sum()),
                              source_mass_kg=float(self.masses[in_source].sum()),spill_mass_kg=float(self.masses[spilled].sum()),
                              soil_speed_m_s=soil_speed,lip_position_m=lip.tolist(),tracking_error_rad=tracking)
        self.mission.update(t,tracking,soil_speed,bmass,float(lip[2]),depmass,False,lip_x=float(lip[0]))
        self.last_metrics.update(done=self.mission.done,failures=self.mission.failures.copy(),events=self.mission.events.copy())
        return self.last_metrics

    def tick(self,t,positions,quaternions_wxyz,linear_velocities,angular_velocities):
        if self.finalized:
            raise ValueError('Bridge already finalized')
        if abs(float(t)-self.step*self.dt)>max(1e-10,self.dt*1e-7):
            raise ValueError(f'Expected consecutive tick t={self.step*self.dt:.12g}, received {t}')
        mapping=self.mapper.update(float(t),positions,quaternions_wxyz,linear_velocities,angular_velocities)
        if self.mode=='zero_effort':
            self.completed_steps=self.step
            self.step+=1
            return dict(efforts=[0.]*self.m.nu,mapping=mapping,metrics={'done':False},hydraulics={},
                        port_positions=self.d.qpos[self.port_qi].tolist(),port_velocities=self.d.qvel[self.port_vi].tolist())
        if not self.ready:
            self.arm=self.arm_class(*self.context,self.mission.initial_tree,self.m,self.d)
            self.travel=self.travel_class(self.m,self.d,self.meta)
            self.command=self.arm.evaluate(*self.mission.arm_reference(0.),self._controller_state())
            self.bank=self.bank_class(self.reference.nominal[self.pi],self.command)
            self.initial_fluid_J=self.bank.stored_energy(self.d.qpos[self.qi][self.pi])+self.drive.stored_energy()
            self.ready=True
        self._commit()
        desired=self.mission.arm_reference(t)
        if self.step % self.arm_every==0:
            self.command=self.arm.evaluate(*desired,self._controller_state())
        if self.step % self.servo_every==0:
            self.last_command=self.travel.evaluate(self.d,np.zeros(3),np.zeros(3),.001,
                                                   update_outer=self.step % self.arm_every==0)
        q=self.d.qpos[self.qi]; v=self.d.qvel[self.vi]
        item=self.bank.evaluate(q[self.pi],v[self.pi],self.command,self.dt)
        drive=self.drive.evaluate(self.d.qvel[self.travel.vi],self.last_command,self.dt)
        efforts=np.zeros(self.m.nu);efforts[self.ai]=item['effort'];efforts[self.travel.ai]=drive['effort']
        if self.step % self.sample_every==0:
            self._metrics(t,desired)
        self.pending=dict(arm=item,travel=drive,effort=efforts.copy(),speed=self.d.qvel[self.port_vi].copy())
        hydraulic=dict(arm_pressure_Pa=self.bank.pressure.copy(),arm_rotary_effort=self.bank.rotary.copy(),
                       travel_pressure_Pa=self.drive.pressure.copy(),arm_power_W=item['power'],travel_power_W=drive['power'],
                       arm_fluid_identity_W=item['fluid_identity_W'],travel_fluid_identity_W=drive['fluid_identity_W'],
                       completed_physics_intervals=self.completed_steps,cumulative=self.totals,
                       fluid_energy_J=self.bank.stored_energy(q[self.pi])+self.drive.stored_energy(),
                       initial_fluid_energy_J=self.initial_fluid_J)
        hydraulic['energy_audit'] = self.energy_audit()
        self.step+=1
        return jsonable(dict(efforts=efforts,mapping=mapping,metrics=self.last_metrics,hydraulics=hydraulic,
                             port_positions=self.d.qpos[self.port_qi],port_velocities=self.d.qvel[self.port_vi],
                             base_measured_acceleration=self.d.qacc[:6],
                             passive_pin_gauge=self._spin_gauge_report()))

    def energy_audit(self):
        """Raw finite-step hydraulic residual, not a fabricated conservation pass."""
        if not self.ready:
            return {}
        arm=self.totals['arm'];travel=self.totals['travel']
        arm_net=sum(arm.get(k,0.) for k in ('supply','rotary','compressibility_geometry'))-sum(
            arm.get(k,0.) for k in ('throttle','leakage','relief','friction','mechanical'))
        travel_net=travel.get('supply',0.)-sum(travel.get(k,0.) for k in (
            'throttle','leakage','relief','friction','numerical_storage_loss','mechanical'))
        fluid=self.bank.stored_energy(self.d.qpos[self.qi][self.pi])+self.drive.stored_energy()
        return dict(fluid_energy_J=fluid,fluid_energy_change_J=fluid-self.initial_fluid_J,
            integrated_held_speed_fluid_budget_J=arm_net+travel_net,
            fluid_finite_step_defect_J=fluid-self.initial_fluid_J-arm_net-travel_net,
            held_speed_mechanical_work_J=arm.get('mechanical',0.)+travel.get('mechanical',0.),
            port_work_accounting_error_J=self.totals['actuator_held_speed_work_J']-arm.get('mechanical',0.)-travel.get('mechanical',0.),
            measured_midpoint_actuator_work_J=self.totals['actuator_midpoint_work_J'],
            hydraulic_native_partition_error_J=self.totals['actuator_midpoint_work_J']-self.totals['actuator_held_speed_work_J'],
            scope='Hydraulic and observed shaft work only; contact and constraint work are not measured here, so the full native mechanical energy balance remains unclosed')

    def finalize(self,t,positions,quaternions_wxyz,linear_velocities,angular_velocities):
        """Commit the terminal native state without creating another interval."""
        if self.finalized or not self.step:
            raise ValueError('Finalization requires an active, previously ticked simulation')
        if abs(float(t)-self.step*self.dt)>max(1e-10,self.dt*1e-7):
            raise ValueError(f'Expected terminal time {self.step*self.dt:.12g}, received {t}')
        self.mapper.update(float(t),positions,quaternions_wxyz,linear_velocities,angular_velocities)
        if self.mode=='mission':
            self._commit()
            self._metrics(t,self.mission.arm_reference(t))
        else:
            self.completed_steps=self.step
        self.finalized=True
        result=self.summary()
        if self.ready:
            result['final_hydraulic_state']=dict(arm_pressure_Pa=self.bank.pressure.tolist(),
                arm_rotary_effort=self.bank.rotary.tolist(),travel_pressure_Pa=self.drive.pressure.tolist(),
                fluid_energy_J=self.bank.stored_energy(self.d.qpos[self.qi][self.pi])+self.drive.stored_energy(),
                initial_fluid_energy_J=self.initial_fluid_J)
        return result

    def summary(self):
        result=dict(case=self.case,mode=self.mode,requested_ticks=self.step,completed_intervals=self.completed_steps,
                    finalized=self.finalized,final_observation_time_s=self.mapper.previous_t,
                    pending_interval_not_committed=self.pending is not None,hydraulic_energies_J=self.totals,
                    last_metrics=self.last_metrics,mapping=self.mapper.last,independent_physx_validation=False)
        if self.ready:
            result['controller']=self.arm.diagnostics()
            result['hydraulic_energy_audit']=self.energy_audit()
        if self.passive_spin_gauge is not None:
            result['passive_spin_gauge']=self.passive_spin_gauge
        return jsonable(result)
