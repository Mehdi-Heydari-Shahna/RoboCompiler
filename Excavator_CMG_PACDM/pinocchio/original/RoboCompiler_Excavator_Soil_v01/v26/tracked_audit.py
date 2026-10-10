"""Independent rigid-body backend, contact and energy audits for v26.

The Pinocchio tree is reconstructed from *compiled model parameters*, not fitted
to MuJoCo results.  This checks independent CRBA/RNEA/FK implementations on the
same inertial model.  It does not validate the original CAD/source conversion,
identify physical track parameters, or recover unique closed-loop pin loads.
Equality constraints and contacts remain separate, explicitly audited forces.

Conventions: MuJoCo free-joint linear velocity is world aligned and angular
velocity is body aligned. Pinocchio free-flyer velocities are both body aligned.
The velocity-coordinate derivative is included in bias/inverse dynamics.
MuJoCo 3.3.7 and Pinocchio 3.8.0 are the validation versions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

AUDIT_REVISION = 'complete_native_global_wrench_v2'


def quat_matrix(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()


def relative_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b), initial=0) /
                 max(1.0, np.max(np.abs(a), initial=0), np.max(np.abs(b), initial=0)))


def _placement(pos, quat=None):
    return pin.SE3(np.eye(3) if quat is None else quat_matrix(quat), np.asarray(pos).copy())


def _constraint_generalized(m, d, mask):
    forces = np.zeros(d.nefc)
    forces[mask] = d.efc_force[mask]
    result = np.zeros(m.nv)
    mujoco.mj_mulJacTVec(m, d, result, forces)
    return result


def contact_wrenches(m, d):
    """Map individual contact forces, independently of efc_J transpose.

    Returns explicit generalized force and world force/moment from contacts
    crossing the world/moving-system boundary. Moment is about world origin.
    Internal wheel/shoe and bucket/grain contacts cancel in global momentum.
    """
    generalized = np.zeros(m.nv)
    force, moment = np.zeros(3), np.zeros(3)
    raw = np.zeros(6)
    peak = 0.0
    for ci in range(d.ncon):
        contact = d.contact[ci]
        if contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(m, d, ci, raw)
        rotation = contact.frame.reshape(3, 3).T
        f, torque = rotation @ raw[:3], rotation @ raw[3:]
        b1, b2 = int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])
        if b2:
            mujoco.mj_applyFT(m, d, f, torque, contact.pos, b2, generalized)
        if b1:
            mujoco.mj_applyFT(m, d, -f, -torque, contact.pos, b1, generalized)
        # A fixed geom may belong to a fixed descendant of world, not body 0.
        # The robust classification is the presence of any ancestor DOF.
        def mobile(b):
            while b:
                if m.body_dofnum[b]:
                    return True
                b = int(m.body_parentid[b])
            return False
        moving1, moving2 = mobile(b1), mobile(b2)
        if moving1 != moving2:
            sign = 1.0 if moving2 else -1.0
            force += sign * f
            moment += sign * (np.cross(contact.pos, f) + torque)
        peak = max(peak, float(np.linalg.norm(f)))
    return dict(generalized=generalized, external_force=force,
                external_moment=moment, peak_pair_force_N=peak)


class CompiledTreeAudit:
    """A Pinocchio tree and explicit MuJoCo/Pinocchio coordinate conversion."""

    def __init__(self, model):
        if model.nflex:
            raise NotImplementedError("This audit covers rigid bodies, not flex elements")
        self.mj = model
        self.pin = pin.Model()
        self.frames = {}
        self.joints = []
        self.free = []
        self.support = {0: 0}
        self.relative = {0: pin.SE3.Identity()}
        self.body_parent_frame = {0: 0}
        for bid in range(1, model.nbody):
            parent = int(model.body_parentid[bid])
            support = self.support[parent]
            body_placement = self.relative[parent] * _placement(model.body_pos[bid], model.body_quat[bid])
            # Each joint rotates/translates the body reference frame in XML order.
            residual = body_placement
            jids = range(int(model.body_jntadr[bid]), int(model.body_jntadr[bid] + model.body_jntnum[bid]))
            for jid in jids:
                typ = int(model.jnt_type[jid])
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
                if typ == int(mujoco.mjtJoint.mjJNT_FREE):
                    if parent != 0 or model.body_jntnum[bid] != 1:
                        raise ValueError("A MuJoCo free joint must be the only joint of a world child")
                    kind, place = pin.JointModelFreeFlyer(), pin.SE3.Identity()
                    after = pin.SE3.Identity()
                else:
                    place = residual * _placement(model.jnt_pos[jid])
                    after = _placement(-model.jnt_pos[jid])
                    axis = model.jnt_axis[jid]
                    if typ == int(mujoco.mjtJoint.mjJNT_HINGE):
                        kind = pin.JointModelRevoluteUnaligned(axis)
                    elif typ == int(mujoco.mjtJoint.mjJNT_SLIDE):
                        kind = pin.JointModelPrismaticUnaligned(axis)
                    elif typ == int(mujoco.mjtJoint.mjJNT_BALL):
                        kind = pin.JointModelSpherical()
                    else:
                        raise NotImplementedError(f"joint type {typ}")
                support = self.pin.addJoint(support, kind, place, joint_name)
                pj = self.pin.joints[support]
                entry = dict(mj=jid, pin=support, typ=typ,
                             mq=int(model.jnt_qposadr[jid]), mv=int(model.jnt_dofadr[jid]),
                             pq=pj.idx_q, pv=pj.idx_v, nq=pj.nq, nv=pj.nv, bid=bid)
                self.joints.append(entry)
                if typ == int(mujoco.mjtJoint.mjJNT_FREE):
                    self.free.append(entry)
                residual = after
            inertia_rotation = quat_matrix(model.body_iquat[bid])
            inertia = inertia_rotation @ np.diag(model.body_inertia[bid]) @ inertia_rotation.T
            self.pin.appendBodyToJoint(support, pin.Inertia(float(model.body_mass[bid]),
                                      model.body_ipos[bid].copy(), inertia), residual)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
            fid = self.pin.addFrame(pin.Frame(name, support, self.body_parent_frame[parent],
                                             residual, pin.FrameType.BODY), False)
            self.frames[bid] = fid
            self.support[bid], self.relative[bid], self.body_parent_frame[bid] = support, residual, fid
        if self.pin.nv != model.nv:
            raise ValueError(f"DOF count mismatch: MJ {model.nv}, Pin {self.pin.nv}")
        self.data = self.pin.createData()
        self.dynamic_bodies = [b for b in range(1, model.nbody) if self.support[b] != 0]
        self.mass = float(sum(model.body_mass[b] for b in self.dynamic_bodies))

    def coordinates(self, qpos, qvel):
        """v_pin = T v_mj, a_pin = T a_mj + Tdot v_mj."""
        q = pin.neutral(self.pin)
        T = np.zeros((self.pin.nv, self.mj.nv))
        derivative = np.zeros(self.pin.nv)
        for j in self.joints:
            mq, mv, pq, pv, typ = (j[x] for x in ("mq", "mv", "pq", "pv", "typ"))
            if typ == int(mujoco.mjtJoint.mjJNT_FREE):
                q[pq:pq+3] = qpos[mq:mq+3]
                q[pq+3:pq+7] = qpos[mq+3:mq+7][[1, 2, 3, 0]]
                R = quat_matrix(qpos[mq+3:mq+7])
                T[pv:pv+3, mv:mv+3] = R.T
                T[pv+3:pv+6, mv+3:mv+6] = np.eye(3)
                derivative[pv:pv+3] = -np.cross(qvel[mv+3:mv+6], R.T @ qvel[mv:mv+3])
            elif typ == int(mujoco.mjtJoint.mjJNT_BALL):
                q[pq:pq+4] = qpos[mq:mq+4][[1, 2, 3, 0]]
                T[pv:pv+3, mv:mv+3] = np.eye(3)
            else:
                q[pq] = qpos[mq] - self.mj.qpos0[mq]
                T[pv, mv] = 1.0
        return q, T @ qvel, T, derivative

    def evaluate(self, qpos, qvel, gravity=None, acceleration=None):
        q, v, T, derivative = self.coordinates(np.asarray(qpos), np.asarray(qvel))
        gravity = self.mj.opt.gravity if gravity is None else gravity
        self.pin.gravity.linear = np.asarray(gravity)
        self.pin.gravity.angular = np.zeros(3)
        Mp = pin.crba(self.pin, self.data, q).copy()
        hp = pin.nonLinearEffects(self.pin, self.data, q, v).copy()
        M = T.T @ Mp @ T + np.diag(self.mj.dof_armature)
        h = T.T @ (hp + Mp @ derivative)
        kinetic = float(pin.computeKineticEnergy(self.pin, self.data, q, v))
        kinetic += float(0.5 * np.dot(self.mj.dof_armature, np.asarray(qvel)**2))
        potential = float(pin.computePotentialEnergy(self.pin, self.data, q))
        gravity_force = T.T @ pin.computeGeneralizedGravity(self.pin, self.data, q).copy()
        pin.forwardKinematics(self.pin, self.data, q, v)
        pin.updateFramePlacements(self.pin, self.data)
        poses = {bid: self.data.oMf[fid].homogeneous.copy() for bid, fid in self.frames.items()}
        momentum = pin.computeCentroidalMomentum(self.pin, self.data, q, v)
        p = momentum.linear.copy()
        com = pin.centerOfMass(self.pin, self.data, q).copy()
        angular = momentum.angular.copy() + np.cross(com, p)
        result = dict(mass_matrix=M, bias_forces=h, gravity_forces=gravity_force,
                      kinetic_energy=kinetic, potential_energy=potential,
                      linear_momentum=p, angular_momentum=angular, center_of_mass=com,
                      poses=poses)
        if acceleration is not None:
            ap = T @ acceleration + derivative
            force = T.T @ pin.rnea(self.pin, self.data, q, v, ap).copy()
            force += self.mj.dof_armature * acceleration
            result["inverse_dynamics"] = force
        return result

    def energy_momentum(self, qpos, qvel, gravity=None, kinetic_only=False):
        """O(n) evaluation for directional derivatives, without CRBA."""
        q, v, _, _ = self.coordinates(np.asarray(qpos), np.asarray(qvel))
        self.pin.gravity.linear = self.mj.opt.gravity if gravity is None else np.asarray(gravity)
        kinetic = float(pin.computeKineticEnergy(self.pin, self.data, q, v))
        kinetic += float(0.5 * np.dot(self.mj.dof_armature, np.asarray(qvel)**2))
        potential = float(pin.computePotentialEnergy(self.pin, self.data, q))
        h = pin.computeCentroidalMomentum(self.pin, self.data, q, v)
        p, angular = h.linear.copy(), h.angular.copy()
        com = pin.centerOfMass(self.pin, self.data, q).copy()
        return kinetic if kinetic_only else kinetic+potential, p, angular+np.cross(com, p)

    def audit_snapshot(self, d, finite_difference=True):
        """Call after mj_forward at the exact saved state/control/applied forces.

        Do not call directly on mj_step's stale pre-integration derived arrays.
        This method never alters its caller's MuJoCo state.
        """
        m = self.mj
        result = self.evaluate(d.qpos, d.qvel, acceleration=d.qacc)
        mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, mass, d.qM)
        native_id = mass @ d.qacc + d.qfrc_bias
        contact = contact_wrenches(m, d)
        types = np.asarray(d.efc_type)
        is_contact = types >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        contact_generalized = _constraint_generalized(m, d, is_contact)
        equality_generalized = _constraint_generalized(m, d, types == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        other_generalized = d.qfrc_constraint - contact_generalized - equality_generalized
        applied = d.qfrc_applied.copy()
        xforce, xmoment = np.zeros(3), np.zeros(3)
        for b in self.dynamic_bodies:
            f, t = d.xfrc_applied[b, :3], d.xfrc_applied[b, 3:]
            mujoco.mj_applyFT(m, d, f, t, d.xipos[b], b, applied)
            xforce += f
            xmoment += t + np.cross(d.xipos[b], f)
        rhs = d.qfrc_actuator + d.qfrc_passive + applied + d.qfrc_constraint
        kinetic_body, potential_body = 0.0, 0.0
        p_body, h_body = np.zeros(3), np.zeros(3)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        gravity_world, gravity_moment = np.zeros(3), np.zeros(3)
        max_pose = 0.0
        for b in self.dynamic_bodies:
            mujoco.mj_jacBodyCom(m, d, jp, jr, b)
            vc, w = jp @ d.qvel, jr @ d.qvel
            Ri = d.ximat[b].reshape(3, 3)
            inertia = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
            kinetic_body += .5*m.body_mass[b]*float(vc@vc) + .5*float(w@inertia@w)
            potential_body -= m.body_mass[b]*float(m.opt.gravity @ d.xipos[b])
            p_body += m.body_mass[b]*vc
            h_body += np.cross(d.xipos[b], m.body_mass[b]*vc) + inertia @ w
            gf = m.body_mass[b]*m.opt.gravity
            gravity_world += gf
            gravity_moment += np.cross(d.xipos[b], gf)
            actual = np.eye(4)
            actual[:3,:3], actual[:3,3] = d.xmat[b].reshape(3,3), d.xpos[b]
            max_pose = max(max_pose, float(np.max(np.abs(actual-result['poses'][b]))))
        kinetic_body += .5*float(np.dot(m.dof_armature, d.qvel**2))
        power = {"actuator_W": float(d.qfrc_actuator@d.qvel),
                 "passive_W": float(d.qfrc_passive@d.qvel),
                 "applied_W": float(applied@d.qvel),
                 "contact_W": float(contact_generalized@d.qvel),
                 "equality_W": float(equality_generalized@d.qvel),
                 "other_constraint_W": float(other_generalized@d.qvel)}
        metrics = dict(mass_matrix_relative=relative_error(result['mass_matrix'], mass),
                       bias_relative=relative_error(result['bias_forces'], d.qfrc_bias),
                       inverse_dynamics_relative=relative_error(result['inverse_dynamics'], native_id),
                       native_force_balance_relative=relative_error(native_id, rhs),
                       pin_force_balance_relative=relative_error(result['inverse_dynamics'], rhs),
                       body_pose_max_abs=max_pose,
                       kinetic_energy_relative=relative_error(result['kinetic_energy'], kinetic_body),
                       potential_energy_relative=relative_error(result['potential_energy'], potential_body),
                       linear_momentum_relative=relative_error(result['linear_momentum'], p_body),
                       angular_momentum_relative=relative_error(result['angular_momentum'], h_body),
                       contact_mapping_relative=relative_error(contact['generalized'], contact_generalized))
        momentum_accounting = dict(available=False)
        momentum_assumptions = False
        if finite_difference:
            eps = 2e-6 / max(1.0, float(np.max(np.abs(d.qvel), initial=0)),
                            float(np.sqrt(np.max(np.abs(d.qacc), initial=0))))
            qp, qm = d.qpos.copy(), d.qpos.copy()
            mujoco.mj_integratePos(m, qp, d.qvel, eps)
            mujoco.mj_integratePos(m, qm, d.qvel, -eps)
            ep, pp, hp = self.energy_momentum(qp, d.qvel+eps*d.qacc, kinetic_only=True)
            em, pm, hm = self.energy_momentum(qm, d.qvel-eps*d.qacc, kinetic_only=True)
            # Differentiate K numerically and U analytically. Subtracting two
            # large absolute gravitational potentials destroys precision when
            # the vehicle settles with nearly zero mechanical power.
            energy_rate = (ep-em)/(2*eps) + float(result['gravity_forces']@d.qvel)
            linear_rate, angular_rate = (pp-pm)/(2*eps), (hp-hm)/(2*eps)
            metrics['differential_energy_power_relative'] = relative_error(energy_rate, sum(power.values()))
            # Global momentum closure is only assessed for the declared setting:
            # no generalized external forces, no gravcomp/fluid forces, no rotor
            # armature shortcut, and all moving roots free. Joint motor/damping
            # forces are internal. Finite-gap loop couples are measured below.
            momentum_assumptions = (not np.any(m.dof_armature) and not np.any(d.qfrc_applied)
                                    and not np.any(m.body_gravcomp) and m.opt.density == 0
                                    and m.opt.viscosity == 0 and self.all_roots_free()
                                    and self.internal_joint_actuators_only())
            if momentum_assumptions:
                # A soft connect constraint can act at two noncoincident sites.
                # Its opposite forces then have a finite gap x force couple.
                # This numerical constraint effect is not a physical external
                # load and is not zero merely because the ideal joint is internal.
                # Project generalized constraint forces onto independent rigid
                # translations/rotations of every free tree to measure it.
                virtual = np.zeros((m.nv, 6))
                for joint in self.free:
                    bid, mv = joint['bid'], joint['mv']
                    virtual[mv:mv+3, :3] = np.eye(3)
                    for axis in range(3):
                        direction = np.eye(3)[axis]
                        virtual[mv:mv+3, axis+3] = np.cross(direction, d.xpos[bid])
                        virtual[mv+3:mv+6, axis+3] = d.xmat[bid].reshape(3, 3).T@direction
                equality_wrench = virtual.T@equality_generalized
                other_wrench = virtual.T@other_generalized
                actuator_wrench = virtual.T@d.qfrc_actuator
                passive_wrench = virtual.T@d.qfrc_passive
                external_wrench = np.r_[gravity_world+contact['external_force']+xforce,
                                        gravity_moment+contact['external_moment']+xmoment]
                complete_wrench = external_wrench+equality_wrench+other_wrench+actuator_wrench+passive_wrench
                # Pinocchio's analytic centroidal momentum variation independently
                # confirms that a residual is not a finite-difference artifact.
                qp, vp, transform, transform_rate = self.coordinates(d.qpos, d.qvel)
                variation = pin.computeCentroidalMomentumTimeVariation(
                    self.pin, self.data, qp, vp, transform@d.qacc+transform_rate)
                analytic_linear = variation.linear.copy()
                analytic_angular_com = variation.angular.copy()
                com = pin.centerOfMass(self.pin, self.data, qp).copy()
                analytic_angular = analytic_angular_com+np.cross(com, analytic_linear)
                metrics['linear_momentum_rate_relative'] = relative_error(linear_rate, complete_wrench[:3])
                metrics['angular_momentum_rate_relative'] = relative_error(angular_rate, complete_wrench[3:])
                metrics['analytic_linear_momentum_rate_relative'] = relative_error(analytic_linear, complete_wrench[:3])
                metrics['analytic_angular_momentum_rate_relative'] = relative_error(analytic_angular, complete_wrench[3:])
                momentum_accounting = dict(available=True, revision=AUDIT_REVISION,
                    wrench_convention='World [Fx,Fy,Fz,Mx,My,Mz], moments about world origin',
                    physical_external_wrench=external_wrench.tolist(),
                    finite_gap_equality_global_wrench=equality_wrench.tolist(),
                    other_native_constraint_global_wrench=other_wrench.tolist(),
                    actuator_global_wrench=actuator_wrench.tolist(),
                    passive_global_wrench=passive_wrench.tolist(),
                    complete_native_wrench=complete_wrench.tolist(),
                    finite_difference_linear_rate=linear_rate.tolist(),
                    finite_difference_angular_rate=angular_rate.tolist(),
                    analytic_linear_rate=analytic_linear.tolist(),
                    analytic_angular_rate=analytic_angular.tolist(),
                    raw_ideal_linear_balance_relative=relative_error(linear_rate, external_wrench[:3]),
                    raw_ideal_angular_balance_relative=relative_error(angular_rate, external_wrench[3:]),
                    analytic_raw_ideal_angular_balance_relative=relative_error(analytic_angular, external_wrench[3:]),
                    fd_analytic_linear_difference_relative=relative_error(linear_rate, analytic_linear),
                    fd_analytic_angular_difference_relative=relative_error(angular_rate, analytic_angular),
                    interpretation='Equality global couples are measured finite-gap soft-constraint artifacts, not physical external loads. Complete native force accounting is checked; exact ideal closed-joint momentum conservation is not expected. Raw ideal-closure residuals remain visible.')
            power['finite_difference_energy_rate_W'] = float(energy_rate)
            power['energy_rate_step_s'] = float(eps)
        thresholds = {key: 1e-9 for key in metrics}
        thresholds.update(native_force_balance_relative=1e-6, pin_force_balance_relative=1e-6,
                          differential_energy_power_relative=2e-5,
                          linear_momentum_rate_relative=2e-5, angular_momentum_rate_relative=2e-5,
                          analytic_linear_momentum_rate_relative=2e-5,
                          analytic_angular_momentum_rate_relative=2e-5)
        checks = {key: bool(np.isfinite(value) and value <= thresholds[key]) for key, value in metrics.items()}
        return dict(time_s=float(d.time), nq=m.nq, nv=m.nv, bodies=m.nbody-1,
                    contacts=d.ncon, constraints=d.nefc, metrics=metrics, limits=thresholds,
                    checks=checks, passed=all(checks.values()), power=power,
                    energy=dict(kinetic_J=result['kinetic_energy'], potential_J=result['potential_energy']),
                    peak_contact_pair_force_N=contact['peak_pair_force_N'],
                    audit_revision=AUDIT_REVISION, momentum_accounting=momentum_accounting,
                    momentum_rate_assessed=bool(finite_difference and momentum_assumptions),
                    momentum_rate_scope="Complete native wrench balance including measured finite-gap constraint couples; raw ideal-closure residuals retained separately. Requires zero armature, zero generalized applied forces, no gravcomp/fluid forces, free roots and internal joint actuation.")

    def internal_joint_actuators_only(self):
        m = self.mj
        for actuator in range(m.nu):
            if m.actuator_trntype[actuator] != int(mujoco.mjtTrn.mjTRN_JOINT):
                return False
            jid = int(m.actuator_trnid[actuator,0])
            if jid < 0 or m.jnt_type[jid] == int(mujoco.mjtJoint.mjJNT_FREE):
                return False
        return True

    def all_roots_free(self):
        for b in self.dynamic_bodies:
            if self.mj.body_parentid[b] == 0:
                jid = int(self.mj.body_jntadr[b])
                if self.mj.body_jntnum[b] != 1 or self.mj.jnt_type[jid] != int(mujoco.mjtJoint.mjJNT_FREE):
                    return False
        return True

    def frame_covariance(self, qpos, qvel):
        """Rotate/translate all free roots and gravity, comparing tree dynamics.

        This is a rigid-body backend test; ground geometry/contact states are
        deliberately excluded. It cannot validate a rotated contact experiment.
        """
        if not self.all_roots_free():
            return dict(passed=False, available=False, reason="A moving tree is anchored to the world")
        Q = Rotation.from_rotvec([.31, -.27, .43]).as_matrix()
        translation = np.array([.72, -.43, .28])
        qp, vp = np.asarray(qpos).copy(), np.asarray(qvel).copy()
        C = np.eye(self.mj.nv)
        for j in self.free:
            mq, mv = j['mq'], j['mv']
            qp[mq:mq+3] = Q@qp[mq:mq+3]+translation
            R = Q @ quat_matrix(qp[mq+3:mq+7])
            qp[mq+3:mq+7] = Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
            C[mv:mv+3, mv:mv+3] = Q
        vp = C @ vp
        original = self.evaluate(qpos, qvel)
        gravity = Q @ self.mj.opt.gravity
        changed = self.evaluate(qp, vp, gravity)
        metrics = dict(mass_relative=relative_error(changed['mass_matrix'], C@original['mass_matrix']@C.T),
                       bias_relative=relative_error(changed['bias_forces'], C@original['bias_forces']),
                       kinetic_relative=relative_error(changed['kinetic_energy'], original['kinetic_energy']),
                       potential_relative=relative_error(changed['potential_energy'], original['potential_energy']-self.mass*gravity@translation),
                       momentum_relative=relative_error(changed['linear_momentum'], Q@original['linear_momentum']),
                       angular_momentum_relative=relative_error(changed['angular_momentum'], Q@original['angular_momentum']+np.cross(translation,Q@original['linear_momentum'])))
        return dict(passed=all(value < 1e-9 for value in metrics.values()), available=True, metrics=metrics,
                    limit=1e-9, scope="Rigid-body tree and gravity; does not rotate or test ground contacts")


def audit_model(xml_path, steps=0, samples=3, output=None):
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    audit = CompiledTreeAudit(m)
    snapshots = [audit.audit_snapshot(d)]
    covariance = audit.frame_covariance(d.qpos, d.qvel)
    marks = set(np.linspace(1, max(steps,1), samples).astype(int))
    for i in range(1, steps+1):
        mujoco.mj_step(m, d)
        if i in marks:
            mujoco.mj_forward(m, d)
            snapshots.append(audit.audit_snapshot(d))
    report = dict(versions=dict(mujoco=mujoco.__version__, pinocchio=pin.__version__),
                  xml=str(xml_path), snapshots=snapshots, frame_covariance=covariance,
                  scope="Independent rigid-body recursions on shared compiled inertial parameters; explicit native constraint/contact audit",
                  passed=all(s['passed'] for s in snapshots) and covariance['passed'])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2))
    return report


def randomized_tree_checks(model, count=12, seed=26019):
    """Unconstrained tree parity under finite rotations, motion and gravity.

    These off-manifold configurations test only rigid-body algorithms; they are
    never described as feasible assembled closed-chain states or task trials.
    This deliberately exercises velocity-coordinate derivative terms which a
    zero-velocity reference-state comparison cannot detect.
    """
    audit = CompiledTreeAudit(model)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    original_gravity = model.opt.gravity.copy()
    records = []
    try:
        for k in range(count):
            data.qpos[:] = model.qpos0
            mujoco.mj_integratePos(model, data.qpos, rng.normal(0,.025,model.nv), 1.)
            data.qvel[:] = rng.normal(0,.35,model.nv)
            acceleration = rng.normal(0,.75,model.nv)
            model.opt.gravity[:] = original_gravity if k%3 == 0 else rng.normal(0,5,3)
            # Forward dynamics itself is not assessed off the closure manifold.
            mujoco.mj_forward(model, data)
            got = audit.evaluate(data.qpos, data.qvel, acceleration=acceleration)
            M = np.zeros((model.nv,model.nv))
            mujoco.mj_fullM(model,M,data.qM)
            pose_error = 0.
            for b in audit.dynamic_bodies:
                pose = got['poses'][b]
                pose_error = max(pose_error, float(np.max(abs(pose[:3,3]-data.xpos[b]))),
                                 float(np.max(abs(pose[:3,:3]-data.xmat[b].reshape(3,3)))))
            metrics = dict(mass_relative=relative_error(got['mass_matrix'],M),
                           bias_relative=relative_error(got['bias_forces'],data.qfrc_bias),
                           inverse_dynamics_relative=relative_error(got['inverse_dynamics'],M@acceleration+data.qfrc_bias),
                           kinetic_relative=relative_error(got['kinetic_energy'],.5*data.qvel@M@data.qvel),
                           pose_max_abs=pose_error)
            covariance = audit.frame_covariance(data.qpos,data.qvel)
            records.append(dict(index=k,gravity_m_s2=model.opt.gravity.tolist(),metrics=metrics,
                                covariance=covariance,passed=all(v<1e-9 for v in metrics.values()) and covariance['passed']))
    finally:
        model.opt.gravity[:] = original_gravity
    return dict(seed=seed,count=count,limit=1e-9,passed=all(r['passed'] for r in records),
                scope="Off-manifold rigid-body tree parity only; not assembled closure or contact task validation",
                records=records)


def converter_regression():
    """Exercise free/ball joints, multiple joints/body, refs and inertial frames."""
    fixture = '''<mujoco><option gravity="1 -2 -9"/>
    <worldbody><body name="root_a" pos=".2 .3 2" euler="20 10 30">
    <freejoint/><geom type="box" size=".2 .1 .15" mass="6"/>
    <body name="multi_joint" pos=".5 .1 .3" euler="10 20 15">
    <joint name="hinge" pos=".1 .2 .3" axis=".3 .5 .1" ref="20"/>
    <joint name="slide" type="slide" pos=".3 -.2 .1" axis=".5 .1 .2" ref=".2"/>
    <geom type="box" size=".3 .2 .1" mass="3"/>
    <body name="ball_child" pos=".2 .5 .1" euler="23 12 8">
    <joint type="ball" pos=".1 -.1 .2"/><geom size=".1" mass="1"/>
    </body></body></body>
    <body name="root_b" pos="2 0 1" euler="30 21 11">
    <freejoint/><geom size=".3" mass="4"/>
    </body></worldbody></mujoco>'''
    return randomized_tree_checks(mujoco.MjModel.from_xml_string(fixture),10,72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml')
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--output', default='outputs/dynamics_audit.json')
    parser.add_argument('--randomized-tree', action='store_true')
    args = parser.parse_args()
    report = audit_model(args.xml, args.steps, output=args.output)
    if args.randomized_tree:
        report['randomized_tree'] = randomized_tree_checks(mujoco.MjModel.from_xml_path(args.xml))
        report['converter_regression'] = converter_regression()
        report['passed'] = report['passed'] and report['randomized_tree']['passed'] and report['converter_regression']['passed']
        Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=report['passed'], output=args.output,
                         failed=[{k:v for k,v in s['metrics'].items() if not s['checks'][k]} for s in report['snapshots']]), indent=2))
