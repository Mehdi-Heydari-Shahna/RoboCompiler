"""Guarded stationary cut/lift/slew/dump references; no writes to plant state."""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from mobile_mission import Mission, smooth
from digging_path import tip_pose


class SoilMission(Mission):
    def __init__(self, engine, reference, cut_depth=.18, cut_policy='load-aware'):
        super().__init__(engine, reference, approach=0., distance=0., yaw=0.)
        active = np.array(engine.active)
        base = reference.nominal.copy()
        nominal = base[active]
        seed = self.arm_path['independent'][0, 1:4].copy()
        # A shallow below-surface cut, within the original mechanism guards.
        goals = [[7.55, .65, -1.5], [7.30, -cut_depth, -1.70],
                 [6.55, -cut_depth, -1.85], [5.85, .20, -2.25]]
        knots = []
        for goal in goals:
            def residual(x):
                q = base.copy(); q[active[1:4]] = x
                return tip_pose(engine, q)-goal
            fit = least_squares(residual, seed,
                                bounds=(nominal[1:4]-1.49, nominal[1:4]+1.49),
                                xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if np.max(abs(residual(fit.x))) > 1e-7:
                raise ValueError('Below-ground waypoint is unreachable')
            u = nominal.copy(); u[1:4] = fit.x
            # Reconstruct now to reject branch/closure failures before stepping.
            reference.reconstruct(u)
            knots.append(u); seed = fit.x
        self.initial_tree = reference.reconstruct(knots[0])
        # Increase the slew angle so receiver walls cannot overlap the dig bed.
        fractions = np.linspace(0., 1., 41)
        opening = self.unload_curve(fractions); opening[:, 0] += .60
        self.unload_curve = CubicSpline(fractions, opening, axis=0)
        self.receiver_closed = opening[0].copy(); self.dump = opening[-1].copy()
        closed = reference.reconstruct(self.receiver_closed)
        T = engine.source.evaluate(closed, np.zeros(23), [0, 0, -9.81])['poses']['body_56']
        self.depot_center = T[:3,:3]@self.mouth_local+T[:3,3]
        self.depot_half_size = np.array([1., 1.])
        self.receiver_plan.update(source_mouth_world_m=self.depot_center,
                                  closed_independent=self.receiver_closed,
                                  open_independent=self.dump)
        self.names = ['settle soil', 'lower below surface', 'draw through soil',
                      'curl bucket', 'lift clear', 'hold load', 'slew to receiver',
                      'dump material', 'wait for deposition', 'close bucket', 'return']
        self.durations = [2., 4., 4., 3., 3., 1., 6., 5., 3., 5., 6.]
        self.starts = [knots[0], knots[0], knots[1], knots[2], knots[3],
                       self.transport, self.transport, self.receiver_closed,
                       self.dump, self.dump, self.receiver_closed]
        self.ends = [knots[0], knots[1], knots[2], knots[3], self.transport,
                     self.transport, self.receiver_closed, self.dump, self.dump,
                     self.receiver_closed, self.transport]
        self.phase = 0; self.entered = 0.; self.done = False
        self.events = []; self.failures = []; self.finished_at = None
        self.last = (knots[0].copy(), np.zeros(7), np.zeros(7))
        self.required_payload_kg = 1.
        self.cut_policy = cut_policy
        self.cut_load_target_kg = 25.

    def arm_reference(self, t):
        elapsed = t-self.entered; dur = self.durations[self.phase]
        if self.phase in (7, 9):
            arm = self._open_reference(elapsed, 0., dur, reverse=self.phase == 9)
        else:
            f, df, ddf = smooth(elapsed, 0., dur)
            delta = self.ends[self.phase]-self.starts[self.phase]
            arm = (self.starts[self.phase]+f*delta, df*delta, ddf*delta)
        self.last = arm
        return arm

    def update(self, t, tracking_error, soil_speed, bucket_mass, lip_height,
               deposited_mass, settle_only=False, lip_x=None):
        if self.done:
            return
        elapsed = t-self.entered
        ready = elapsed >= self.durations[self.phase] and tracking_error < .035
        # Once a completed draw reference has collected material, permit curl
        # even if soil resistance prevents its precise joint endpoint. The
        # next polynomial starts at the same commanded endpoint with zero
        # velocity/acceleration, so this does not jump the reference or plant.
        if self.phase == 2 and self.cut_policy == 'load-aware':
            loaded = (bucket_mass >= self.cut_load_target_kg and lip_x is not None and lip_x <= 7.05)
            ready = elapsed >= self.durations[self.phase] and (tracking_error < .035 or loaded)
        if self.phase == 0:
            ready = ready and soil_speed < .04
        if self.phase == 4:
            ready = ready and lip_height > .8
        if self.phase == 8:
            ready = ready and bucket_mass < self.required_payload_kg and soil_speed < .08
        if elapsed > self.durations[self.phase]+6.:
            self.failures.append(f'Phase timeout: {self.names[self.phase]}')
            self.done = True; self.finished_at = t
            return
        if not ready:
            return
        self.events.append(dict(time_s=t, phase=self.names[self.phase],
                                bucket_mass_kg=bucket_mass, deposited_mass_kg=deposited_mass))
        if self.phase == 5 and bucket_mass < self.required_payload_kg:
            self.failures.append('Lift completed with less than 1 kg in bucket')
        if self.phase == 8 and deposited_mass < self.required_payload_kg:
            self.failures.append('Less than 1 kg settled in receiving area')
        if settle_only or self.phase == len(self.names)-1:
            self.done = True; self.finished_at = t
        else:
            self.phase += 1; self.entered = t
