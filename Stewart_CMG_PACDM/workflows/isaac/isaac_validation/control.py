"""Unchanged length-PD and cached feedforward, shared by both verification paths.

This module never assigns simulator state or computes a desired body pose for
an engine. It emits the same six actuator forces and world wrench as v3.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from .mechanics import disturbance


class ReferenceController:
    def __init__(self, cmg: dict, reference_path: Path, name: str):
        self.name = name
        self.feedforward = name not in ('no_feedforward', 'zero_actuation')
        with np.load(reference_path, allow_pickle=False) as z:
            self.initial_q = z['q'][0].copy()
            self.qs = CubicSpline(z['time'], z['q'])
            self.vs = CubicSpline(z['time'], z['velocity'])
            self.fs = CubicSpline(z['time'], z['feedforward_force'])
        self.active = np.array([cmg['coordinate_ids'].index(n) for n in cmg['independent_ids']])
        self.kp = cmg['actuation']['length_kp_N_m']
        self.kd = cmg['actuation']['length_kd_N_s_m']
        self.limit = cmg['actuation']['force_limit_N']

    def evaluate(self, t: float, q: np.ndarray, velocity: np.ndarray):
        target, target_v = self.qs(t), self.vs(t)
        if self.name == 'zero_actuation':
            requested, wrench = np.zeros(6), np.zeros(6)
        else:
            requested = ((self.fs(t) if self.feedforward else np.zeros(6))
                         + self.kp*(target[self.active]-q[self.active])
                         + self.kd*(target_v[self.active]-velocity[self.active]))
            wrench = disturbance(t)
        if not all(np.all(np.isfinite(x)) for x in (target, target_v, requested, wrench)):
            raise FloatingPointError(f'{self.name}: nonfinite controller output at {t:.6g}s')
        force = np.clip(requested, -self.limit, self.limit)
        return target, target_v, requested, force, wrench
