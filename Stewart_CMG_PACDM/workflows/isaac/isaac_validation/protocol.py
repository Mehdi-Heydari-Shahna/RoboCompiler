"""Shared case schedule and fail-closed runtime guards (SI units).

The gravity diagnostic is intentionally shorter than a tracking mission. At
0.2 s the uploaded v3 recording fell beyond pose_z >= 0.4 m; deleting its
tracking guard would therefore NOT fix its joint-limit failure. The 0.1 s
horizon retains the original >= 0.01 m gravity-motion requirement.
"""
from __future__ import annotations
import numpy as np

ZERO_ACTUATION_DURATION = 0.1
ZERO_ACTUATION_TIMESTEP = 0.002
ZERO_MIN_DISPLACEMENT_M = 0.01
TRACKING_DIVERGENCE_M = 0.20
CLOSURE_DIVERGENCE_M = 0.02
CASE_NAMES = ('nominal', 'fine', 'heavy_payload', 'no_feedforward', 'zero_actuation')


def validate_request(name: str, dt: float, duration: float) -> None:
    if name not in CASE_NAMES:
        raise ValueError(f'Unknown case: {name}')
    if not np.isfinite(dt) or not np.isfinite(duration) or dt <= 0 or duration <= 0:
        raise ValueError('Timestep and duration must be positive and finite')
    if round(duration / dt) < 1 or abs(round(duration / dt)*dt - duration) > 1e-9:
        raise ValueError('Duration must be an integer multiple of the physics timestep')
    if duration > (ZERO_ACTUATION_DURATION if name == 'zero_actuation' else 22.) + 1e-9:
        raise ValueError(f'{name}: duration exceeds the declared safe protocol horizon')


class RuntimeGuard:
    """Stop broken constraints/bounds in every mode; track only actuated cases."""
    def __init__(self, cmg: dict):
        ids = cmg['coordinate_ids']
        joints = {j['id']: j for j in cmg['joints']}
        self.lower = np.array([joints[n]['limits']['lower'] for n in ids], float)
        self.upper = np.array([joints[n]['limits']['upper'] for n in ids], float)
        self.ids = ids

    def check(self, name: str, state: dict, position_error: float, time_s: float) -> None:
        q = np.asarray(state['q'], float)
        v = np.asarray(state['velocity'], float)
        values = np.r_[q, v, position_error, state['closure_error_m'],
                        state['joint_error_m'], state['joint_error_rad']]
        if q.shape != self.lower.shape or v.shape != self.lower.shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f'{name}: nonfinite/invalid state guard at {time_s:.3f}s')
        if state['closure_error_m'] > CLOSURE_DIVERGENCE_M:
            raise RuntimeError(f'{name}: closure divergence guard at {time_s:.3f}s: '
                               f'{state["closure_error_m"]:.6g} m')
        # Same 1e-8 numerical boundary allowance as the existing final scorer.
        margin = np.minimum(q-self.lower, self.upper-q)
        if np.min(margin) < -1e-8:
            i = int(np.argmin(margin))
            raise RuntimeError(f'{name}: joint-limit guard at {time_s:.3f}s: '
                               f'{self.ids[i]}={q[i]:.9g}, allowed '
                               f'[{self.lower[i]:.9g}, {self.upper[i]:.9g}]')
        if name != 'zero_actuation' and position_error > TRACKING_DIVERGENCE_M:
            raise RuntimeError(f'{name}: tracking divergence guard at {time_s:.3f}s: '
                               f'{position_error:.6g} m')
