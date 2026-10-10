"""Declared Pinocchio case table (mirrors the v22 MuJoCo contact benchmark).

The five positive cases use the v22 scenario definitions with a Pinocchio
step of 1 ms (0.5 ms for the refinement case); v22 used 25 us and 12.5 us
for MuJoCo's soft contact.  The three negative controls use the v22 control
durations.  Acceptance code receives these records from here and never
trusts labels stored inside result files.
"""
from __future__ import annotations

POSITIVE = [
    dict(name='landing_nominal', dt=1e-3, duration=10., config={}),
    dict(name='landing_refined', dt=5e-4, duration=10., config={}),
    dict(name='landing_low_friction', dt=1e-3, duration=10., config={'friction': .4}),
    dict(name='landing_higher_drop_push', dt=1e-3, duration=10.,
         config={'drop_height_m': .08, 'push_force_N': 80.}),
    dict(name='landing_slow_actuators', dt=1e-3, duration=10.,
         config={'actuator_time_constant_s': .004}),
]

NEGATIVE = [
    dict(name='negative_no_contact', dt=1e-3, duration=.3, config={}, no_contact=True),
    dict(name='negative_no_loops_contact', dt=1e-3, duration=.3, config={}, open_tree=True),
    dict(name='negative_passive', dt=1e-3, duration=1., config={}, passive=True, may_stop_early=True),
]

ALL = POSITIVE + NEGATIVE
BY_NAME = {case['name']: case for case in ALL}
# Evidence replay covers every case integrated by run_case (not the open tree).
EVIDENCE = [case for case in ALL if not case.get('open_tree')]


def run(name, output=None, progress=True, plant=None):
    """Integrate one declared case and write results/<name>.npz/.json."""
    from .pin_simulation import run_case, run_open_tree
    case = BY_NAME[name]
    if case.get('open_tree'):
        return run_open_tree(name, dt=case['dt'], config=case['config'], duration=case['duration'],
                             output=output, progress=progress, plant=plant)
    return run_case(name, dt=case['dt'], config=case['config'], duration=case['duration'],
                    no_contact=case.get('no_contact', False), passive=case.get('passive', False),
                    output=output, progress=progress, plant=plant,
                    stop_on_bound=case.get('may_stop_early', False))
