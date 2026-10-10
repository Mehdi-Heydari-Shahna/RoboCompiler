"""Fixed benchmark definitions. Changes are recorded with every result.

Solver settings are part of every case and therefore of every suite check.

solver='PGS', position_iterations=4, velocity_iterations=1 (v1: TGS 64/8).
PhysX TGS executes every position iteration as a sub-step of dt/N and advances
articulation joint angles in float32 at each sub-step. With 64 iterations at
dt=0.25 ms (3.9 us sub-steps) joints near |q|=2.3 rad could not move below
~0.03 rad/s although PhysX still reported that velocity; this stalled joints
4/6 and caused the 2.001 mm tracking failure of the first native suite. PGS
integrates positions once per step, so its iterations improve contact and
friction convergence without that float32 penalty. In the offline PhysX twin,
TGS with one iteration fixed the arm but let the heavy/low-friction cartridge
rotate ~4 deg in the grasp and jam; more TGS iterations re-introduced grasp
creep (the whole grasp island is sub-stepped). With PGS 4/1 all 13 cases
passed in the twin, with and without friction in every solver iteration.
joint_zero_offsets=True: joint zero is authored at the middle of the reference
range so float32 joint angles stay small (see geometry.joint_zero_offsets).
"""
BASE = dict(dt=.00025, mass=.15, width=.044, friction=.8, offset=0.,
            initial_offset=False, feedforward=True, grasp=True,
            solver='PGS', position_iterations=4, velocity_iterations=1,
            joint_zero_offsets=True)
CONTACT = {
    'nominal': {},
    'fine': {'dt': .000125},
    'heavy_low_friction': {'mass':.30, 'friction':.5},
    'offset_pick': {'offset':.004},
    'tight_socket': {'width':.048, 'offset':-.003},
    'no_feedforward': {'feedforward':False},
    'no_grasp': {'grasp':False},
}
WRENCH = {
    'nominal': {},
    'half_step': {'dt':.000125},
    'quarter_step': {'dt':.0000625},
    'heavy_load': {'mass':.30},
    'initial_offset': {'initial_offset':True},
    'no_feedforward': {'feedforward':False},
}

def get_case(mode, name):
    if mode not in ('contact','wrench'):
        raise ValueError('Mode must be contact or wrench')
    suite = CONTACT if mode == 'contact' else WRENCH
    if name not in suite:
        raise ValueError(f'Unknown {mode} case {name}; choose {list(suite)}')
    return dict(BASE, **suite[name])
