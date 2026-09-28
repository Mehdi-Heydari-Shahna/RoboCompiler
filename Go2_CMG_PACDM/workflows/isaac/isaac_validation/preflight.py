"""Measured free-flight acceleration and FK checks against the CMG backend."""
import numpy as np


def audit(world, robot, worker, info, cmg):
    checks, samples = [], []

    def check(name, value, limit):
        checks.append(dict(name=name, value=float(value), limit=float(limit),
                           passed=bool(np.isfinite(value) and value <= limit)))

    masses = robot.view.get_masses()[0]
    authored = {b['id']: b['mass_kg'] for b in cmg['bodies']}
    mass_error = max(abs(float(m)-authored[name]) for m, name in zip(masses, robot.link_names))
    check('PhysX imported link masses / kg', mass_error, 2e-6)
    records = {b['id']: b for b in cmg['bodies']}
    com_poses = np.asarray(robot.view.get_coms()[0], dtype=float)
    expected_coms = np.asarray([records[name]['com_m'] for name in robot.link_names])
    check('PhysX imported link COM translations / m', np.max(abs(com_poses[:, :3]-expected_coms)), 2e-7)
    # Tensor API 107.3: get_inertias gives column-major inertia tensors AT the
    # COM, expressed in the RIGID-BODY-PRIM frame. They are not expressed in
    # the principal-axes frame returned by get_coms; do not rotate them again.
    imported_inertias = np.asarray(robot.view.get_inertias()[0], dtype=float).reshape(-1, 3, 3).transpose(0, 2, 1)
    expected_inertias = np.asarray([records[name]['inertia_kg_m2'] for name in robot.link_names])
    check('PhysX imported full link inertia tensors / kg m2', np.max(abs(imported_inertias-expected_inertias)), 2e-7)
    # Read back real physics properties; USD authoring alone is insufficient.
    arm = robot.view.get_dof_armatures()[0, robot.order]
    check('PhysX imported joint armature / kg m2', np.max(abs(arm-np.asarray(info['armature'])[6:])), 1e-7)
    check('PhysX drive stiffness disabled', np.max(abs(robot.view.get_dof_stiffnesses())), 0.)
    check('PhysX drive damping disabled', np.max(abs(robot.view.get_dof_dampings())), 0.)
    # Recent PhysX versions expose static, dynamic and viscous terms together.
    # Older Isaac distributions provide only the legacy coefficient getter.
    native_friction = (robot.view.get_dof_friction_properties()
                       if hasattr(robot.view, 'get_dof_friction_properties')
                       else robot.view.get_dof_friction_coefficients())
    check('PhysX native joint friction disabled', np.max(abs(native_friction)), 0.)
    rng = np.random.default_rng(48271)
    initial = np.asarray(info['q'], float)
    try:
        for i in range(6):
            q = initial.copy()
            q[:3] = [0., 0., 2.]
            q[3:6] = rng.uniform(-.2, .2, 3)
            q[6:] += rng.uniform(-.07, .07, 12)
            v = np.zeros(18) if i < 3 else rng.uniform(-.2, .2, 18)
            motor = rng.uniform(-3., 3., 12)
            for dt in (.0001, .00005):
                world.set_simulation_dt(physics_dt=dt, rendering_dt=.04)
                robot.set_state(q, v)
                observed_q, observed_v = robot.state()
                check(f'Pose initialization roundtrip sample {i}, dt={dt}', np.max(abs(observed_q-q)), 5e-6)
                check(f'Velocity initialization roundtrip sample {i}, dt={dt}', np.max(abs(observed_v-v)), 5e-6)
                mechanics = worker.audit_state(observed_q, observed_v)
                fk = np.max(abs(robot.foot_positions()-np.asarray(mechanics['feet'])))
                check(f'FK sample {i}, dt={dt} / m', fk, 2e-5)
                mass = np.asarray(mechanics['mass']) + np.diag(info['armature'])
                bias = np.asarray(mechanics['bias'])
                # Passive terms are explicit in both engines: smooth Coulomb is declared.
                passive = -(np.asarray(info['damping'])*observed_v +
                            np.asarray(info['frictionloss'])*np.tanh(observed_v/.02))
                expected = np.linalg.solve(mass, np.r_[np.zeros(6), motor] + passive - bias)
                robot.effort(motor + passive[6:])
                world.step(render=False)
                _, after = robot.state()
                measured = (after-observed_v)/dt
                relative = np.linalg.norm(measured-expected)/max(1., np.linalg.norm(expected))
                check(f'Free-flight forward dynamics sample {i}, dt={dt}', relative, .02)
                samples.append(dict(sample=i, dt_s=dt, q=observed_q.tolist(), v=observed_v.tolist(),
                                    motor_torque=motor.tolist(), expected_acceleration=expected.tolist(),
                                    measured_acceleration=measured.tolist(), relative_error=float(relative)))
    finally:
        robot.effort(np.zeros(12))
    return dict(passed=all(x['passed'] for x in checks), checks=checks, samples=samples,
                engine='Isaac Sim / PhysX',
                scope='Native PhysX FK, property readback and free-flight forward dynamics versus independent CMG/Pinocchio; no contact forces imposed.')
