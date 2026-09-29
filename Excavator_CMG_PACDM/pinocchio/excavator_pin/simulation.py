"""Pinocchio plant + original PACDM controller + original hydraulics + original mission.

Execution per physics step, in the same order as the original ``run_soil.py``:

1. the original ``SoilMission.arm_reference`` supplies the arm reference;
2. every 10 ms the original online-PACDM controller (fixed-base subclass)
   computes eight requested port efforts from the measured plant state;
3. the original ``HydraulicBank.evaluate`` converts pressures into the actual
   port efforts, which are held for the step (zero-order hold, as native
   actuator controls are held over a MuJoCo step);
4. every 40 ms the original ``SoilMission.update`` evaluates the guards with
   the step-start state (the loop stops before integrating once the mission is
   done);
5. the reduced closed-chain plant is advanced by classical RK4.  Its state is
   the seven independent coordinates and rates, port work, soil work and four
   payload-mass ledgers.  Dependent coordinates are solved from native
   Pinocchio closure equations at every stage; accelerations come from native
   Pinocchio ``constraintDynamics``;
6. the original ``HydraulicBank.advance`` performs its original explicit
   pressure update.

MuJoCo is never imported.  The soil bed is the declared surrogate in
``soil.py``; there are no particles and no contact solver.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

from . import paths, sentinel
from .engine import (FixedBaseArmController, ControllerView, context, PORTS)
from .plant import ReducedPlant, scaled_body_cmg
from .soil import SoilParameters, SoilSurrogate, BucketGeometry

sentinel.install()
paths.add_original_to_path()
from soil_mission import SoilMission  # noqa: E402  (preserved original)
from hydraulic_actuators import HydraulicBank, PARAMETERS as HYDRAULIC_PARAMETERS  # noqa: E402
from digging_export import SECTION, BUCKET_X  # noqa: E402
from digging_path import LIP, HEEL  # noqa: E402

HYDRAULIC_KEYS = ['supply', 'throttle', 'leakage', 'relief', 'friction',
                  'compressibility_geometry', 'rotary', 'mechanical']

# Terrain settings are those of RUN_RECIPES.json in the preserved package.
FILLED_BED = dict(entry_slot_m=0.0, cut_depth_m=.11, cut_policy='position-only')
EXPOSED_FACE = dict(entry_slot_m=.9, cut_depth_m=.18, cut_policy='load-aware')

CASES = {
    'nominal': dict(dt=1e-3, **FILLED_BED, role='positive'),
    'half_step': dict(dt=5e-4, **FILLED_BED, role='positive'),
    'quarter_step': dict(dt=2.5e-4, **FILLED_BED, role='positive'),
    'exposed_face': dict(dt=1e-3, **EXPOSED_FACE, role='positive'),
    'heavy_soil': dict(dt=1e-3, **FILLED_BED, role='positive',
                       soil=dict(resistance_factor_N_gamma=14., bulk_density_kg_m3=1500.)),
    'model_mismatch': dict(dt=1e-3, **FILLED_BED, role='positive', bucket_mass_scale=1.25),
    'initial_offset': dict(dt=1e-3, **FILLED_BED, role='positive',
                           initial_offset_rad=[.012, -.010, .008, .010, -.006, .004, 0.]),
    'empty_bed': dict(dt=1e-3, **EXPOSED_FACE, role='negative_control',
                      soil=dict(enabled=False)),
    # Load-sensitivity case (added after independent review): N_gamma chosen so that
    # N_gamma*rho*g*w*d^2 at the 0.11 m cut depth equals the median MuJoCo grain-contact
    # force on the bucket in the soil_demo draw phase (18.4 kN); the load-aware cut policy
    # is the original package's policy for load-limited cuts; the velocity regularization
    # is 0.1 m/s because 0.02 m/s made the 1 ms RK4 step inaccurate at this load level.
    # The mission outcome is reported, not gated; all numerical gates apply.
    'stress_load': dict(dt=1e-3, entry_slot_m=0.0, cut_depth_m=.11, cut_policy='load-aware', role='stress',
                        soil=dict(resistance_factor_N_gamma=145., velocity_regularization_m_s=.1)),
    # Same load with the position-only policy of the original soil_demo recipe (outcome reported).
    'stress_load_position_only': dict(dt=1e-3, **FILLED_BED, role='stress',
                                      soil=dict(resistance_factor_N_gamma=145., velocity_regularization_m_s=.1)),
    # Step refinement of the initial hydraulic transient (hydraulic-ledger convergence).
    'initial_offset_half_step': dict(dt=5e-4, **FILLED_BED, role='positive',
                                     initial_offset_rad=[.012, -.010, .008, .010, -.006, .004, 0.]),
    # Numerical negative control: the stress load with the default 0.02 m/s velocity
    # regularization is too stiff for 1 ms RK4; the energy-balance gate must reject it.
    'stiff_soil_regularization': dict(dt=1e-3, **FILLED_BED, role='negative_numerics', duration_s=12.,
                                      soil=dict(resistance_factor_N_gamma=145.)),
}


def case_parameters(name):
    spec = dict(CASES[name])
    spec.setdefault('soil', {})
    spec.setdefault('bucket_mass_scale', 1.0)
    spec.setdefault('initial_offset_rad', [0.] * 7)
    spec.setdefault('duration_s', 65.0)
    return spec


class Experiment:
    """One closed-loop run; owns the plant, controller, bank and surrogate."""

    def __init__(self, name, spec):
        self.name, self.spec = name, spec
        self.dt = float(spec['dt'])
        # Integer step rate: time is step/rate exactly, so guard instants and
        # the 10 ms controller / 40 ms mission ticks coincide at every dt level.
        self.rate = int(round(1. / self.dt))
        if self.dt <= 0 or abs(1. / self.rate - self.dt) > 1e-15 or self.rate % 100:
            raise ValueError('dt must be 1/N s with N a multiple of 100 (divides the 10 ms control period)')
        started = time.perf_counter()
        self.cmg, self.mapping, self.e, self.a, self.r, self.inverse = context()
        self.mission = SoilMission(self.e, self.r, cut_depth=spec['cut_depth_m'],
                                   cut_policy=spec['cut_policy'])
        plant_cmg = (scaled_body_cmg(self.cmg, 'body_56', spec['bucket_mass_scale'])
                     if spec['bucket_mass_scale'] != 1. else self.cmg)
        self.plant = ReducedPlant(plant_cmg, self.e.cut_ids, self.e.independent_ids)
        pm = self.plant.model
        self.geometry = BucketGeometry(SECTION, BUCKET_X, LIP, HEEL)
        pm.add_bucket_point('centroid', self.geometry.centroid_local)
        soil = SoilParameters(entry_slot_m=spec['entry_slot_m'],
                              receiver_center_xy_m=tuple(self.mission.depot_center[:2]),
                              receiver_half_size_m=tuple(self.mission.depot_half_size))
        self.soil = SoilSurrogate(soil.scaled(**spec['soil']), self.geometry)
        # Coordinate maps: controller order is the original PinocchioBackend BFS order.
        self.perm = np.array([pm.index[j] for j in self.e.tree_ids])
        ports_ctrl = np.array([self.e.tree_ids.index(k) for k in PORTS])
        self.ports = self.perm[ports_ctrl]
        self.E = np.zeros((self.plant.n, 8))
        self.E[self.ports, np.arange(8)] = 1.
        q0 = np.empty(self.plant.n)
        q0[self.perm] = self.mission.initial_tree
        offset = np.asarray(spec['initial_offset_rad'], dtype=float)
        self.plant.initialize(q0)
        if np.any(offset):
            q_offset, _, _, _ = self.plant.project(q0[self.plant.active] + offset)
        else:
            q_offset = q0
        self.plant.initialize(q_offset)
        self.controller = FixedBaseArmController(self.cmg, self.mapping, self.e, self.a,
                                                 self.r, self.inverse, self.mission.initial_tree)
        self.setup_wall_s = time.perf_counter() - started
        self.maxima = dict(projection_residual_m=0., tangency_m_s=0., constraint_acceleration=0.,
                           tangent_dynamics_relative=0., native_vs_reduced_relative=0.,
                           newton_iterations=0)

    # ------------------------------------------------------------------ helpers
    def to_controller(self, x):
        return np.asarray(x)[self.perm]

    def kinematics(self, y):
        """Closed configuration, exact tangent velocity, bucket points and soil load."""
        plant, pm = self.plant, self.plant.model
        u, ud = y[:7], y[7:14]
        q, jacobian, tangent, error = plant.project(u)
        v = tangent @ ud
        pts, rotation = pm.points(q, ('lip', 'mouth', 'centroid'), refresh=False)
        lip, j_lip = pts['lip']
        mouth, _ = pts['mouth']
        centroid, j_centroid = pts['centroid']
        lip_velocity = j_lip @ v
        normal = rotation @ self.geometry.mouth_normal_local
        soil = self.soil.evaluate(lip, lip_velocity, mouth, normal, y[16])
        tau_soil = j_lip.T @ soil['cutting_force_N'] + j_centroid.T @ soil['weight_force_N']
        return dict(y=y, q=q, v=v, jacobian=jacobian, tangent=tangent, error=error, lip=lip,
                    lip_velocity=lip_velocity, j_centroid=j_centroid, mouth=mouth,
                    centroid=centroid, normal=normal, soil=soil, tau_soil=tau_soil)

    def evaluate(self, y, effort, *, check_reduced=False, context=None):
        """RHS of the RK4 state and all diagnostics at this stage."""
        plant, pm = self.plant, self.plant.model
        c = self.kinematics(y) if context is None else context
        q, v, jacobian, tangent, error = c['q'], c['v'], c['jacobian'], c['tangent'], c['error']
        soil, lip_velocity, j_centroid = c['soil'], c['lip_velocity'], c['j_centroid']
        ud = y[7:14]
        tau = self.E @ effort + c['tau_soil']
        acceleration = plant.forward(q, v, tau)
        # Independent certification of the native solution: separate CRBA/NLE
        # calls and native classical-acceleration drift, not solver internals.
        mass, bias = pm.mass_bias(q, v)
        gamma = pm.gamma(q, v)
        constraint_acceleration = float(np.max(np.abs(jacobian @ acceleration + gamma)))
        tangent_residual = tangent.T @ (mass @ acceleration + bias - tau)
        scale = max(1., float(np.max(np.abs(tangent.T @ tau))), float(np.max(np.abs(tangent.T @ bias))))
        m = self.maxima
        m['projection_residual_m'] = max(m['projection_residual_m'], error)
        m['tangency_m_s'] = max(m['tangency_m_s'], float(np.max(np.abs(jacobian @ v))))
        m['constraint_acceleration'] = max(m['constraint_acceleration'], constraint_acceleration)
        m['tangent_dynamics_relative'] = max(m['tangent_dynamics_relative'],
                                             float(np.max(np.abs(tangent_residual))) / scale)
        if check_reduced:
            # Force-equivalent (backward-error) difference, as in mechanics.py.
            reduced = plant.reduced(q, v, tau, jacobian, tangent, gamma)
            force_scale = max(1., float(np.max(np.abs(tau))), float(np.max(np.abs(bias))))
            m['native_vs_reduced_relative'] = max(
                m['native_vs_reduced_relative'],
                float(np.max(np.abs(mass @ (reduced - acceleration)))) / force_scale)
        port_power = float(effort @ v[self.ports])
        soil_power = float(soil['cutting_force_N'] @ lip_velocity + soil['weight_force_N'] @ (j_centroid @ v))
        derivative = np.r_[ud, acceleration[plant.active], port_power, soil_power,
                           soil['fill_rate_kg_s'] - soil['release_rate_kg_s'], soil['fill_rate_kg_s'],
                           soil['deposit_rate_kg_s'], soil['spill_rate_kg_s']]
        return derivative

    # ---------------------------------------------------------------------- run
    def run(self, results_dir, *, duration=None, progress=True):
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        dt = self.dt
        duration = float(self.spec['duration_s'] if duration is None else duration)
        steps = round(duration * self.rate)
        control_every = round(.01 / dt)
        mission_every = round(.04 / dt)
        record_every = round(.01 / dt)
        plant, pm, mission = self.plant, self.plant.model, self.mission
        started = time.perf_counter()
        u0 = plant.q_cache[plant.active].copy()
        # State: u(7) ud(7) port_work soil_work bucket_mass cut_mass deposited spilled
        y = np.r_[u0, np.zeros(7), 0., 0., 0., 0., 0., 0.]
        q_now = plant.q_cache.copy()
        v_now = np.zeros(plant.n)
        desired = mission.arm_reference(0.)
        command = self.controller.evaluate(*desired, ControllerView(
            self.to_controller(q_now), self.to_controller(v_now), 0.))
        bank = HydraulicBank(self.r.nominal[[self.e.tree_ids.index(k) for k in PORTS]], command)
        kinetic0, potential0 = pm.energy(q_now, v_now)
        energy0 = kinetic0 + potential0
        fluid0 = bank.stored_energy(q_now[self.ports])
        hydraulic_work = np.zeros(len(HYDRAULIC_KEYS))
        hydraulic_absolute = np.zeros(len(HYDRAULIC_KEYS))
        keys = ['time', 'phase', 'u', 'ud', 'desired', 'desired_velocity', 'tracking_error',
                'q', 'v', 'command', 'effort', 'pressure', 'rotary', 'flow', 'valve_saturated',
                'pressure_limit_active', 'lip', 'lip_velocity', 'mouth', 'soil_depth',
                'cutting_force', 'payload_mass', 'cut_mass', 'deposited_mass', 'spilled_mass',
                'capacity', 'kinetic_energy', 'potential_energy', 'port_work', 'soil_work',
                'mechanical_balance', 'hydraulic_work', 'fluid_energy', 'fluid_identity_W',
                'mass_ledger_error']
        trace = {k: [] for k in keys}
        extrema = dict(tracking_rad=0., mechanical_balance_J=0., mass_ledger_kg=0.,
                       fluid_identity_W=0., pressure_min_Pa=np.inf, pressure_max_Pa=0.,
                       valve_saturated_steps=0, pressure_limit_steps=0, peak_effort=np.zeros(8),
                       peak_command=np.zeros(8), peak_cutting_force_N=0., peak_payload_kg=0.,
                       peak_speed=0., max_passive_prismatic_m=0.)
        status, error = 'INCOMPLETE', None
        last_print = -1.
        executed_steps = 0
        try:
            for step in range(steps + 1):
                t = step / self.rate
                desired = mission.arm_reference(t)
                info = self.kinematics(y)  # measured plant state at step start
                q_now, v_now = info['q'], info['v']
                executed_steps = step + 1
                if step % control_every == 0:
                    command = self.controller.evaluate(*desired, ControllerView(
                        self.to_controller(q_now), self.to_controller(v_now), t))
                item = bank.evaluate(q_now[self.ports], v_now[self.ports], command, dt)
                effort = item['effort']
                tracking = float(np.max(np.abs(y[:6] - desired[0][:6])))
                kinetic, potential = pm.energy(q_now, v_now)
                balance = kinetic + potential - energy0 - y[14] - y[15]
                ledger = abs(y[17] - y[16] - y[18] - y[19])
                ex = extrema
                ex['tracking_rad'] = max(ex['tracking_rad'], tracking)
                ex['mechanical_balance_J'] = max(ex['mechanical_balance_J'], abs(balance))
                ex['mass_ledger_kg'] = max(ex['mass_ledger_kg'], ledger)
                ex['fluid_identity_W'] = max(ex['fluid_identity_W'], abs(item['fluid_identity_W']))
                ex['pressure_min_Pa'] = min(ex['pressure_min_Pa'], float(bank.pressure.min()))
                ex['pressure_max_Pa'] = max(ex['pressure_max_Pa'], float(bank.pressure.max()))
                ex['valve_saturated_steps'] += int(item['valve_saturated'])
                ex['pressure_limit_steps'] += int(item['pressure_limit_active'])
                ex['peak_effort'] = np.maximum(ex['peak_effort'], np.abs(effort))
                ex['peak_command'] = np.maximum(ex['peak_command'], np.abs(command))
                ex['peak_cutting_force_N'] = max(ex['peak_cutting_force_N'],
                                                 float(np.linalg.norm(info['soil']['cutting_force_N'])))
                ex['peak_payload_kg'] = max(ex['peak_payload_kg'], y[16])
                ex['peak_speed'] = max(ex['peak_speed'], float(np.max(np.abs(y[7:14]))))
                if step % record_every == 0 or step == steps:
                    values = dict(time=t, phase=mission.phase, u=y[:7], ud=y[7:14], desired=desired[0],
                                  desired_velocity=desired[1], tracking_error=tracking, q=q_now, v=v_now,
                                  command=command, effort=effort, pressure=bank.pressure, rotary=bank.rotary,
                                  flow=item['flow'], valve_saturated=item['valve_saturated'],
                                  pressure_limit_active=item['pressure_limit_active'], lip=info['lip'],
                                  lip_velocity=info['lip_velocity'], mouth=info['mouth'],
                                  soil_depth=info['soil']['depth_m'],
                                  cutting_force=info['soil']['cutting_force_N'], payload_mass=y[16],
                                  cut_mass=y[17], deposited_mass=y[18], spilled_mass=y[19],
                                  capacity=info['soil']['capacity_kg'], kinetic_energy=kinetic,
                                  potential_energy=potential, port_work=y[14], soil_work=y[15],
                                  mechanical_balance=balance, hydraulic_work=hydraulic_work,
                                  fluid_energy=bank.stored_energy(q_now[self.ports]),
                                  fluid_identity_W=item['fluid_identity_W'], mass_ledger_error=ledger)
                    for key in keys:
                        trace[key].append(np.array(values[key], copy=True))
                if step % mission_every == 0 or step == steps:
                    mission.update(t, tracking, 0., float(y[16]), float(info['lip'][2]),
                                   float(y[18]), lip_x=float(info['lip'][0]))
                    if mission.done:
                        status = 'TASK_FAILED' if mission.failures else 'COMPLETED'
                        break
                if progress and t - last_print >= 2. - 1e-9:
                    last_print = t
                    print(f"{self.name}: t={t:5.2f}s {mission.names[mission.phase]:22s} err={tracking:.4f} "
                          f"payload={y[16]:6.1f}kg dep={y[18]:6.1f}kg F={np.linalg.norm(info['soil']['cutting_force_N']):7.0f}N "
                          f"balance={balance:.2e}J wall={time.perf_counter()-started:.0f}s", flush=True)
                if step == steps:
                    break
                k1 = self.evaluate(y, effort, check_reduced=True, context=info)
                k2 = self.evaluate(y + .5 * dt * k1, effort)
                k3 = self.evaluate(y + .5 * dt * k2, effort)
                k4 = self.evaluate(y + dt * k3, effort)
                y = y + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
                if not np.all(np.isfinite(y)):
                    raise FloatingPointError('Nonfinite integrated plant state')
                power = np.array([item['power'][k] for k in HYDRAULIC_KEYS])
                hydraulic_work += dt * power
                hydraulic_absolute += dt * np.abs(power)
                bank.advance(item, dt)
        except Exception as exc:  # retained as evidence, never silently passed
            import traceback
            status, error = 'ERROR', traceback.format_exc()
            print(error, flush=True)
        wall = time.perf_counter() - started
        arrays = {k: np.asarray(v) for k, v in trace.items()}
        np.savez_compressed(results_dir / f'{self.name}.npz', **arrays)
        q_end = plant.q_cache
        v_end = plant.N_cache @ y[7:14]
        kinetic, potential = pm.energy(q_end, v_end)
        fluid_end = bank.stored_energy(q_end[self.ports])
        hw = dict(zip(HYDRAULIC_KEYS, map(float, hydraulic_work)))
        ha = dict(zip(HYDRAULIC_KEYS, map(float, hydraulic_absolute)))
        mechanical_change = kinetic + potential - energy0
        total_rhs = (hw['supply'] - hw['throttle'] - hw['leakage'] - hw['relief'] - hw['friction']
                     + hw['compressibility_geometry'] + hw['rotary'] + float(y[15]))
        total_defect = mechanical_change + fluid_end - fluid0 - total_rhs
        activity = sum(ha[k] for k in HYDRAULIC_KEYS if k != 'mechanical')
        ex = extrema
        report = dict(
            case=self.name, spec=self.spec, role=self.spec.get('role'), status=status, error=error, dt_s=dt,
            plant_tree_order=list(pm.tree_ids),
            simulated_duration_s=float(arrays['time'][-1]) if len(arrays['time']) else 0.,
            integration_steps=max(0, executed_steps - 1),
            wall_s=wall, setup_wall_s=self.setup_wall_s,
            phase_names=mission.names, events=mission.events, failures=mission.failures,
            completed_phases=len(mission.events),
            peak_tracking_error_rad=ex['tracking_rad'],
            max_mechanical_balance_error_J=ex['mechanical_balance_J'],
            final_mechanical_balance_J=float(kinetic + potential - energy0 - y[14] - y[15]),
            port_work_J=float(y[14]), soil_work_J=float(y[15]),
            mechanical_energy_change_J=float(mechanical_change),
            mechanical_absolute_port_work_rectangle_J=ha['mechanical'],
            port_work_rectangle_minus_exact_J=float(hw['mechanical'] - y[14]),
            hydraulic_work_J=hw, hydraulic_absolute_work_J=ha,
            fluid_energy_change_J=float(fluid_end - fluid0),
            hydraulic_total_balance_defect_J=float(total_defect),
            hydraulic_total_balance_relative=float(abs(total_defect) / max(1., activity)),
            max_fluid_identity_W=ex['fluid_identity_W'],
            pressure_min_Pa=float(ex['pressure_min_Pa']), pressure_max_Pa=float(ex['pressure_max_Pa']),
            supply_pressure_Pa=float(bank.ps),
            evaluated_steps=executed_steps,
            valve_saturated_step_fraction=ex['valve_saturated_steps'] / max(1, executed_steps),
            pressure_limit_step_fraction=ex['pressure_limit_steps'] / max(1, executed_steps),
            peak_effort_SI=ex['peak_effort'].tolist(), peak_command_SI=ex['peak_command'].tolist(),
            peak_cutting_force_N=ex['peak_cutting_force_N'], peak_payload_kg=float(ex['peak_payload_kg']),
            final_payload_kg=float(y[16]), cut_mass_kg=float(y[17]), deposited_mass_kg=float(y[18]),
            spilled_mass_kg=float(y[19]), max_mass_ledger_error_kg=ex['mass_ledger_kg'],
            peak_independent_speed_SI=ex['peak_speed'],
            plant_numerics=dict(self.maxima, **plant.stats),
            controller=self.controller.diagnostics(),
            soil_surrogate=dict(parameters=self.soil.p.as_dict(), geometry=self.geometry.as_dict(),
                                capacity_kg=self.soil.capacity_kg),
            hydraulic_parameters=HYDRAULIC_PARAMETERS,
            mujoco=sentinel.report(),
            scope='Pinocchio closed-chain plant, fixed undercarriage, declared soil surrogate wrench; '
                  'no particles, contact solver, floating base, tracks or hardware')
        (results_dir / f'{self.name}.json').write_text(json.dumps(report, indent=2, default=_plain,
                                                                  allow_nan=False) + '\n', encoding='utf-8')
        return report


def _plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def run_case(name, results_dir=paths.RESULTS, **overrides):
    spec = case_parameters(name)
    spec.update(overrides)
    try:
        experiment = Experiment(name, spec)
    except Exception:  # setup failure is recorded as evidence, never raised through the runner
        import traceback
        report = dict(case=name, spec=spec, role=spec.get('role'), status='ERROR', error=traceback.format_exc(),
                      mujoco=sentinel.report())
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        (Path(results_dir) / f'{name}.json').write_text(json.dumps(report, indent=2, default=_plain) + '\n',
                                                        encoding='utf-8')
        print(report['error'], flush=True)
        return report
    return experiment.run(results_dir)
