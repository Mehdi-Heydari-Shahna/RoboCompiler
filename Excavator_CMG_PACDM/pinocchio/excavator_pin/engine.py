"""MuJoCo-free adapters around the preserved original controller stack.

The original ``benchmark_model.context()`` constructs ``ClosedDynamics``, whose
``NativeBiasDynamics`` member loads a MuJoCo model.  The Pinocchio backend uses
the same original classes, except that the engine object below supplies only
the Pinocchio and source-record members these classes actually require:

* ``PinocchioOnlyNative`` subclasses the original ``NativeDynamics``.  It
  constructs the original ``PinocchioBackend`` kinematic compiler and calls the
  original ``_make_compact_pinocchio_model`` method.  The MuJoCo half of the
  original option audit is replaced by the Pinocchio half only.
* ``FixedBaseArmController`` subclasses the original v26
  ``TrackedArmController``.  Its constructor repeats the original
  initialization without MuJoCo model lookups; ``evaluate``, ``_assemble``,
  ``_polish`` and ``diagnostics`` are inherited unchanged.  The base state is
  the fixed source-world undercarriage: zero position offset, identity rotation,
  zero velocity and zero acceleration.

No original file is modified.  All imports go through the fail-closed MuJoCo
sentinel installed by the runner.
"""
from __future__ import annotations

import json

import numpy as np

from . import paths, sentinel

sentinel.install()
paths.add_original_to_path()

from cmg_io import load_cmg  # noqa: E402  (preserved original)
from pinocchio_backend import PinocchioBackend  # noqa: E402
from native_dynamics import NativeDynamics  # noqa: E402
from source_bias import SourceBiasDynamics  # noqa: E402
from actuation_maps import ActuationMaps  # noqa: E402
from extended_reference import ExtendedReference  # noqa: E402
from inverse_dynamics import SourceInverseDynamics  # noqa: E402
from benchmark_model import GUARDS, PORTS, INDEPENDENT, GRAVITY  # noqa: E402
from excavator_pacdm import compile_pacdm  # noqa: E402
from floating_dynamics import FloatingSource  # noqa: E402
from tracked_arm import TrackedArmController, _vector  # noqa: E402


class PinocchioOnlyNative(NativeDynamics):
    """Original compact Pinocchio dynamics copy, without the MuJoCo half."""

    def __init__(self, cmg):  # noqa: D401 - replaces only MuJoCo construction
        self.pin_backend = PinocchioBackend(cmg)
        self.tree_ids = list(self.pin_backend.tree_ids)
        self._make_compact_pinocchio_model()  # original method, unchanged
        self._audit_options()

    def _audit_options(self):
        # Pinocchio half of the original NativeDynamics._audit_options.
        for name in ('armature', 'damping', 'friction'):
            if np.any(np.asarray(getattr(self.pin_model, name)) != 0.):
                raise ValueError(f'Unexpected Pinocchio {name} in the rigid-body experiment')


class PinocchioOnlyEngine:
    """Duck-typed replacement for ``ClosedDynamics`` used by the controller stack."""

    def __init__(self, cmg):
        self.native = PinocchioOnlyNative(cmg)
        self.tree_ids = self.native.tree_ids
        self.independent_ids = self.native.pin_backend.independent_ids
        self.active = [self.tree_ids.index(x) for x in self.independent_ids]
        self.passive = [i for i in range(len(self.tree_ids)) if i not in self.active]
        self.cut_ids = self.native.pin_backend.cut_ids
        self.source = SourceBiasDynamics(cmg, self.tree_ids)


class _Options:
    def __init__(self, gravity):
        self.gravity = np.asarray(gravity, dtype=float).copy()


class _GravityOnlyModel:
    """The original controller reads only ``m.opt.gravity`` during evaluation."""

    def __init__(self, gravity):
        self.opt = _Options(gravity)


class ControllerView:
    """Measured plant state presented with the original controller's attributes."""

    def __init__(self, qpos, qvel, time_s):
        self.qpos = np.asarray(qpos, dtype=float).copy()
        self.qvel = np.asarray(qvel, dtype=float).copy()
        self.time = float(time_s)


class FixedBaseArmController(TrackedArmController):
    """Original v26 online-PACDM arm controller with a fixed source-world base."""

    def __init__(self, cmg, mapping, e, a, r, inverse, initial_closed_tree,
                 *, kp=144.0, kd=24.0, gravity=GRAVITY):
        # Mirrors TrackedArmController.__init__ line by line, minus MuJoCo lookups.
        self.e, self.a, self.r, self.inverse = e, a, r, inverse
        self.m = _GravityOnlyModel(gravity)
        self.graph, self.pacdm = compile_pacdm(cmg, mapping, e.tree_ids)
        self.floating = FloatingSource(cmg, e.source)
        self.active = np.asarray(e.active, dtype=int)
        self.qi = np.arange(len(e.tree_ids))
        self.vi = np.arange(len(e.tree_ids))
        record = next(j for j in cmg['joints'] if j['id'] == 'fixed_world_base')
        self.T0 = np.asarray(record['T_BJ']) @ np.linalg.inv(np.asarray(record['T_FJ']))
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (6,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (6,)).copy()
        self.acceleration_filter_s = 0.0
        if (not np.isfinite(self.kp).all() or not np.isfinite(self.kd).all() or
                np.any(self.kp < 0) or np.any(self.kd < 0)):
            # Same validation as the original constructor (tracked_arm.py).
            raise ValueError("Controller gains and filter time constant must be finite and nonnegative")
        self.seed = self.graph.lift(_vector(initial_closed_tree, 23, 'Initial closed tree'))
        self.N, self.mapping = self.pacdm.mapping(self.seed)
        self.initial_acquisition = None
        if self.N is None:
            self.seed, self.initial_acquisition = self.pacdm.acquire(
                self.seed[self.graph.active].copy(), self.seed)
            if not self.initial_acquisition['success']:
                raise ValueError(f'Initial PACDM assembly failed: {self.initial_acquisition}')
            self.N, self.mapping = self.pacdm.mapping(self.seed)
        if self.N is None:
            raise ValueError(f'Initial PACDM mapping failed: {self.mapping}')
        self.seed = self._polish(self.seed, np.asarray(self.mapping['rows'], dtype=int))
        self.N, self.mapping = self.pacdm.mapping(self.seed)
        self.command_peak = np.zeros(8)
        self.shadow_max = 0.0
        self.shadow_velocity_max = 0.0
        self.closure_max = 0.0
        self.tangent_max = 0.0
        self.mapping_difference_max = 0.0
        self.inverse_equilibrium_max = 0.0
        self.acceleration_closure_max = 0.0
        self.base_coupling_force_max = 0.0
        self.base_acceleration_peak = np.zeros(6)
        self.evaluations = self.continuation_steps = self.fallback_count = 0
        self.control_elapsed_s = self.control_max_elapsed_s = 0.0
        self.last_solution = self.last_component = self.last_floating = None
        self.last_base_state = None
        self._filtered_acceleration = None
        self._last_sample_time = None

    def base_state(self, d):
        """Fixed undercarriage at the CMG source-world placement.

        The inherited ``diagnostics()`` still labels the controller output
        'floating_base_feedforward' / 'conditioned on measured base motion'
        (original text); here that base motion is identically zero.
        """
        zero3, zero6 = np.zeros(3), np.zeros(6)
        return dict(position=zero3.copy(), rotation=np.eye(3), velocity=zero6.copy(),
                    acceleration=zero6.copy(), native_position=self.T0[:3, 3].copy(),
                    native_rotation=self.T0[:3, :3].copy(), native_velocity=zero6.copy(),
                    native_acceleration=zero6.copy())


def load_inputs():
    cmg = load_cmg(paths.CMG_PATH)
    mapping = json.loads(paths.MAPPING_PATH.read_text(encoding='utf-8'))
    return cmg, mapping


def context():
    """MuJoCo-free counterpart of the original ``benchmark_model.context``."""
    cmg, mapping = load_inputs()
    e = PinocchioOnlyEngine(cmg)
    a = ActuationMaps(cmg, e.tree_ids)
    r = ExtendedReference(e, a, GUARDS)
    return cmg, mapping, e, a, r, SourceInverseDynamics(a)
