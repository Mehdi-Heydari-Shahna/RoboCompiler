"""Lifecycle-safe access to Isaac-owned PhysX views and their tensor frontend.

Importing this module does not import Isaac, Omni, USD, Warp, or Torch. All
simulation objects are supplied by the already-running SimulationApp. The
controller remains NumPy/CPU; this is not an alternate physics engine.
"""
from __future__ import annotations

import importlib
from typing import Any
import numpy as np


class PhysicsBindingError(RuntimeError):
    """The active native simulation cannot be used safely."""


def require_methods(obj: Any, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not callable(getattr(obj, name, None))]
    if missing:
        raise PhysicsBindingError(f'{label} lacks required native methods: {", ".join(missing)}')


def managed_simulation_view(world: Any, manager: Any = None, audit: dict | None = None):
    """Reuse the view created for this World, never create an ambient-stage view.

    World.reset() must precede this call. Isaac 6.1 creates its views with an
    explicit stage_id. Calling create_simulation_view('numpy') independently
    uses the factory's default stage, which is not a safe binding in 6.1.

    The current World view is preferred (preserves the requested NumPy
    frontend). The SimulationManager accessor also supports a Warp-owned view.
    A missing/invalid view triggers one documented initialize_physics call;
    continued failure is fatal. No stepping, stage detachment, private-manager
    mutation, sleep, or global TensorAPI reset is used as a recovery trick.
    """
    audit = {} if audit is None else audit
    audit.update(view_origin=None, explicit_reinitializations=0,
                 application_created_extra_simulation_view=False,
                 application_warmup_steps=0, SDK_initialization_may_warm_start=True)
    if manager is None:
        from isaacsim.core.simulation_manager import SimulationManager
        manager = SimulationManager
    engine_getter = getattr(manager, 'get_active_physics_engine', None)
    if callable(engine_getter):
        engine = str(engine_getter()).lower()
        audit['active_physics_engine'] = engine
        if engine != 'physx':
            raise PhysicsBindingError(f'Expected PhysX, active engine is {engine!r}; refusing a different engine')
    else:
        # Older versions expose no engine selector; the authored scene is
        # independently checked to use PhysX by native.audit_active_stage.
        audit['active_physics_engine'] = 'API_NOT_EXPOSED'

    def select():
        candidates = [('World.physics_sim_view', getattr(world, 'physics_sim_view', None))]
        for name in ('get_physics_simulation_view', 'get_physics_sim_view'):
            getter = getattr(manager, name, None)
            if callable(getter):
                candidates.append(('SimulationManager.' + name, getter()))
        seen = set()
        for origin, view in candidates:
            if view is None or id(view) in seen:
                continue
            seen.add(id(view))
            require_methods(view, ('check', 'set_subspace_roots', 'create_articulation_view',
                                  'create_rigid_contact_view', 'update_articulations_kinematic'), origin)
            if bool(view.check()):
                audit['view_origin'] = origin
                return view
        return None

    view = select()
    if view is None:
        require_methods(world, ('initialize_physics',), 'World')
        audit['explicit_reinitializations'] = 1
        world.initialize_physics()
        view = select()
    if view is None:
        raise PhysicsBindingError('World.reset()/initialize_physics() did not produce a valid managed '
                                  'PhysX simulation view. See startup_diagnostics.json and console.log.')
    # This project owns the whole World. Global roots keep all logged poses and
    # forces in the same world frame; no replicated subspace origins are used.
    view.set_subspace_roots('/')
    audit['managed_view_valid'] = True
    return view


class TensorIO:
    """Marshal CPU arrays using the frontend actually returned by PhysX.

    All native state remains measured. Shape checks permit flat scalar Warp
    buffers but never discard data. Unknown tensor types and CUDA pipelines
    are rejected rather than guessed or silently moved to the CPU.
    """
    def __init__(self, sample: Any):
        if isinstance(sample, np.ndarray):
            self.frontend = 'numpy'
            self.module = np
        else:
            module_name = type(sample).__module__.split('.')[0]
            if module_name not in ('warp', 'torch') or not callable(getattr(sample, 'numpy', None)):
                raise PhysicsBindingError(f'Unsupported native tensor type: {type(sample)!r}')
            device = getattr(sample, 'device', None)
            if str(device) != 'cpu':
                raise PhysicsBindingError(f'Expected CPU native tensors, got {device!s}')
            self.frontend = module_name
            self.module = importlib.import_module(module_name)
        self.read(sample, name='frontend probe')

    @staticmethod
    def read(value: Any, shape: tuple[int, ...] | None = None, name: str = 'native tensor') -> np.ndarray:
        if isinstance(value, np.ndarray):
            out = value
        else:
            module_name = type(value).__module__.split('.')[0]
            if module_name not in ('warp', 'torch'):
                raise PhysicsBindingError(f'{name}: unsupported tensor type {type(value)!r}')
            if str(getattr(value, 'device', None)) != 'cpu':
                raise PhysicsBindingError(f'{name}: expected a CPU tensor')
            if module_name == 'torch':
                value = value.detach()
            out = np.asarray(value.numpy())
        if out.dtype.kind not in 'fiu' or not np.isfinite(out).all():
            raise PhysicsBindingError(f'{name}: expected finite real numeric data, got {out.dtype}')
        if shape is not None:
            if out.size != int(np.prod(shape)):
                raise PhysicsBindingError(f'{name}: native shape {out.shape} cannot represent {shape}')
            # Warp tensor frontends may expose flattened scalar/vector buffers.
            out = out.reshape(shape)
        return out

    def array(self, value: Any, *, indices: bool = False):
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu':
            raise PhysicsBindingError('Cannot submit non-real or non-numeric data to native physics')
        if indices:
            raw = np.asarray(value)
            if raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw > np.iinfo(np.int32).max):
                raise PhysicsBindingError('Articulation indices must be nonnegative 32-bit integers')
            dtype = np.int32 if self.frontend == 'torch' else np.uint32
        else:
            dtype = np.float32
        host = np.ascontiguousarray(value, dtype=dtype)
        if not np.isfinite(host).all():
            raise PhysicsBindingError('Cannot submit nonfinite data to native physics')
        if self.frontend == 'numpy':
            return host
        if self.frontend == 'torch':
            return self.module.from_numpy(host)
        return self.module.from_numpy(host,
            dtype=self.module.uint32 if indices else self.module.float32, device='cpu')

    def buffer(self, host: np.ndarray):
        """Return an owned host/native buffer pair; never assume zero-copy works."""
        return TensorBuffer(self, host)


class TensorBuffer:
    """Persistent command buffer; verifies sharing before using it as zero-copy."""
    def __init__(self, io: TensorIO, host: np.ndarray):
        if host.dtype != np.float32 or not host.flags.c_contiguous:
            raise PhysicsBindingError('Persistent buffers must be C-contiguous float32 arrays')
        self.io, self.host = io, host
        self._native = io.array(host)
        self.shared = np.shares_memory(io.read(self._native), host)

    @property
    def native(self):
        # If a future CPU frontend copies, keep commands correct at the cost of
        # conversion. The measured runtime factor reports the actual cost.
        if not self.shared:
            self._native = self.io.array(self.host)
        return self._native


def fill_disturbance_buffers(model, poses: np.ndarray, body_map: np.ndarray, push: np.ndarray,
                             forces: np.ndarray, positions: np.ndarray) -> None:
    """Apply the source torso disturbance at its COM, without spurious moment.

    Position=None in PhysX applies at the actor/link transform, not necessarily
    the COM. Source MuJoCo xfrc_applied and our work observer use the COM.
    Arrays have the documented (articulation_count, link_count, 3) shape.
    """
    forces.fill(0.)
    forces[0, body_map[model.torso]] = push
    com_world = poses[:, :3, 3] + np.einsum('bij,bj->bi', poses[:, :3, :3], model.com_local)
    positions[0, body_map] = com_world
