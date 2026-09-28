"""Version-aware Isaac startup and shutdown helpers.

Imports of Isaac/Omniverse remain lazy so the controller and offline tests do
not load Isaac's native libraries. These helpers never step/reset the world,
attach a different stage, or substitute another physics engine for PhysX.
"""
from __future__ import annotations

import inspect
from numbers import Integral


def select_physx_engine():
    """Select PhysX before World creation on multi-engine Isaac versions."""
    from isaacsim.core.simulation_manager import SimulationManager

    get_engine = getattr(SimulationManager, "get_active_physics_engine", None)
    if get_engine is None:
        # Isaac 4.5/5.x do not expose the multi-engine selection interface.
        return {"engine": "physx", "engine_selection": "legacy PhysX runtime"}
    previous = get_engine()
    if previous != "physx":
        switch = getattr(SimulationManager, "switch_physics_engine", None)
        if switch is None:
            raise RuntimeError(f"PhysX is required, but the active engine is {previous!r}")
        switch("physx")
    active = get_engine()
    if active != "physx":
        raise RuntimeError(f"Could not select PhysX; active engine is {active!r}")
    return {"engine": active, "engine_selection": "SimulationManager",
            "previous_engine": previous}


def current_stage_id(expected_stage=None):
    """Return the real USD-context stage ID, not the tensor API's -1 default."""
    import omni.usd

    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open; build the scene before initializing tensors")
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeError("The active USD stage changed during physics initialization")
    stage_id = context.get_stage_id()
    if isinstance(stage_id, bool) or not isinstance(stage_id, Integral) or stage_id <= 0:
        raise RuntimeError(f"Invalid USD stage ID {stage_id!r}; cannot bind PhysX tensors")
    return int(stage_id)


def create_physx_numpy_view(stage_id):
    """Bind a CPU/NumPy view to the stage already initialized by World.reset.

    Omni Physics 110.x cannot always resolve stage_id=-1 through the legacy
    PhysX simulation interface. Pass the actual USD stage ID explicitly, as
    Isaac's SimulationManager does. Older tensor APIs lack the `backend`
    keyword, but support `stage_id`; never retry without the explicit ID.
    """
    import omni.physics.tensors as tensors

    if isinstance(stage_id, bool) or not isinstance(stage_id, Integral) or stage_id <= 0:
        raise ValueError("A positive, explicit USD stage ID is required")
    kwargs = {"stage_id": int(stage_id)}
    try:
        parameters = inspect.signature(tensors.create_simulation_view).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "backend" in parameters:
        kwargs["backend"] = "physx"
    try:
        view = tensors.create_simulation_view("numpy", **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize the NumPy PhysX tensor view for USD stage {stage_id}. "
            "The same stage must contain the physics scene and must have been "
            f"initialized with World.reset(). Original error: {exc}"
        ) from exc
    if view is None:
        raise RuntimeError(f"PhysX returned no tensor view for USD stage {stage_id}")
    return view


def shutdown_configuration(app_type, mode="auto"):
    """Use supported fast shutdown without losing nonzero application status.

    Modern SimulationApp.close(exit_code=...) preserves failure codes while
    avoiding full native extension unload. Legacy versions without that API
    use graceful shutdown. Explicit fast mode on such versions is rejected.
    This is for standalone launch.py processes, not embedded/notebook usage.
    """
    if mode not in {"auto", "fast", "graceful"}:
        raise ValueError(f"Unknown shutdown mode: {mode!r}")
    try:
        supports_exit_code = "exit_code" in inspect.signature(app_type.close).parameters
    except (TypeError, ValueError, AttributeError):
        supports_exit_code = False
    if mode == "fast" and not supports_exit_code:
        raise RuntimeError("Fast shutdown requires SimulationApp.close(exit_code=...). "
                           "Use --shutdown-mode auto or graceful with this runtime.")
    fast = mode != "graceful" and supports_exit_code
    return dict(requested=mode, selected="fast" if fast else "graceful",
                fast_shutdown=fast, supports_exit_code=supports_exit_code,
                skip_cleanup=False)


def close_simulation_app(app, exit_code):
    """Save case evidence first; keep normal app cleanup and the requested status.

    Application-owned resources are released before this call. Do not use
    skip_cleanup=True or an unconditional os._exit(0) to hide native failures.
    The parent must still inspect the actual process exit after this call.
    """
    try:
        parameters = inspect.signature(app.close).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "exit_code" in parameters:
        app.close(exit_code=int(exit_code))
    else:
        app.close()
