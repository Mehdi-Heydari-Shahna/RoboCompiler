"""Explicit numerical settings with fail-closed USD readback.

Sources: Isaac Sim 6.1 physics limitations/performance handbook and the public
Isaac Lab simulation_context (main, consulted 25 September 2026). These settings
are candidate numerical choices, not a demonstrated native stability result.
"""
from __future__ import annotations
from .control import Config


def app_configuration(cfg: Config, headless: bool) -> dict:
    cfg.validate()
    options={'headless':headless,'renderer':'RayTracedLighting','width':1280,'height':900,
             'anti_aliasing':0,'sync_loads':True,'multi_gpu':False,
             'extra_args':[f'--/persistent/physics/numThreads={cfg.physx_threads}']}
    if headless:options['disable_viewport_updates']=True
    return options


def expected_settings(cfg: Config) -> dict:
    cfg.validate()
    return {'SolverType':cfg.solver_type,
            'MinPositionIterationCount':cfg.solver_position_iterations,
            'MaxPositionIterationCount':cfg.solver_position_iterations,
            'MinVelocityIterationCount':cfg.solver_velocity_iterations,
            'MaxVelocityIterationCount':cfg.solver_velocity_iterations,
            'EnableExternalForcesEveryIteration':cfg.external_forces_every_iteration}


def author_solver_settings(scene_api, cfg: Config) -> dict:
    # The explicit scene bounds prevent a scene default from silently overriding
    # per-articulation settings, especially a requested zero velocity count.
    for key,value in expected_settings(cfg).items():
        setter=getattr(scene_api,'Create'+key+'Attr',None)
        if not callable(setter):
            raise RuntimeError(f'Installed PhysX schema lacks Create{key}Attr; no solver fallback was used')
        setter(value)
    return read_solver_settings(scene_api,cfg)


def read_solver_settings(scene_api, cfg: Config) -> dict:
    actual={}
    for key,wanted in expected_settings(cfg).items():
        getter=getattr(scene_api,'Get'+key+'Attr',None)
        if not callable(getter):raise RuntimeError(f'Installed PhysX schema lacks Get{key}Attr')
        value=getter().Get()
        if value!=wanted:raise RuntimeError(f'Solver readback mismatch: {key}: {value!r} != {wanted!r}')
        actual[key]=value
    return {'status':'PASS','scope':'Authored USD scene settings; native dynamics still require convergence tests',
            'requested_profile':cfg.solver_profile,'actual':actual}


def read_stage_solver_settings(stage,cfg: Config) -> dict:
    from pxr import PhysxSchema
    from .scene_paths import SCENE
    return read_solver_settings(PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(SCENE)),cfg)
