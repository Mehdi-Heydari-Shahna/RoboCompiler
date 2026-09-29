# Offline pipeline stand-in for Isaac Sim (tests only)

These modules imitate the small part of the Isaac Sim / Omniverse / pxr API that
`run_isaac.py` calls, backed by the original MuJoCo source model. They exist so
the complete pipeline (bridge server, native runner loop, checkpoints, CPU video,
assessment, ZIPs) can be exercised without Isaac. PhysX-like choices: no rolling
or torsional friction, no MuJoCo joint damping (the runner applies joint losses
through `wrenches.py`), actuator efforts applied as body wrenches. Setting
`FAKE_PIN_SPIN_RATE=0.12` drives the q22 pin spin the way PhysX let it drift.

This is **not** PhysX and **not** evidence of Isaac behaviour. It never affects
a normal run: these folders are only importable when this directory is put on
`PYTHONPATH` explicitly, as `tests/test_offline_pipeline.py` does.

Manual full-pipeline dry run (Linux/macOS shell; on Windows set PYTHONPATH and
call `python run_isaac.py ...` directly):

    PYTHONPATH=tests/fake_isaac python launch.py --isaac-python "$(which python)" --headless --video

`launch.py` passes the environment to its children, so `run_isaac.py` then
imports these stand-ins instead of Isaac Sim.
