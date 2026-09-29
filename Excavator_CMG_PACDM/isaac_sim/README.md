# Excavator: Isaac Sim / PhysX workflow

Native PhysX execution of the excavator `soil_final` mission (excavation at an exposed face and
material transfer to a receiving bay) with the unchanged PACDM/hydraulic controller. PhysX
integrates the floating vehicle, the closed arm linkage, the tracks and the soil grains. The
controller runs in a separate Python process and receives the measured native state at every
1 ms physics step over a localhost socket; MuJoCo is used there only for the compiled model
layout and kinematics and is never stepped.

## Contents

| Path | Role |
| --- | --- |
| `launch.py` | One-command coordinator, run with the controller Python |
| `run_isaac.py` | Native runner, executed by the Isaac Sim Python |
| `bridge.py`, `bridge_server.py` | Controller bridge: PACDM controller, hydraulic drives and mission logic |
| `rolling_resistance.py`, `wrenches.py` | Grain rolling resistance and joint/actuator wrenches applied in PhysX |
| `measurement_store.py`, `result_status.py` | Checkpointed measurements and status recovery |
| `assess_results.py` | Task, residual, closure and reference comparison |
| `replay_video.py`, `software_replay.py`, `render_replay_worker.py`, `encode_video.py`, `video_capture.py` | Post-run video from the saved native states |
| `source/` | Controller source: PACDM core, hydraulic drives, soil mission and MuJoCo source scene |
| `generated/soil_final/` | USD scene (`scene.usda`) and its export manifest |
| `reference/` | MuJoCo trajectory and report of the same mission, used for comparison |
| `tests/` | Offline tests with a MuJoCo-backed stand-in for the Isaac API |

`source/RoboCompiler_Excavator_Soil_v01/` holds the controller source (the same excavator package as
`../pinocchio/original/`). The manifest records the SHA-256 of its `outputs/soil_final/scene.xml`, and
the runner refuses to start if the scene differs.

## Requirements

- Isaac Sim 6.1 and its Python environment (reference run: Isaac Sim 6.1.0.0, Kit 110.3, `omni.physx` 110.3.2).
- A separate controller environment with Python 3.12:

  ```bash
  python -m pip install -r ../requirements.txt
  ```

## Run

From this folder, in the controller environment:

```bash
python launch.py --isaac-python <Isaac Sim python executable> --headless --full-mission --video --full-results-zip
```

- `--full-mission` runs all 11 guarded phases (at most 65 simulated seconds). Without it, a 2 s smoke run is executed.
- `--video` renders the saved native states after physics has finished (CPU renderer by default;
  `--video-renderer isaac` uses Isaac RTX workers instead).
- `--full-results-zip` additionally writes a complete results archive.
- `--grain-rolling-friction 0` disables the grain rolling resistance.

Each run writes `results/soil_final_<UTC timestamp>/` and a ZIP archive next to it. Physics takes
roughly 70 s of wall time per simulated second, i.e. about 50 minutes for the full mission.

Main outputs of a run:

| File | Content |
| --- | --- |
| `validation.json` | Physics, video and overall output gates |
| `comparison.json` | Task metrics, controller residuals, closure gaps and comparison with the MuJoCo reference |
| `status.json` | Native runner status, versions and model adjustments |
| `states.npz`, `applied_wrenches.npz`, `native_contact_forces.npz` | Native body states, applied efforts and contact forces |
| `bridge_diagnostics.jsonl` | Controller, hydraulic and mission observations per control interval |
| `isaac_native.mp4` | Replay of the saved native states |

`native_task_validation_passed` in `validation.json` requires all 11 phases to complete, at least
1 kg lifted and deposited, no controller fallback or state projection, and the declared residual and
closure bounds.

Reference run (Isaac Sim 6.1.0.0, Windows 11): all 11 phases completed in 42.001 s with 92.153 kg
delivered to the receiving bay, 4202 PACDM controller evaluations without fallback, and peak arm and
track closure gaps of 11.46 µm and 0.423 mm.

## Modelling notes

- **Cross-pin spin (q22).** The uncommanded spin of the axisymmetric cross pin `body_59` changes no other
  coordinate and does not enter the controller's mass matrix, bias forces or PACDM maps. The controller
  receives it at its nominal angle; this symmetry is verified numerically at start-up, and the measured
  drift is reported.
- **Grain rolling resistance.** PhysX has no rolling friction. The MuJoCo condim-6 rolling (0.008 m) and
  torsional (0.003 m) resistance of the source model is applied to the 105 soil grains as explicit
  torques computed from each grain's PhysX net contact force.

## Other commands

Run these from this folder in the controller environment, with the result folder of a run:

```bash
python replay_video.py --result <result folder>                                  # re-render the video
python assess_results.py --result <result folder> --output <result folder>/comparison_rerun.json
python measurement_store.py --result <result folder>                             # assemble checkpoints after an interruption
```

## Tests

The tests do not need Isaac Sim; `tests/fake_isaac` imitates the used part of the Isaac/PhysX API with
MuJoCo, so they check the Python pipeline, not PhysX behaviour.

```bash
python -m pip install pytest
python -m pytest tests
```
