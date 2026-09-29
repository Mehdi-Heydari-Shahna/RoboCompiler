# RoboCompiler

**Graph-Native Compilation of Closed-Chain Robots for Consistent Modeling, Control, and Simulation**

Mehdi Heydari Shahna · Joongheon Kim · Jouni Mattila

RoboCompiler compiles a **canonical mechanism graph (CMG)**—bodies, joints, attachment frames, inertias, and actuator ports—into consistent interfaces for robot assembly, motion, actuation, and rigid-body dynamics. PACDM-based closure and rank-checked coordinate reduction connect the physical mechanism description to controller and simulator models.

This repository provides research implementations and simulation examples for five robots, with compiler studies and workflows involving **MuJoCo, Pinocchio, and Isaac Sim/PhysX**.

## Main capabilities

- Graph-derived closure paths, analytical residual Jacobians, and dependency modules.
- Feasible assembly and tangent maps from independent coordinates to robot, task, and actuator motion.
- Virtual-work-consistent actuator effort maps and curvature-aware reduced dynamics.
- Generated local evaluation and selective updates, with independent mechanics checks and simulation studies.

## Robot examples

| Robot | Example | Mechanism graph | Workflow |
|---|---|---|---|
| Komatsu excavator | Closed linkages, cylinder actuation, excavation and material transfer | [CMG](figures/excavator_cmg.png) | [Diagram](figures/excavator_workflow.png) |
| Unitree Go2 | Floating-base tree with foot tasks and prescribed support modes | [CMG](figures/go2_cmg.png) | [Diagram](figures/go2_workflow.png) |
| Franka Panda | Serial arm with coupled fingers; grasping and socket placement | [CMG](figures/panda_cmg.png) | [Diagram](figures/panda_workflow.png) |
| Kangaroo | Closed-chain legs; landing, crouching, and push response | [CMG](figures/kangaroo_cmg.png) | [Diagram](figures/kangaroo_workflow.png) |
| Six-UPS Stewart platform | Parallel mechanism; six-axis tracking and inspection | [CMG](figures/stewart_cmg.png) | [Diagram](figures/stewart_workflow.png) |

Go2 and Panda demonstrate task and actuation constraints on tree mechanisms; the excavator, Kangaroo, and Stewart platform contain physical closed chains.

## Getting started

The packages retain separate launchers and dependency files. Use **Python 3.12 on Linux** for the pinned reference environments, and create a separate environment for each package. The main MuJoCo examples pin **MuJoCo 3.3.7** and **Pinocchio 3.8.0**; additional studies may use different requirements.

For example, from the repository root:

```bash
cd Panda_CMG_PACDM
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux.txt
python run_panda.py
```

For another example, enter the directory below, create its environment, install its listed requirements, and run its entry point.

| Working directory | Requirements file | Entry point |
|---|---|---|
| `Panda_CMG_PACDM/` | `requirements-linux.txt` | `python run_panda.py` |
| `Go2_CMG_PACDM/` | `requirements-linux.txt` | `python reproduce.py mujoco` |
| `Kangaroo_CMG_PACDM/` | `requirements_validated.txt` | `python reproduce.py --simulation-only` |
| `Stewart_CMG_PACDM/` | `requirements-linux.txt` | `python run_stewart.py` |
| `Excavator_CMG_PACDM/compiler/` | `requirements-lock-linux.txt` | `python run_excavator.py --profile smoke --out results_smoke` |

The excavator entry above runs the compiler smoke study. Use a new output directory when repeating it. Its MuJoCo soil example is launched separately from `Excavator_CMG_PACDM/` using `run_soil.py` and the `requirements.txt` in that folder.

The code packages include model inputs and assets; the requirements files install the dependencies for their listed workflows. Isaac Sim requires its own runtime and workflow-specific setup and is not installed by the commands above.

The compiler studies and the Pinocchio and Isaac Sim/PhysX workflows of each robot are in the folders below (entry point in parentheses).

| Robot | Compiler study | Pinocchio | Isaac Sim/PhysX |
|---|---|---|---|
| Excavator | `Excavator_CMG_PACDM/compiler/` (`run_excavator.py`) | `Excavator_CMG_PACDM/pinocchio/` (`run_validation.py`) | `Excavator_CMG_PACDM/isaac_sim/` (`launch.py`, see its README) |
| Go2 | `Go2_CMG_PACDM/workflows/generated/` (`run_go2.py`) | `Go2_CMG_PACDM/workflows/pinocchio/` (`run_pinocchio.py`) | `Go2_CMG_PACDM/workflows/isaac/` (`launch.py`) |
| Panda | `Panda_CMG_PACDM/compiler_studies/benchmark/` (`run_franka.py`) | `Panda_CMG_PACDM/pinocchio_validation/` (`run_validation.py`) | `Panda_CMG_PACDM/isaac_sim/` (`run_isaac.py`) |
| Kangaroo | `Kangaroo_CMG_PACDM/extensions/compiler/benchmark/` (`run_kangaroo.py`) | `Kangaroo_CMG_PACDM/extensions/pinocchio/` (`run_pinocchio.py`) | `Kangaroo_CMG_PACDM/extensions/isaacsim/` (`run_isaac.py`) |
| Stewart platform | `Stewart_CMG_PACDM/workflows/compiler/` (`run_stewart.py`) | `Stewart_CMG_PACDM/workflows/pinocchio/` (`run_pinocchio.py`) | `Stewart_CMG_PACDM/workflows/isaac/` (`run_checked.py`) |

## Demonstration videos

| Robot | MuJoCo | Pinocchio | Isaac Sim |
|---|---|---|---|
| Excavator | [Task 1](videos/excavator_mujoco_task1.mp4), [Task 2](videos/excavator_mujoco_task2.mp4) | [Video](videos/excavator_pinocchio.mp4) | [Video](videos/excavator_isaac_sim.mp4) |
| Go2 | [Video](videos/go2_mujoco.mp4) | [Video](videos/go2_pinocchio.mp4) | [Video](videos/go2_isaac_sim.mp4) |
| Panda | [Video](videos/panda_mujoco.mp4) | [Video](videos/panda_pinocchio.mp4) | [Video](videos/panda_isaac_sim.mp4) |
| Kangaroo | [Video](videos/kangaroo_mujoco.mp4) | [Video](videos/kangaroo_pinocchio.mp4) | — |
| Stewart platform | [Video](videos/stewart_mujoco.mp4) | [Video](videos/stewart_pinocchio.mp4) | [Video](videos/stewart_isaac_sim.mp4) |

Videos illustrate the associated studies; backend models and contact treatments differ. The compact packages provide code to generate results and do not include every saved report or trajectory. Replay and existing-result audits require their corresponding outputs first.

## Scope

The compilers operate on physical mechanism records, with locally valid assembly branches and rank conditions. MuJoCo and PhysX provide native contact simulation; Pinocchio workflows use their own stated contact or external-load assumptions. The excavator soil example uses an uncalibrated granular model.

## Paper and citation

Please cite the accompanying manuscript when using this work:

Mehdi Heydari Shahna, Joongheon Kim, and Jouni Mattila. **RoboCompiler: Graph-Native Compilation of Closed-Chain Robots for Consistent Modeling, Control, and Simulation.** 2026. [Manuscript](docs/robocompiler.pdf).

The arXiv link and identifier will be added after publication.

Third-party model licenses and attribution notices are retained within the corresponding packages.
