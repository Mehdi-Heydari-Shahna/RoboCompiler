# Source and model provenance

`original/` retains the source and data subset required from the Go2 Pinocchio
validation workflow. Retained files are identified by
`original/ORIGINAL_HASHES.json`. The nominal `data/reference.npz` provides input
configurations and the target course; it is not a locomotion rollout produced by
this compiler benchmark.

The Go2 model derives from the pinned MuJoCo Menagerie snapshot recorded in
`original/data/go2_cmg.json`. The source XML and upstream license are retained
under `original/upstream/unitree_go2/`. Model ownership and licensing remain with
their original rights holders. The PACDM implementation (M. Dastranj and
J. Mattila, arXiv:2609.11338) is retained unchanged in
`original/vendor/pacdm_original.py`.

`src/numpy_reference_base.py` contains a classical NumPy tree reference, shared
with the Stewart benchmark. Go2 point-site geometry is provided by
`src/physics.py`; the Stewart-specific geometry helper is not used. The reference
does not invoke PACDM, Pinocchio or MuJoCo. All formulations use the same declared
physical geometry and inertia records.

The compiler, pruned point evaluator and dependency scheduler in `src/` extend the
retained PACDM implementation and are kept separate from the original solver.
`PACKAGE_MANIFEST.json` at the workflow root verifies the current file layout.

Primary implementation references:

- SciPy 1.17.0 least-squares API: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
