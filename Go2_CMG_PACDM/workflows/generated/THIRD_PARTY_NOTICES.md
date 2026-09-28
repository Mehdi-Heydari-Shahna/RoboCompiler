# Source and model provenance

`original/` retains the source and data subset required from the Go2 Pinocchio
validation workflow. Retained files are identified by
`original/ORIGINAL_HASHES.json`. The nominal `data/reference.npz` supplies input
configurations and the target course; it is not presented as a new locomotion
rollout produced by this compiler benchmark.

The Go2 model derives from the pinned MuJoCo Menagerie snapshot recorded in
`original/data/go2_cmg.json`. The source XML and upstream license are retained
under `original/upstream/unitree_go2/`. Model ownership and licensing remain with
their original rights holders. The original PACDM implementation is retained
byte-for-byte in `original/vendor/pacdm_original.py`.

`src/numpy_reference_base.py` contains our classical NumPy tree reference,
reused from the Stewart benchmark. Go2 point-site geometry is provided by
`src/physics.py`; the Stewart-specific geometry helper is not used. The reference
does not invoke PACDM, Pinocchio or MuJoCo. All formulations use the same declared
physical geometry and inertia records.

The compiler, pruned point evaluator and dependency scheduler in `src/` are
extensions to the retained PACDM implementation. The release distinguishes these
interfaces from the original solver. Historical package/source manifests in
`provenance/` record the earlier archive layouts; only `PACKAGE_MANIFEST.json`
at the workflow root verifies the current release layout.

Primary implementation references:

- SciPy 1.17.0 least-squares API: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
