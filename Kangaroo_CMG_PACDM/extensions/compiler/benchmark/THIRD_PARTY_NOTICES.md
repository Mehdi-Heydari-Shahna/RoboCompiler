# Source and provenance notices

`original/original_v22/` contains the subset of the user-supplied `Kangaroo_RoboIR_full_body_v22` release
that this benchmark executes (the PACDM core `pacdm.py`, the accepted `CutGraph`/`polish` definitions, the
contact reference and the CMG data). Files are retained byte-for-byte: `original/original_v22/MANIFEST_SHA256.json`
is the supplied manifest, and `original/ORIGIN_VERIFICATION.json`, the unit tests and `audit_results.py` check
every file against it. `original/kangaroo_pin/` is the unchanged code of the delivered Kangaroo Pinocchio
validation package.

The Kangaroo model data of the v22 release were derived from the upstream MuJoCo model
`hucebot/mujoco_kangaroo_sim2sim` (https://github.com/hucebot/mujoco_kangaroo_sim2sim), commit
`c020b68f3690930f5228d9f301b59ad9ab405e9a`, as recorded in `original/original_v22/data/provenance.json`, which
also records `author_verified_geometry: false`. The upstream BSD 2-Clause license (Copyright (c) 2025, HuCeBot
Inria/Loria team) is retained byte-for-byte as `licenses/hucebot_mujoco_kangaroo_sim2sim_LICENSE`
(SHA-256 `968f998777ff252e254660419d9756ed96d55db35fba989563196a3f23107cac`, identical to
`upstream/hucebot/LICENSE` in the supplied v22 manifest). Upstream meshes and MJCF files are not needed by this
numerical benchmark and are not redistributed here. This package does not change licenses or claim ownership of
the upstream model, the supplied v22 release or the original PACDM code.

`src/numpy_reference_base.py` is a byte-identical copy of the file of the same name in the Go2 benefit package
(the generic independent NumPy rigid-body tree reference, first used in the Stewart comparison). It imports only
NumPy and does not call PACDM, Pinocchio or MuJoCo. Sharing the physical inertias and geometry between the routes
remains an explicit limitation of the numerical consistency comparisons.

New files under `src/` (compiler, generated evaluator, module scheduler, independent NumPy routes, native
chart-base oracle, rollout comparison, runner) are an identified extension of the user's framework, not
functionality attributed retrospectively to the v22 code.

Software used but not redistributed: NumPy, SciPy, threadpoolctl and Pinocchio 3.8.0 (exact versions in
`requirements-tested.txt`). Documentation and literature cited by the paper text are listed in
`../paper/Kangaroo_References.bib`.

Related framework (not executed as a baseline): URDF+ (Chignoli, Slotine, Wensing and Kim), arXiv:2411.19753,
version 1: https://arxiv.org/abs/2411.19753v1
