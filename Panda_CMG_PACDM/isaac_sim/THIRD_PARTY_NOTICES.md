# Source and third-party notices

The CMG/PACDM implementation, Panda compiler, task generator, data and original benchmark evidence are retained from the research source archives identified in `PROVENANCE.json`. Existing ownership and terms remain unchanged. The files listed as preserved in that record retain their original SHA-256 values.

Panda geometry and model assets are retained in `upstream/franka_emika_panda/`, including their upstream LICENSE and provenance/readme files. The retained asset license is Apache License 2.0; consult the original notice for its terms. Bundled mesh arrays derive from those assets and are mapped to source filenames and hashes in `data/geometry.json`.

Isaac Sim, PhysX, USD, NumPy, SciPy and trimesh runtimes are not redistributed. Optional development tools use separately installed SAPIEN, MuJoCo and usd-core packages under their respective terms. Trimesh is required only to regenerate mesh arrays.

The adapter is research companion software. Redistribution of third-party model assets remains subject to the retained upstream terms.
