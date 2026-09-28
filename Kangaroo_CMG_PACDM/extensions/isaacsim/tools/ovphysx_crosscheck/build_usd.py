"""Author the Kangaroo USD with the PACKAGE's own usd_builder (usd-core + schema shim).

Development only: reproduces what the Isaac worker authors before world.reset(),
so the modified builder itself is exercised, then runs the package's USD audit.
"""
from __future__ import annotations
import json, sys
from pathlib import Path


def build(src, cfg, out_path, isaac_scene_defaults=True):
    sys.path.insert(0, str(Path(src).resolve()))
    import physxschema_shim
    physxschema_shim.install()
    from pxr import Usd, Sdf
    from kangaroo_isaac.model import Model
    from kangaroo_isaac.control import Reference
    from kangaroo_isaac.usd_builder import build as package_build, audit_stage
    from kangaroo_isaac.scene_paths import SCENE
    model = Model(); ref = Reference()
    stage = Usd.Stage.CreateInMemory()
    P0 = package_build(stage, model, ref, cfg, visuals=False)
    if isaac_scene_defaults:
        # Isaac's World/PhysicsContext pre-authors this on /physicsScene (seen in
        # the user's exported scene). It has no effect without per-body CCD flags.
        stage.GetPrimAtPath(SCENE).CreateAttribute('physxScene:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
    audit = audit_stage(stage, model, cfg)
    stage.GetRootLayer().Export(str(out_path))
    return P0, audit


if __name__ == '__main__':
    import argparse
    from dataclasses import replace
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--case', default='nominal')
    ap.add_argument('--out', required=True); ap.add_argument('--config-json', default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.src).resolve()))
    from kangaroo_isaac.control import case_config
    cfg = case_config(a.case)
    if a.config_json:
        cfg = replace(cfg, **json.loads(a.config_json)).validate()
    _, audit = build(a.src, cfg, a.out)
    print(json.dumps({k: audit[k] for k in audit if k != 'solver_settings'}, indent=1, default=str)[:3000])
