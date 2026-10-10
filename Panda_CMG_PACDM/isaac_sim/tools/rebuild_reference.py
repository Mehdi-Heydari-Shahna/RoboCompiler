"""Regenerate the reference with the preserved CMG compiler and PACDM task generator.

Writes a NEW reference; never modifies bundled source evidence. This can run
outside Isaac, without Pinocchio/MuJoCo/pxr. A changed route is not automatically
validated merely because this generator completes.
"""
from pathlib import Path
import argparse,sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from panda.model import build_model
from panda.task import build_reference
from cmg_isaac.rigid import RigidTree


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(ROOT/'results/rebuilt_reference'))
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    model=build_model();original=json.loads((ROOT/'data/panda_cmg.json').read_text())
    if model!=original:
        raise RuntimeError('Compiled CMG differs from the bundled model. Review before reusing native acceptance limits.')
    start=time.perf_counter();print('Running original PACDM route compilation...',flush=True)
    ref=build_reference(model)
    np.savez_compressed(out/'reference.npz',**{k:v for k,v in ref.items() if k!='info'})
    (out/'reference.json').write_text(json.dumps(ref['info'],indent=2)+'\n')
    with np.load(ROOT/'data/reference.npz',allow_pickle=False) as old:
        differences={k:float(np.max(abs(np.asarray(ref[k])-old[k]))) for k in old.files}
    result=dict(status='REFERENCE_REBUILT',wall_seconds=time.perf_counter()-start,differences_from_bundled=differences,
                pacdm_sha256=hashlib.sha256((ROOT/'vendor/pacdm_original.py').read_bytes()).hexdigest(),
                note='PACDM compilation only, not an Isaac physics test.')
    (out/'rebuild_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('To use this route: python run_isaac.py --reference "'+str(out/'reference.npz')+'"')
if __name__=='__main__':main()
