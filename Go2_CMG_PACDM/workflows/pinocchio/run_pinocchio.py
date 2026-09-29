#!/usr/bin/env python
"""Reproduce the CMG/PACDM Pinocchio validation and recorded video."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, hashlib, json, platform, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import importlib.abc

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_pinocchio'

class NoMuJoCo(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mujoco' or fullname.startswith('mujoco.'):
            raise ImportError('MuJoCo is forbidden in the independent Pinocchio rollout')
        return None

def job(item):
    sys.meta_path.insert(0,NoMuJoCo())
    if 'mujoco' in sys.modules:raise RuntimeError('Rollout process already contains MuJoCo')
    from go2.pin_simulation import run_case
    return run_case(ROOT,name=item[0],**item[1])

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

RUN_MANIFEST=OUT/'SHA256SUMS.json'

def manifest():
    # Run evidence manifest: written to results_pinocchio/, never to a tracked file.
    files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
           and p.name!='SHA256SUMS.json' and p.suffix!='.pyc' and not any(x.startswith('.') for x in p.relative_to(ROOT).parts)}
    write_json(RUN_MANIFEST,files)

def _same_values(a,b,atol=1e-12):
    """Structural equality that ignores floating-point round-off."""
    import numpy as np
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_same_values(a[k],b[k],atol) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(_same_values(x,y,atol) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):return isinstance(b,(int,float)) and not isinstance(b,bool) and bool(np.isclose(a,b,rtol=0,atol=atol))
    return a==b

def _same_arrays(generated,shipped,rtol=1e-6,atol=1e-7):
    import numpy as np
    with np.load(generated,allow_pickle=False) as a,np.load(shipped,allow_pickle=False) as b:
        if sorted(a.files)!=sorted(b.files):return 'array names differ'
        for k in a.files:
            x,y=a[k],b[k]
            if x.shape!=y.shape:return f'{k}: shape {x.shape} != {y.shape}'
            if x.dtype.kind in 'fc' or y.dtype.kind in 'fc':
                if not np.allclose(x,y,rtol=rtol,atol=atol):return f'{k}: max |difference| {float(np.max(np.abs(x-y))):.3g}'
            elif not np.array_equal(x,y):return f'{k}: values differ'
    return ''

def check_shipped(name):
    """The workflow reads data/go2_cmg.json and data/reference.npz, which a run never rewrites.
    They must match the copies this run has just rebuilt into results_pinocchio/."""
    if name=='go2_cmg.json':
        same=_same_values(json.loads((OUT/name).read_text()),json.loads((ROOT/'data'/name).read_text()));why=''
    else:
        why=_same_arrays(OUT/name,ROOT/'data'/name);same=not why
    if not same:
        raise RuntimeError(f'The rebuilt results_pinocchio/{name} differs from the shipped data/{name}'+(f' ({why})' if why else '')+
                           f'. If the model or task change is intended, copy results_pinocchio/{name} into data/ and rerun.')

def audit_existing():
    if not RUN_MANIFEST.is_file():
        raise SystemExit('No run evidence manifest (results_pinocchio/SHA256SUMS.json); run python run_pinocchio.py first.')
    hashes=json.loads(RUN_MANIFEST.read_text())
    bad=[name for name,digest in hashes.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if bad:raise RuntimeError('Integrity mismatch: '+', '.join(bad))
    verdict=json.loads((OUT/'validation.json').read_text())
    if not verdict['passed']:raise RuntimeError('Saved validation did not pass')
    print(f"Integrity PASS ({len(hashes)} files); saved validation PASS ({verdict['passed_count']}/{verdict['check_count']}).")

def independent_audits():
    # MuJoCo is used ONLY as an independent rigid-body comparator here, in a
    # child interpreter that cannot supply data/forces to the rollout workers.
    code="""
import json
from pathlib import Path
from go2.validation import validate_mechanics,validate_contacts
p=Path('results_pinocchio')
a=validate_mechanics(output=p/'mechanics_validation.json')
b=validate_contacts(output=p/'contact_validation.json')
assert a['passed'] and b['passed'], 'Independent model/PACDM audit failed'
print('Mechanics and PACDM support audits PASS')
"""
    run=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
    (OUT/'mechanics_run.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Model audit failed; see mechanics_run.log')
    print(run.stdout.strip(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',action='store_true',help='Also render the validated 26-second MP4.')
    parser.add_argument('--render-only',action='store_true',help='Render previously validated saved states.')
    parser.add_argument('--audit-existing',action='store_true',help='Verify recorded hashes and saved verdict.')
    parser.add_argument('--workers',type=int,default=3,choices=range(1,8))
    args=parser.parse_args()
    if args.audit_existing:audit_existing();return
    OUT.mkdir(exist_ok=True)
    if args.render_only:
        if not json.loads((OUT/'validation.json').read_text())['passed']:raise RuntimeError('Validation must pass before final rendering')
        from go2.pin_render import render
        render(ROOT)
        from go2.pin_report import build_report
        build_report(ROOT);manifest();return
    write_json(OUT/'validation.json',dict(passed=False,status='Running; evidence incomplete'))
    try:
        import numpy, scipy, pinocchio, osqp
        if pinocchio.__version__!='3.8.0':raise RuntimeError('Use Pinocchio 3.8.0 from the pinned environment')
        write_json(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),pinocchio=pinocchio.__version__,numpy=numpy.__version__,scipy=scipy.__version__,osqp=osqp.__version__,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        from go2.model import build_model,save_model
        # Run outputs go to results_pinocchio/ only; tracked files in data/ are checked, never rewritten.
        save_model(build_model(),OUT/'go2_cmg.json');check_shipped('go2_cmg.json')
        independent_audits()
        from go2.task import make_reference
        (ROOT/'results').mkdir(exist_ok=True)
        print('Reassembling reference with unchanged PACDM...',flush=True)
        summary=make_reference(ROOT,path=OUT/'reference.npz');write_json(OUT/'reference_regeneration.json',summary)
        check_shipped('reference.npz')
        from go2.task_validation import validate_reference
        reference=validate_reference(ROOT);write_json(OUT/'task_validation.json',reference)
        if not reference['passed']:raise RuntimeError('PACDM reference audit failed')
        from go2.pin_law_validation import validate_contact_law
        if not validate_contact_law(ROOT)['passed']:raise RuntimeError('Contact-law audit failed')
        from go2.pin_acceptance import CASES,POSITIVE,aggregate
        if args.workers==1:
            for item in CASES.items():job(item)
        else:
            import multiprocessing
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for result in pool.map(job,CASES.items()):print(f"Completed {result['name']}: {result['completed']}",flush=True)
        from go2.pin_clearance import audit_clearance
        for case in POSITIVE:
            print('Collision audit: '+case,flush=True)
            audit_clearance(ROOT,case=case)
        result=aggregate(ROOT)
        from go2.pin_report import build_report
        if not result['passed']:
            build_report(ROOT)
            failed=[k for k,v in result['checks'].items() if not v['passed']]
            raise RuntimeError('Acceptance failed: '+', '.join(failed))
        if args.render:
            from go2.pin_render import render
            render(ROOT)
        build_report(ROOT);manifest()
        print(f"PASS: {result['passed_count']}/{result['check_count']} acceptance gates",flush=True)
    except Exception as error:
        saved=json.loads((OUT/'validation.json').read_text())
        saved.update(passed=False,run_error=f'{type(error).__name__}: {error}')
        write_json(OUT/'validation.json',saved)
        raise

if __name__=='__main__':main()
