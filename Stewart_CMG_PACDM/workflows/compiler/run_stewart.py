#!/usr/bin/env python3
"""One-command reproduction; run from any working directory."""
from __future__ import annotations
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import argparse, hashlib, json, platform, sys, time, traceback
from pathlib import Path
from src.bootstrap import ROOT,PRIOR
from src.runner import *
from src.compiler import compile_graph,from_original_cmg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','full'],default='full')
    parser.add_argument('--native',choices=['auto','off','required'],default='auto')
    parser.add_argument('--stages',default='all',help='all or comma-separated: pipeline,warm,derivatives,dynamics,verification,native')
    parser.add_argument('--out',default='results_local')
    args=parser.parse_args()
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output folder is not empty. Choose a fresh --out folder to avoid mixing runs.')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'inputs/protocol.json').read_text());config=protocol[args.profile]
    nominal=json.loads((PRIOR/'original/data/stewart.cmg.json').read_text())
    stages=['pipeline','warm','derivatives','dynamics','verification'] if args.stages=='all' else args.stages.split(',')
    allowed={'pipeline','warm','derivatives','dynamics','verification','native'}
    if set(stages)-allowed:parser.error('Unknown stage(s): '+str(set(stages)-allowed))
    if args.native=='required' or 'native' in stages:
        try:import pinocchio
        except Exception as exc:
            save_json(out/'native_status.json',dict(status='not_run',reason=str(exc)))
            parser.error('Pinocchio unavailable. Install conda-forge pinocchio, or use --native off. '+str(exc))
    save_json(out/'protocol.json',protocol);save_json(out/'environment.json',environment())
    summary=dict(status='running',profile=args.profile,stages=stages,protocol_sha256=canonical_hash(protocol),
        native_request=args.native,scope=protocol['scope'],new_extension='src/compiler.py; original PACDM file unchanged',
        external_baseline_status={'SciPy_TRF':'implemented','NumPy_KKT':'implemented','URDFplus_generalized_rbda':'not implemented or measured'})
    save_json(out/'summary.json',summary)
    start=time.perf_counter();failure=None
    try:
        for stage in stages:
            print('\nSTAGE',stage,flush=True);t=time.perf_counter()
            if stage=='pipeline':res=pipeline_stage(nominal,config,protocol,out)
            elif stage=='warm':res=warm_stage(nominal,config,protocol,out)
            elif stage=='derivatives':res=derivative_stage(nominal,config,protocol,out)
            elif stage in ('dynamics','native'):res=dynamics_stage(nominal,config,protocol,out,'required' if stage=='native' else args.native)
            else:res=fault_stage(nominal,out)
            res['elapsed_s']=time.perf_counter()-t;summary[stage]=res
            print(stage,':',json.dumps(clean({k:v for k,v in res.items() if k not in ('summary','moving_summary','repeat_summary')})),flush=True)
            save_json(out/'summary.json',summary)
        verified=[k for k in stages if k in ('pipeline','dynamics','native','verification')]
        summary['verification_stages']=verified
        if verified:
            # A stage passes only if it attempted at least one case and every attempt passed.
            def stage_ok(k):
                attempts=summary[k].get('attempts',summary[k].get('tests',0))
                return attempts>0 and summary[k].get('passed',0)==attempts
            verification_ok=all(stage_ok(k) for k in verified)
            summary['status']='completed' if verification_ok else 'completed_with_verification_failures'
        else:
            # No verification stage was selected, so no verification result is claimed.
            verification_ok=None
            summary['status']='completed_without_verification'
        summary['all_verification_passed']=verification_ok
    except Exception as exc:
        summary['status']='error';failure=traceback.format_exc();summary['error']=failure
        (out/'exception.txt').write_text(failure);print(failure,flush=True)
    summary['elapsed_s']=time.perf_counter()-start;save_json(out/'summary.json',summary)
    try:
        from src.reporting import build_report
        build_report(out,summary)
    except ImportError as exc:print('Report module not available:',exc,flush=True)
    except Exception:
        (out/'report_error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    save_json(out/'source_hashes.json',files)
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='RUN_MANIFEST.json'}
    save_json(out/'RUN_MANIFEST.json',dict(files=hashes,source_sha256=files,protocol_sha256=canonical_hash(protocol)))
    print('RESULTS:',out,flush=True)
    # Exit 1 on an error or a failed verification; a run without verification stages exits 0 but claims no pass.
    return 1 if failure or (summary.get('verification_stages') and not summary.get('all_verification_passed')) else 0

if __name__=='__main__':raise SystemExit(main())
