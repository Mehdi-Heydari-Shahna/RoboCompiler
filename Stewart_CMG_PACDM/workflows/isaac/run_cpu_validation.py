"""Explicit independent CPU verification. Never a replacement for run_checked.bat."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import traceback
import uuid
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--case',default='all',choices=['all','zero_actuation','nominal','fine','heavy_payload','no_feedforward'])
    args=parser.parse_args()
    import numpy as np
    from cpu_verification.runner import run_cpu_case, write_json, ENGINE
    from isaac_validation.protocol import ZERO_ACTUATION_DURATION, ZERO_ACTUATION_TIMESTEP
    from isaac_validation.scoring import compare_trajectories, LIMITS
    run_id='CPU_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    output=(args.output or ROOT/'results_cpu'/run_id).resolve()
    if output.exists() and any(output.iterdir()): parser.error('Output must be a new or empty directory')
    output.mkdir(parents=True,exist_ok=True)
    manifest=dict(run_id=run_id,engine=ENGINE,native_isaac_execution=False,command=sys.argv,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(ROOT.rglob('*.py')) if 'evidence' not in p.parts},
                  input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [ROOT/'data/stewart.cmg.json',ROOT/'baseline/reference.npz',ROOT/'vendor/pacdm_original.py']})
    write_json(output/'run_manifest.json',manifest)
    cases=[('zero_actuation',ZERO_ACTUATION_TIMESTEP,ZERO_ACTUATION_DURATION),('nominal',.002,22.),
           ('fine',.001,22.),('heavy_payload',.002,22.),('no_feedforward',.002,22.)]
    if args.case!='all': cases=[x for x in cases if x[0]==args.case]
    results={}; error=None
    try:
        for name,dt,duration in cases:
            results[name]=run_cpu_case(ROOT,output,name,run_id,dt,duration)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        write_json(output/'failure.json',dict(error=error,traceback=traceback.format_exc(),run_id=run_id))
        traceback.print_exc()
    def load(name,folder=output):
        with np.load(folder/f'{name}.npz',allow_pickle=False) as z: return {k:z[k] for k in z.files}
    comparisons={}
    if args.case=='all' and error is None:
        nominal=load('nominal')
        for label,other,plimit,alimit in [
            ('cpu_timestep_refinement',load('fine'),LIMITS['refinement_position_m'],LIMITS['refinement_orientation_deg']),
            ('cpu_vs_archived_mujoco',load('nominal',ROOT/'baseline'),LIMITS['cross_engine_position_m'],LIMITS['cross_engine_orientation_deg']),
            ('cpu_vs_archived_pacdm',load('pacdm',ROOT/'baseline'),LIMITS['cross_engine_position_m'],LIMITS['cross_engine_orientation_deg'])]:
            comparisons[label]=compare_trajectories(nominal,other,plimit,alimit,label)
    ratio=None
    if 'nominal' in results and 'no_feedforward' in results:
        ratio=results['nominal']['rms_position_error_m']/results['no_feedforward']['rms_position_error_m']
    passed=(error is None and len(results)==len(cases)
            and all(x['passed'] and x.get('sampled_model_audit_passed',True) for x in results.values())
            and all(x['passed'] for x in comparisons.values())
            and (args.case!='all' or (ratio is not None and ratio<1)))
    report=dict(schema='stewart.cpu.verification/1.0',run_id=run_id,engine=ENGINE,
                status='CPU_VERIFICATION_PASSED' if passed else 'CPU_VERIFICATION_FAILED',
                cpu_verification_passed=passed,suite_complete=args.case=='all' and len(results)==5,
                native_isaac_execution=False,isaac_status='UNVERIFIED',
                native_isaac_validation_passed=None,cases=results,comparisons=comparisons,
                feedforward_rms_ratio=ratio,thresholds=LIMITS,error=error,
                interpretation='Fresh force-driven CPU trajectories; not archived replay, not an API double, and not native Isaac Sim execution.')
    write_json(output/'validation_cpu.json',report)
    print(f'\n{report["status"]}. Native Isaac: UNVERIFIED. Evidence: {output}',flush=True)
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
