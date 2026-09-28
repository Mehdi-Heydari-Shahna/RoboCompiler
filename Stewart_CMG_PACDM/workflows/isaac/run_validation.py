"""Run the complete Stewart Isaac Sim validation, including live nominal video."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parent


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='New empty output directory; default results_isaac/<UTC timestamp>.')
    p.add_argument('--gui',action='store_true',help='Open an Isaac window while the suite runs.')
    p.add_argument('--no-video',action='store_true',help='Physics checks only; skip nominal MP4.')
    p.add_argument('--smoke',action='store_true',help='Two-second nominal startup diagnostic; cannot validate the full mission.')
    p.add_argument('--preflight-only',action='store_true',help='CPU PACDM/reference checks only; does not launch Isaac.')
    p.add_argument('--export-usd',action='store_true',help='Compile and audit USD only (requires pxr); does not launch Isaac.')
    args=p.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    run_id=f'{stamp}_{uuid.uuid4().hex[:8]}'
    output=(args.output or ROOT/'results_isaac'/run_id).resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Output directory must be new or empty so old results cannot enter acceptance.')
    output.mkdir(parents=True,exist_ok=True)
    from isaac_validation.preflight import run_preflight
    from isaac_validation.scoring import aggregate,build_report
    preflight=run_preflight(ROOT)
    from isaac_validation.model_checks import check_reference
    if preflight['passed']:
        try:
            extra=check_reference(ROOT)
        except Exception as exc:
            extra=dict(passed=False,error=f'{type(exc).__name__}: {exc}')
        preflight['fresh_body_dynamics_check']=extra
        preflight['passed']=bool(preflight['passed'] and extra['passed'])
    preflight.update(run_id=run_id,video_requested=not args.no_video)
    write_json(output/'preflight.json',preflight)
    write_json(output/'validation.json',dict(passed=False,physics_passed=False,
        physics_status='UNVERIFIED',status='RUNNING',run_id=run_id))
    manifest={'run_id':run_id,'started_utc':datetime.now(timezone.utc).isoformat(),
        'command':sys.argv,'executable':sys.executable,'smoke':args.smoke,
        'input_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                         ['data/stewart.cmg.json','baseline/reference.npz','vendor/pacdm_original.py']},
        'code_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sorted(ROOT.rglob('*.py')) if '__pycache__' not in f.parts}}
    write_json(output/'run_manifest.json',manifest)
    if not preflight['passed']:
        report=aggregate(output,ROOT/'baseline',preflight)
        report['status']='PREFLIGHT_FAILED'
        write_json(output/'validation.json',report);build_report(output,report)
        print(f'PREFLIGHT FAILED. See {output / "report.html"}')
        return 1
    if args.export_usd:
        import numpy as np
        from isaac_validation.usd_scene import build_scene
        cmg=json.loads((ROOT/'data/stewart.cmg.json').read_text())
        with np.load(ROOT/'baseline/reference.npz',allow_pickle=False) as z:q=z['q'][0]
        scene=build_scene(cmg,q,output/'Stewart.usda')
        write_json(output/'usd_audit.json',scene['audit'])
    if args.preflight_only or args.export_usd:
        report=aggregate(output,ROOT/'baseline',preflight)
        report['status']='PREFLIGHT_ONLY';report['physics_status']='UNVERIFIED'
        write_json(output/'validation.json',report);build_report(output,report)
        print(f'CPU checks PASS. Isaac physics UNVERIFIED. Results: {output}')
        return 0
    from isaac_validation.protocol import ZERO_ACTUATION_DURATION, ZERO_ACTUATION_TIMESTEP
    cases=[('zero_actuation',ZERO_ACTUATION_TIMESTEP,ZERO_ACTUATION_DURATION),('nominal',.002,22.),('fine',.001,22.),
           ('heavy_payload',.002,22.),('no_feedforward',.002,22.)]
    if args.smoke:cases=[('nominal',.002,2.)]
    failures=[]
    for name,dt,duration in cases:
        command=[sys.executable,str(ROOT/'run_case.py'),'--case',name,'--output',str(output),
                 '--run-id',run_id,'--dt',str(dt),'--duration',str(duration)]
        if args.gui:command+=['--gui']
        if name=='nominal' and not args.no_video:command+=['--video']
        print(f'Launching {name} ({duration:g}s, dt={dt:g}s)...',flush=True)
        with (output/f'{name}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        from isaac_validation.worker_check import check_worker
        worker=check_worker(output,name,run_id,code,duration,dt)
        write_json(output/f'{name}.worker_check.json',worker)
        if not worker['passed']:
            failures.append(worker)
            # All cases share one backend: stop after startup/physics failure.
            break
    video_path=output/'video_metadata.json'
    video=json.loads(video_path.read_text()) if video_path.exists() else None
    report=aggregate(output,ROOT/'baseline',preflight,video)
    report['process_failures']=failures
    if failures:
        report.update(passed=False,physics_passed=False,simulation_validation_passed=False,status='EXECUTION_FAILED',physics_status='EXECUTION_FAILED')
    if args.smoke:
        report['passed']=False;report['physics_passed']=False
        from isaac_validation.worker_check import smoke_checks
        smoke=smoke_checks(output,run_id,not args.no_video) if not failures else dict(passed=False)
        report['smoke_diagnostic']=smoke
        report['status']='SMOKE_PASSED' if smoke['passed'] and not failures else 'SMOKE_FAILED'
        report['physics_status']='UNVERIFIED'
        report['simulation_validation_passed']=False
    write_json(output/'validation.json',report);build_report(output,report)
    print(f"\nPhysics: {report['physics_status']}. Overall: {report['status']}.\nReport: {output/'report.html'}",flush=True)
    if args.smoke:
        print('Smoke run is a startup diagnostic, not mission validation.')
        return 0 if report['status']=='SMOKE_PASSED' else 1
    return 0 if report['passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted. The suite is not validated.',file=sys.stderr);sys.exit(130)
