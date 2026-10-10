"""Run every required full-body gate. Any failure gives exit status 1."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,platform,time,sys,traceback,hashlib
import numpy as np,scipy,mujoco,pinocchio
from validate_whole_body import run as instantaneous
from whole_body_motion import run as motion
from floating_motion import run as floating
ROOT=Path(__file__).resolve().parent

def run(visuals=False):
    for folder in ['data','models','results','videos']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    t=time.time();results=[]
    try:
        for label,fn in [('instantaneous',instantaneous),('motor_motion',motion),('floating_motion',floating)]:
            print('\nRunning',label,flush=True);r=fn();results.append((label,r))
            if not all(x['passed'] for x in r['gates']):raise RuntimeError(label+' failed required gates')
        if visuals:
            from make_figures import run as figures
            from render_whole_body import render
            figures();render()
        gates=[dict(suite=label,**g) for label,r in results for g in r['gates']]
        status='PASS_RECONSTRUCTED_MODEL';error=None
    except Exception as exc:
        traceback.print_exc();status='FAIL';error=str(exc);gates=[dict(suite=label,**g) for label,r in results for g in r['gates']]
    report=dict(status=status,scope='Full prototype reconstruction; source joint topology changes and angular frames are explicit assumptions, not author-verified hardware geometry.',gates_passed=sum(g['passed'] for g in gates),gates_total=len(gates),gates=gates,error=error,elapsed_seconds=time.time()-t,environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,mujoco=mujoco.__version__,pinocchio=pinocchio.__version__),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))})
    (ROOT/'results/full_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(status,f"{report['gates_passed']}/{report['gates_total']} gates",flush=True);return status=='PASS_RECONSTRUCTED_MODEL'
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--visuals',action='store_true',help='Also regenerate figure and MP4; needs matplotlib, Pillow, ffmpeg, graphics context.')
    sys.exit(0 if run(parser.parse_args().visuals) else 1)
