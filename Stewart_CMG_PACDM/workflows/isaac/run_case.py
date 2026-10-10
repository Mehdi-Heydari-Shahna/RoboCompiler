"""Internal worker: launch one isolated Isaac Sim process per measured case."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=['nominal','fine','heavy_payload','no_feedforward','zero_actuation'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--run-id',required=True)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--duration',type=float,default=22.)
    p.add_argument('--video',action='store_true')
    p.add_argument('--gui',action='store_true')
    args=p.parse_args();app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':not args.gui,'width':1280,'height':720,
                           'renderer':'RayTracedLighting','anti_aliasing':1})
        from isaac_validation.runtime import run_case
        run_case(app,Path(__file__).resolve().parent,args.output,args.case,args.run_id,
                 args.dt,args.duration,args.video)
        return 0
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/f'{args.case}.failure.json').write_text(json.dumps({
            'passed':False,'completed':False,'run_id':args.run_id,'case':args.case,
            'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()},indent=2),encoding='utf-8')
        traceback.print_exc()
        return 1
    finally:
        if app is not None:app.close()

if __name__=='__main__':sys.exit(main())
