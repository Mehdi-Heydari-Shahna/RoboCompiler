#!/usr/bin/env python
"""Run the recorded audit with each NPZ array decompressed once.

The audit source captured before benchmarking is retained in audit_results.py.
This post-run I/O wrapper avoids repeatedly decompressing the same saved arrays;
all numerical checks, thresholds and hash verification remain unchanged.
"""
from pathlib import Path
import argparse
import numpy as np
import audit_results as recorded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results_here')
    parser.add_argument('--replay',choices=['none','sampled','full'],default='full')
    parser.add_argument('--out',default='audit_local')
    args=parser.parse_args()
    real_load=np.load
    class LoadedArchive(dict):
        def __enter__(self): return self
        def __exit__(self,*exc): self.clear()
    def load_once(path,*pos,**kw):
        obj=real_load(path,*pos,**kw)
        if isinstance(obj,np.lib.npyio.NpzFile):
            try: return LoadedArchive({name:obj[name] for name in obj.files})
            finally: obj.close()
        return obj
    recorded.np.load=load_once
    try: return recorded.audit(args.results,args.replay,args.out)
    finally: recorded.np.load=real_load

if __name__=='__main__': raise SystemExit(main())
