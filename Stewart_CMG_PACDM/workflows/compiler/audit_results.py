#!/usr/bin/env python3
"""Recompute saved result aggregates and selected mechanical witnesses."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from src.bootstrap import ROOT,PRIOR
from src.compiler import compile_graph,from_original_cmg,canonical_hash
from src.experiments import make_state,physical_acceptance,reduced_acceleration,kkt_acceleration
from src.runner import summarize
from numpy_backend import NumpyTree
from run_comparison import clean


def loadcsv(path):
    out=[]
    with path.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            r={}
            for k,v in row.items():
                if v in ('True','False'):r[k]=v=='True'
                elif v=='':r[k]=None
                else:
                    try:r[k]=float(v)
                    except ValueError:r[k]=v
            out.append(r)
    return out


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True);parser.add_argument('--out',type=Path,default=Path('audit_local.json'))
    args=parser.parse_args();p=args.results;checks=[]
    def check(name,ok,detail=None):checks.append(dict(check=name,passed=bool(ok),detail=detail))
    s=json.loads((p/'summary.json').read_text());check('run completed',s['status']=='completed')
    manifest=json.loads((p/'RUN_MANIFEST.json').read_text());bad=[]
    for rel,digest in manifest['files'].items():
        f=p/rel
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:bad.append(rel)
    check('all recorded result hashes',not bad,dict(files=len(manifest['files']),mismatches=bad))
    check('protocol hash',canonical_hash(json.loads((p/'protocol.json').read_text()))==s['protocol_sha256'])
    for section,filename in [('pipeline','pipeline_raw.csv'),('warm','warm_raw.csv'),('derivatives','derivative_raw.csv'),('dynamics','dynamics_raw.csv')]:
        if section not in s:continue
        raw=loadcsv(p/filename);ss=s[section]
        check(section+' raw row count',len(raw)==ss['attempts'],len(raw))
        check(section+' success count',sum(bool(r['success']) for r in raw)==ss['passed'],sum(bool(r['success']) for r in raw))
    for section,filename,fields in [('warm','warm_raw.csv',['total_ms','mapping_ms','oracle_evaluations','gap_m','pose_error']),('derivatives','derivative_raw.csv',['corrector_ms','oracle_evaluations','gap_m','pose_error'])]:
        if section not in s:continue
        raw=loadcsv(p/filename);recomputed={r['method']:r for r in summarize(raw,['method'],fields)}
        for r in s[section]['summary']:
            for field in fields:
                for suffix in ('_median','_p95','_max'):
                    key=field+suffix;a=r[key];b=recomputed[r['method']][key]
                    ok=(a is None and b is None) or (a is not None and b is not None and np.isclose(a,b,atol=1e-14,rtol=1e-12))
                    check(f'{section} {r["method"]} {key}',ok)
        if section=='warm':
            moving=[r for r in raw if r['moving']];computed={r['method']:r for r in summarize(moving,['method'],['total_ms'])}
            for r in s[section]['moving_summary']:
                check('moving-only '+r['method'],r['attempts']==computed[r['method']]['attempts'] and np.isclose(r['total_ms_median'],computed[r['method']]['total_ms_median']))
    original=json.loads((PRIOR/'original/data/stewart.cmg.json').read_text());source=from_original_cmg(original);c=compile_graph(source);ref=NumpyTree(original);back=NumpyTree(c.cmg)
    q0,_,_=make_state(c,original,ref,0.);g=c.graph(q0)
    if (p/'warm_states.npz').exists():
        saved=np.load(p/'warm_states.npz');allstates={k:saved[k] for k in saved.files if k.startswith('PACDM_') or k.startswith('TRF_')}
        check('saved warm state count',sum(len(a) for a in allstates.values())==s['warm']['attempts'])
        selected=np.unique(np.linspace(0,len(saved['time'])-1,9,dtype=int))
        for key,xs in allstates.items():
            if key.startswith('PACDM_original'):continue
            for k in selected:
                q,_,W=make_state(c,original,ref,float(saved['time'][k]));res=physical_acceptance(g,back,source,xs[k],saved['commanded_lengths'][k],W)
                check(f'saved physical witness {key} sample {k}',res['success'] and res['pose_error']<1e-7,{'gap_m':res['gap_m'],'pose_error':res['pose_error']})
    if (p/'derivative_states.npz').exists():
        states=np.load(p/'derivative_states.npz')['states'];raw=loadcsv(p/'derivative_raw.csv')
        check('saved derivative state count',len(states)==len(raw))
        for k in np.unique(np.linspace(0,len(raw)-1,12,dtype=int)):
            row=raw[k];q,_,W=make_state(c,original,ref,row['time_s']);res=physical_acceptance(g,back,source,states[k],q[g.active],W)
            check(f'saved derivative witness {k}',res['success']==row['success'] and res['pose_error']<1e-7)
    if (p/'dynamics_states.json').exists():
        states=json.loads((p/'dynamics_states.json').read_text());check('saved dynamics state count',len(states)==s.get('dynamics',s.get('native'))['passed'])
        for i in np.unique(np.linspace(0,len(states)-1,min(8,len(states)),dtype=int)):
            z={k:np.array(v) for k,v in states[i].items()};a,d=reduced_acceleration(g,back,g.lift(z['q']),z['active_velocity'],z['forces'],z['wrench'])
            ak,_=kkt_acceleration(back,z['q'],d['v'],z['forces'],z['wrench'])
            rel=float(np.max(abs(a-ak))/max(1.,np.max(abs(ak))))
            check(f'recomputed dynamics witness {i}',rel<1e-8 and np.max(abs(a-z['acceleration']))<1e-7,{'relative':rel})
    if 'verification' in s:
        cases=loadcsv(p/'verification_tests.csv');check('verification counts',len(cases)==s['verification']['tests'] and sum(r['success'] for r in cases)==s['verification']['passed'])
    result=dict(checks=len(checks),passed=sum(z['passed'] for z in checks),status='passed' if all(z['passed'] for z in checks) else 'failed',results=str(p),evidence=checks)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))
    return int(result['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
