#!/usr/bin/env python
"""Audit hashes, all scalar gates, aggregates and optional saved-state replay.

Usage: python audit_results.py --results results_here --replay full --out audit_local
No native results are asserted by a NumPy replay. Native execution is a separate
run_go2.py stage; this audit leaves the given result directory unchanged.
"""
from pathlib import Path
import argparse,csv,json,hashlib,math
import numpy as np
from threadpoolctl import threadpool_limits
from src.bootstrap import ROOT
from src.compiler import compile_graph
from src.physics import PhysicalReference,dynamics_witness,mapping_curvature,normmax
from src.task_graph import CompiledStanceGraph
from src.runner import acceptance
from src.io_utils import save_json,save_csv,read_csv,sha


def audit(results,replay,out):
    results=Path(results);out=Path(out);checks=[];recomputed=[]
    def check(name,passed,**extra):checks.append(dict(check=name,passed=bool(passed),**extra))
    release_path=ROOT/'provenance/release_edits.json'
    release=json.loads(release_path.read_text())['files'] if release_path.exists() else {}
    integrity={'exact_historical_matches':0,'declared_release_edits':[]}
    def recorded_hash(label,path,expected):
        if not path.is_file():
            check(label,False,reason='missing file');return
        actual=sha(path)
        if actual==expected:
            check(label,True,comparison='exact historical hash match')
            integrity['exact_historical_matches']+=1;return
        try:relative=path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:relative=''
        edit=release.get(relative,{})
        declared=(edit.get('original_sha256')==expected and edit.get('release_sha256')==actual
                  and edit.get('classification') in ('documentation','reporting','packaging_audit'))
        if declared:
            check('declared release edit: '+relative,True,comparison='changed from recorded bytes',
                  original_sha256=expected,release_sha256=actual,reason=edit.get('reason'))
            integrity['declared_release_edits'].append(relative)
        else:
            check(label,False,comparison='historical hash mismatch',
                  original_sha256=expected,actual_sha256=actual)
    manifest=json.loads((results/'RUN_MANIFEST.json').read_text())
    for name,digest in manifest.items():recorded_hash('result hash: '+name,results/name,digest)
    original=json.loads((ROOT/'original/ORIGINAL_HASHES.json').read_text())
    for name,digest in original.items():recorded_hash('retained original: '+name,ROOT/'original'/name,digest)
    hashes=json.loads((results/'source_hashes.json').read_text())
    for name,digest in hashes.items():recorded_hash('executed source: '+name,ROOT/name,digest)
    summary=json.loads((results/'summary.json').read_text());protocol=json.loads((results/'protocol.json').read_text())
    check('protocol hash',sha(results/'protocol.json')==summary['protocol_sha256'])
    check('completed status',summary['status']=='completed')
    rawfiles=['pipeline_raw.csv','warm_raw.csv','derivative_raw.csv','incremental_raw.csv','dynamics_raw.csv','curvature_ablation.csv','verification_tests.csv']
    counts={}
    for name in rawfiles:
        path=results/name
        if not path.exists():continue
        rows=read_csv(path);counts[name]=len(rows)
        check('all success flags: '+name,all(r.get('success')=='True'for r in rows),records=len(rows))
        numericbad=[]
        for i,r in enumerate(rows):
            for k,v in r.items():
                if not v:continue
                try:num=float(v)
                except ValueError:continue
                if not math.isfinite(num):numericbad.append((i,k))
        check('finite numerical fields: '+name,not numericbad)
        if rows and 'max_gap_m'in rows[0]:check('independent toe gap gate: '+name,all(float(r['max_gap_m'])<=1e-8 for r in rows))
    for rawname,summaryname,groups in [('warm_raw.csv','warm_summary.csv',['method']),('derivative_raw.csv','derivative_summary.csv',['method']),('incremental_raw.csv','incremental_summary.csv',['changed_feet','method'])]:
        if not (results/rawname).exists():continue
        raw=read_csv(results/rawname)
        for s in read_csv(results/summaryname):
            rr=[r for r in raw if all(r[g]==s[g]for g in groups)];v=np.array([float(r['time_ms'])for r in rr]);tag=rawname+': '+','.join(s[g]for g in groups)
            check('attempt count '+tag,len(rr)==int(s['attempts']))
            check('median '+tag,abs(np.median(v)-float(s['median_ms']))<1e-10)
            check('p95 '+tag,abs(np.quantile(v,.95)-float(s['p95_ms']))<1e-10)
    if (results/'curvature_ablation.csv').exists():
        raw=read_csv(results/'curvature_ablation.csv')
        for case in sorted(set(r['case']for r in raw)):
            rr={float(r['speed_scale']):r for r in raw if r['case']==case}
            if 1. in rr:
                one=float(rr[1.]['omitted_residual_m_s2'])
                check('curvature speed-squared '+case,all(abs(float(r['omitted_residual_m_s2'])-s*s*one)<=1e-9*max(1.,s*s*one)for s,r in rr.items()))
    replay_counts={}
    if replay!='none':
        c=compile_graph(json.loads((ROOT/'inputs/physical_graph.json').read_text()));ref=PhysicalReference(c.cmg)
        with threadpool_limits(limits=1):
            if (results/'warm_states.npz').exists():
                inputs=json.loads((results/'warm_inputs.json').read_text());active=np.array(inputs['active']);cache={};bad=0;maximum=0.
                with np.load(results/'warm_states.npz',allow_pickle=False)as z:
                    ids=np.arange(len(z['q'])) if replay=='full' else np.unique(np.linspace(0,len(z['q'])-1,min(140,len(z['q']))).astype(int))
                    for i in ids:
                        q=z['q'][i];N=z['N'][i];sample=int(z['sample'][i]);key=q.tobytes()+N.tobytes()+str(sample).encode()
                        if key not in cache:cache[key]=acceptance(ref,c,q,N,active[sample])
                        r=cache[key];bad+=not r['success'];maximum=max(maximum,r['max_gap_m'])
                replay_counts['warm_states']=len(ids);replay_counts['unique_numerical_warm_checks']=len(cache)
                check('independent saved warm state/map replay',bad==0,records=len(ids),unique_checks=len(cache),max_gap_m=maximum)
            if (results/'dynamics_states.json').exists():
                states=json.loads((results/'dynamics_states.json').read_text())
                for s in states:
                    r,_=dynamics_witness(c,np.array(s['q']),s['sites'],np.array(s['active_velocity']),np.array(s['motors']),np.array(s['wrench']),native=False)
                    check('dynamics replay '+str(s['case']),r['success']);recomputed.append(dict(case=s['case'],**r))
                replay_counts['dynamics_states']=len(states)
            if (results/'curvature_inputs.json').exists():
                states=json.loads((results/'curvature_inputs.json').read_text());count=0
                for s in states:
                    q=np.array(s['q']);sites=s['sites'];g=CompiledStanceGraph(c,q,sites)
                    for speed in (.5,1.,2.,4.):
                        N,v,cc,info=mapping_curvature(g,q,np.array(s['base_active_velocity'])*speed);geo=ref.feet_geometry(q,v)
                        J=geo['jacobian'][sites].reshape(-1,18);gamma=geo['gamma'][sites].ravel();r=normmax(J@(N@np.array(s['active_acceleration'])+cc)+gamma)
                        check(f'curvature replay {s["case"]} x{speed}',r<2e-6);count+=1
                replay_counts['curvature_cases']=count
    result=dict(passed=all(r['passed']for r in checks),checks=len(checks),passed_checks=sum(r['passed']for r in checks),raw_record_counts=counts,replay=replay,replayed=replay_counts,integrity=integrity,
        note='Hash and numerical consistency checks are heterogeneous, not independent experiments. Native values are audited, not recreated by NumPy replay.',details=checks)
    out.mkdir(parents=True,exist_ok=True);save_json(out/'audit.json',result);save_csv(out/'checks.csv',checks)
    if recomputed:save_csv(out/'recomputed_dynamics.csv',recomputed)
    print(json.dumps({k:v for k,v in result.items()if k!='details'},indent=2))
    return 0 if result['passed']else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='results_here');p.add_argument('--replay',choices=['none','sampled','full'],default='sampled');p.add_argument('--out',default='audit_local');a=p.parse_args();raise SystemExit(audit(a.results,a.replay,a.out))
