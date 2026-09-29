from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from time import perf_counter_ns
import argparse,datetime,hashlib,json,sys,traceback
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from .bootstrap import ROOT
from .compiler import compile_graph,variant_inputs
from .task_graph import Solver,GeneratedTaskGraph,METHODS,LABELS,trf_solve
from .physics import PhysicalReference,validate,independent_map,dynamics_witness,curvature_witness,mx,relative
from .io_utils import save_json,save_csv,environment,sha,clean
from vendor.pacdm_original import PACDM,PointGraph,rank
from panda.task import mission_knots,sample_knots

SEED=20260925
PROFILES={
 'smoke':dict(variants=2,witnesses=2,repeats=1,warm_stride=220,cases=4,dynamics=4,curvature=2),
 'standard':dict(variants=12,witnesses=3,repeats=1,warm_stride=10,cases=24,dynamics=24,curvature=6),
 'full':dict(variants=12,witnesses=5,repeats=3,warm_stride=1,cases=48,dynamics=48,curvature=12)}

def log(msg):print(datetime.datetime.now().strftime('%H:%M:%S'),msg,flush=True)
def load_ref():
    r=dict(np.load(ROOT/'original/data/reference.npz'));t,k=mission_knots();active,av,aa=sample_knots(t,k,r['time'])
    for x in [active,av,aa]:x[:,-1]/=2
    if mx(active-r['active'])>1e-12:raise ValueError('Regenerated route differs from the original route')
    r['active'],r['active_v'],r['active_a']=active,av,aa;return r

def aggregate(rows,groups=('method',)):
    outs=[]
    for key in sorted(set(tuple(r[k]for k in groups)for r in rows),key=str):
        z=[r for r in rows if tuple(r[k]for k in groups)==key];times=[r['time_ms']for r in z]
        out=dict(zip(groups,key));out.update(attempts=len(z),accepted=sum(bool(r['success'])for r in z),median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
            max_tool_gap_m=max([r.get('max_gap_m',0)for r in z if r['success']]or[0]),max_orientation_rad=max([r.get('orientation_error_rad',0)for r in z if r['success']]or[0]),
            max_finger_gap_m=max([r.get('coupling_error_m',0)for r in z if r['success']]or[0]),median_evaluations=float(np.median([r.get('evaluations',0)for r in z])),
            fallback_count=sum(r.get('fallback',False)for r in z),reuse_count=sum(r.get('reused',False)for r in z))
        outs.append(out)
    return outs

def structure(comp,out):
    rows=[('Physical backend coordinates',9,9),('Physical coordinates explicitly parameterized',9,8),('Virtual target coordinates',6,6),
        ('Augmented coordinates',15,14),('Dependent nonlinear coordinates',7,6),('Augmented residual rows / rank','12 / 7','6 / 6'),
        ('Physical scalar finger constraints',1,1),('Independent physical mobility',8,8),('Independent task coordinates',8,8),('Physical actuators',8,8),('Structural arm loops',0,0)]
    save_csv(out/'structure.csv',[dict(quantity=a,original=b,compiled=c)for a,b,c in rows])
    save_json(out/'generated/nominal_plan.json',comp.plan);save_json(out/'generated/nominal_model.json',comp.cmg)
    return rows

def pipeline(source,ref,cfg,out):
    rows=[];compile_rows=[];seeds=ref['q'][np.linspace(120,2100,cfg['witnesses'],dtype=int)]
    for vn,s in variant_inputs(source)[:cfg['variants']]:
        t=perf_counter_ns();comp=compile_graph(s);elapsed=(perf_counter_ns()-t)*1e-6
        compile_rows.append(dict(variant=vn,time_ms=elapsed,success=True,charts=len(comp.plan['arm_ids'])))
        save_json(out/f'generated/{vn}/input.json',s);save_json(out/f'generated/{vn}/model.json',comp.cmg);save_json(out/f'generated/{vn}/plan.json',comp.plan)
        ids=comp.cmg['coordinate_ids'];refphys=PhysicalReference(comp.cmg);tree=PointGraph(dict(comp.cmg,closures=[],independent_ids=ids),comp.q_seed)
        for wi,qr in enumerate(seeds):
            named=dict(zip(s['seed'].keys(),qr));q=np.array([named[i]for i in ids]);q=comp.physical(comp.reduced(q));P,E=tree.poses(q);F=refphys.forward(q)
            pose=max(mx(P[b]-np.block([[F[b]['R'],F[b]['p'][:,None]],[np.zeros((1,3)),np.ones((1,1))]]))for b in F)
            g0=GeneratedTaskGraph(comp);T=refphys.tool(q)['T'];x=np.r_[comp.reduced(q),g0.target_coordinates(T)]
            # Off-manifold checks are shared across chart declarations; changing
            # the active-set declaration does not change the residual evaluator.
            pert=x.copy();pert[:7]+=np.array([.0007,-.0004,.0005,-.0006,.0004,.0005,-.0003]);J=g0.residual(pert)[1]
            d=1e-6;FD=np.column_stack([(g0.residual_only(pert+d*np.eye(g0.n)[i])-g0.residual_only(pert-d*np.eye(g0.n)[i]))/(2*d)for i in range(g0.n)])
            jerr=mx(J-FD);M,h,U=refphys.mass_bias(q,np.zeros(9));M+=np.diag(comp.cmg['armature']);positive=np.linalg.eigvalsh(comp.S.T@M@comp.S).min()>0
            for red in comp.plan['arm_ids']:
                g=GeneratedTaskGraph(comp,redundancy=red);x=np.r_[comp.reduced(q),g.target_coordinates(T)];N,info=PACDM(g).mapping(x)
                G=refphys.tool(q)['J'];pi=[ids.index(g.ids[i])for i in g.passive];Aj=G[:,pi]
                r=int(rank(Aj));cond=float(1/np.linalg.cond(Aj,p=1))if r==6 else 0.;expected=r==6 and cond>=1e-10
                maperr=None;gate=False
                if expected and info['success']:
                    Ni,_=independent_map(comp,q,x[g.active],red);maperr=mx(g.physical_map(N)-Ni);gate=maperr<2e-7
                elif not expected and not info['success']:gate=True
                ok=bool(gate and pose<1e-10 and jerr<2e-8 and positive and mx(comp.C@comp.S)<1e-12)
                rows.append(dict(variant=vn,witness=wi,redundancy_joint=red,independent_rank=r,independent_rcond=cond,
                    admitted=bool(info['success']),expected_admissible=bool(expected),diagnostic='admissible'if expected else 'singular_chart_rejected',
                    map_discrepancy=maperr,pose_discrepancy=pose,off_manifold_jacobian_error=jerr,physical_mass_positive=bool(positive),success=ok))
        log(f'pipeline {vn}: {sum(r["success"]for r in rows if r["variant"]==vn)}/{cfg["witnesses"]*7} checks')
    save_csv(out/'pipeline_raw.csv',rows);save_csv(out/'compilation_raw.csv',compile_rows)
    summary=dict(variants=len(compile_rows),candidate_charts=len(compile_rows)*7,witness_checks=len(rows),passed=sum(r['success']for r in rows),
        admissible_maps=sum(r['admitted']for r in rows),singular_chart_rejections=sum(not r['expected_admissible']for r in rows),
        compile_median_ms=float(np.median([r['time_ms']for r in compile_rows])),max_pose_discrepancy=max(r['pose_discrepancy']for r in rows),
        max_off_manifold_jacobian_error=max(r['off_manifold_jacobian_error']for r in rows))
    save_json(out/'pipeline_summary.json',summary);return summary

def runstep(solver,active,comp):
    start=perf_counter_ns()
    try:
        q,N,info=solver.step(active);elapsed=(perf_counter_ns()-start)*1e-6;vr=validate(comp,q,N,active)
        return dict(time_ms=elapsed,**info,**vr,error=''),q,N
    except Exception as e:
        elapsed=(perf_counter_ns()-start)*1e-6
        return dict(time_ms=elapsed,success=False,error=f'{type(e).__name__}: {e}',evaluations=0,reused=False,fallback=False),np.full(9,np.nan),np.full((9,8),np.nan)

def warm(comp,ref,cfg,out):
    idx=np.arange(0,len(ref['time']),cfg['warm_stride']);rows=[];states=[];maps=[];acts=[];rng=np.random.default_rng(SEED+1)
    # Untimed per-method dry initialization. No dry timing is reported.
    for name in METHODS:Solver(comp,comp.q_seed,name).step(ref['active'][0])
    for rep in range(cfg['repeats']):
        solvers={m:Solver(comp,comp.q_seed,m)for m in METHODS}
        for number,i in enumerate(idx):
            active=ref['active'][i]
            for method in rng.permutation(METHODS):
                row,q,N=runstep(solvers[method],active,comp);row.update(method=str(method),repeat=rep,sample=int(i),time_s=float(ref['time'][i]),record=len(rows),arm_input_changed=bool(i>0 and not np.array_equal(active[:7],ref['active'][max(0,i-cfg['warm_stride']),:7])))
                rows.append(row);states.append(q);maps.append(N);acts.append(active)
            if number%400==0:log(f'warm repeat {rep+1}/{cfg["repeats"]}, sample {number+1}/{len(idx)}')
    save_csv(out/'warm_raw.csv',rows);np.savez_compressed(out/'warm_states.npz',q=np.array(states),mapping=np.array(maps),active=np.array(acts))
    sm=aggregate(rows);save_csv(out/'warm_summary.csv',sm);save_csv(out/'warm_by_repeat.csv',aggregate(rows,('method','repeat')))
    moving=[r for r in rows if r['arm_input_changed']]
    if moving:save_csv(out/'warm_moving_summary.csv',aggregate(moving))
    return sm

def local_updates(comp,ref,cfg,out):
    methods=[m for m in METHODS if m!='compiled_no_predictor'];conditions=['gripper_only','tool_only','redundancy_only','tool_and_gripper']
    anchors=max(1,cfg['cases']//4);qseeds=ref['q'][np.linspace(200,1900,anchors,dtype=int)];rng=np.random.default_rng(SEED+2)
    rows=[];qs=[];Ns=[];az=[];seedrows=[]
    for rep in range(cfg['repeats']):
        for condition in conditions:
            for ai,q0 in enumerate(qseeds):
                q=q0.copy();q[comp.plan['master_index']]=.02;q=comp.physical(comp.reduced(q));g=GeneratedTaskGraph(comp);x=g.lift(q);a0=x[g.active].copy()
                solvers={m:Solver(comp,q,m)for m in methods}
                for k in range(4):
                    active=a0.copy()
                    if condition in ['gripper_only','tool_and_gripper']:active[7]=[.015,.025,.01,.03][k]
                    if condition in ['tool_only','tool_and_gripper']:
                        active[:3]+=np.array([1.,-.7,.5])*[.001,-.0015,.002,-.0005][k]
                        active[3:6]+=np.array([.7,-.5,1.])*[.002,-.003,.004,-.001][k]
                    if condition=='redundancy_only':active[6]+=[.003,-.004,.006,-.002][k]
                    for m in rng.permutation(methods):
                        row,qout,N=runstep(solvers[m],active,comp);row.update(method=str(m),condition=condition,repeat=rep,anchor=ai,update=k,record=len(rows))
                        rows.append(row);qs.append(qout);Ns.append(N);az.append(active);seedrows.append(q)
    save_csv(out/'local_updates_raw.csv',rows);save_csv(out/'local_updates_summary.csv',aggregate(rows,('condition','method')))
    np.savez_compressed(out/'local_updates_states.npz',q=np.array(qs),mapping=np.array(Ns),active=np.array(az),initial_physical_seed=np.array(seedrows))
    return aggregate(rows,('condition','method'))

def derivatives(comp,ref,cfg,out):
    g=GeneratedTaskGraph(comp);rng=np.random.default_rng(SEED+3);phys=PhysicalReference(comp.cmg)
    qseeds=ref['q'][np.linspace(150,2050,cfg['cases'],dtype=int)];cases=[]
    for i,q in enumerate(qseeds):
        q=q.copy();q[comp.plan['master_index']]=.02;q=comp.physical(comp.reduced(q));x=g.lift(q);N,info=PACDM(g).mapping(x)
        qtarget=q.copy();qtarget[:7]+=.006*rng.uniform(-1,1,7)
        T=phys.tool(qtarget)['T'];active=np.r_[g.target_coordinates(T),qtarget[comp.cmg['coordinate_ids'].index(comp.plan['redundancy_id'])],.02]
        pred=x[g.passive]+N[g.passive]@(active-x[g.active]);cases.append(dict(seed=q,active=active,predicted=pred))
    modes=[('analytic_dense',True,False),('fd3_dense',False,False),('analytic_sparse',True,True),('fd3_colored',False,True)]
    rows=[];qs=[];Ns=[];active_all=[]
    for rep in range(cfg['repeats']):
        for i,case in enumerate(cases):
            for k in rng.permutation(4):
                mode,anal,sparse=modes[k];start=perf_counter_ns()
                try:
                    x,ci=trf_solve(g,case['active'],case['predicted'],anal,sparse);elapsed=(perf_counter_ns()-start)*1e-6
                    N,mi=PACDM(g).mapping(x);q=g.physical(x);Np=g.physical_map(N)if N is not None else np.full((9,8),np.nan)
                    vr=validate(comp,q,Np,case['active']);vr['success']=bool(vr['success']and ci['success']and mi['success'])
                    row=dict(**vr,**{k:v for k,v in ci.items()if k!='success'},time_ms=elapsed,error='')
                except Exception as e:
                    row=dict(time_ms=(perf_counter_ns()-start)*1e-6,success=False,evaluations=0,error=repr(e));q=np.full(9,np.nan);Np=np.full((9,8),np.nan)
                row.update(method=mode,repeat=rep,case=i,record=len(rows));rows.append(row);qs.append(q);Ns.append(Np);active_all.append(case['active'])
    save_json(out/'derivative_cases.json',cases);save_csv(out/'derivative_raw.csv',rows);save_csv(out/'derivative_summary.csv',aggregate(rows))
    np.savez_compressed(out/'derivative_states.npz',q=np.array(qs),mapping=np.array(Ns),active=np.array(active_all))
    return aggregate(rows)

def dynamics(comp,ref,cfg,out,native=False):
    rng=np.random.default_rng(SEED+4);indices=np.linspace(40,2160,cfg['dynamics'],dtype=int);rows=[];states=[];cur=[];cstates=[]
    for wi,i in enumerate(indices):
        q=comp.physical(comp.reduced(ref['q'][i]));va=rng.uniform(-1,1,8)*np.array([.08,.08,.06,.3,.3,.3,.15,.018]);u=rng.uniform(-1,1,8)*np.array([12,12,10,10,3,3,3,15]);w=rng.uniform(-1,1,6)*[3,3,5,.3,.3,.3]
        r,st=dynamics_witness(comp,q,va,u,w,native);r.update(witness=wi,reference_index=int(i),reference_time_s=float(ref['time'][i]));rows.append(r);states.append(st)
    # Curvature witness values are deterministic and independent of native import.
    rngc=np.random.default_rng(SEED+5)
    for wi,i in enumerate(np.linspace(150,2050,cfg['curvature'],dtype=int)):
        q=comp.physical(comp.reduced(ref['q'][i]));va=rngc.uniform(-1,1,8)*np.array([.08,.08,.06,.35,.35,.35,.18,.018])
        for scale in [.5,1.,2.,4.]:
            r=curvature_witness(comp,q,va,scale);r.update(witness=wi,reference_index=int(i));cur.append(r);cstates.append(dict(q=q,active_velocity=va,speed_scale=scale))
    save_csv(out/'dynamics_raw.csv',rows);save_json(out/'dynamics_states.json',states);save_csv(out/'curvature_raw.csv',cur);save_json(out/'curvature_states.json',cstates)
    cur_summary=[]
    for scale in [.5,1,2,4]:
        subset=[r for r in cur if r['speed_scale']==scale];cs=dict(speed_scale=scale,witnesses=len(subset),accepted=sum(r['success']for r in subset))
        for k in ['full_linear_m_s2','full_angular_rad_s2','omitted_linear_m_s2','omitted_angular_rad_s2']:cs[k]=max(r[k]for r in subset)
        cur_summary.append(cs)
    save_csv(out/'curvature_summary.csv',cur_summary)
    maxima={k:max(float(r[k])for r in rows)for k in rows[0]if isinstance(rows[0][k],(float,int,np.floating,np.integer))and not isinstance(rows[0][k],bool)and not k.startswith('reference_')and k!='witness'}
    summary=dict(witnesses=len(rows),accepted=sum(r['success']for r in rows),maxima=maxima,curvature=cur_summary,native_executed=native)
    save_json(out/'dynamics_summary.json',summary)
    save_json(out/'native_status.json',dict(status='completed'if native else 'not_run',version=states[0]['native_version']if native else None,
        scope='Native CRBA/RNEA/FK/tool Jacobians/classical acceleration/ABA; scalar finger equality via explicit KKT on native matrices. Not constraintDynamics.',
        reason=None if native else 'Native stage not requested or Pinocchio unavailable.'))
    return summary

def source_hashes():
    files=list((ROOT/'src').glob('*.py'))+list((ROOT/'inputs').glob('*.json'))+list((ROOT/'original/vendor').glob('*.py'))+list((ROOT/'original/panda').glob('*.py'))+[ROOT/'run_franka.py']
    return {p.relative_to(ROOT).as_posix():sha(p)for p in files}

def main(argv=None):
    p=argparse.ArgumentParser(description='Reproducible Franka physical-graph framework study')
    p.add_argument('--profile',choices=PROFILES,default='full');p.add_argument('--native',choices=['off','auto','required'],default='off')
    p.add_argument('--stages',default='pipeline,warm,local,derivatives,dynamics',help='Comma-separated stages; native runs native dynamics and curvature only')
    p.add_argument('--out',default='results_local');args=p.parse_args(argv);out=Path(args.out).resolve()
    if out.exists()and any(out.iterdir()):p.error('Output folder must be empty. Preserve old evidence and choose a new --out name.')
    out.mkdir(parents=True,exist_ok=True);cfg=PROFILES[args.profile];source=json.loads((ROOT/'inputs/physical_graph.json').read_text());comp=compile_graph(source);ref=load_ref()
    native=False;status=dict(status='not_run',reason='--native off')
    if args.native!='off' or 'native' in args.stages.split(','):
        try:
            from .native import NativeOracle
            native=True;status=dict(status='available',version=NativeOracle(comp).version)
        except ImportError as e:
            status=dict(status='not_run',reason=str(e))
            save_json(out/'native_status.json',status)
            if args.native=='required' or args.stages=='native':
                save_json(out/'FAILURE.json',dict(error='Native Pinocchio required but import failed',details=str(e)))
                raise SystemExit('Pinocchio import failed. Activate the working conda environment. No native result was produced.')
    stages=args.stages.split(',')
    summary=dict(profile=args.profile,seed=SEED,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='running',new_results_only=True,
        original_core_sha256=sha(ROOT/'original/vendor/pacdm_original.py'),native_availability=status,config=cfg,stages=stages)
    save_json(out/'protocol.json',dict(profile=args.profile,config=cfg,seed=SEED,stages=stages,
        warm_duration_s=22,warm_reference_interval_s=.01,warm_timing='assembly plus physical mapping; own accepted history; initialization and physical checks excluded; random method order',
        local_timing='assembly plus physical mapping; fixed-tool and local-tool/redundancy/gripper changes; identical initial information',
        derivative_timing='corrector only, shared predictor and output map/validation excluded; matched analytic/FD TRF settings',
        acceptance=dict(tool_gap_m=1e-8,orientation_rad=1e-8,finger_gap_m=1e-8,active_coordinates=1e-12,bound_allowance=1e-12,min_rcond=1e-10),
        trf=dict(ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=1500,x_scale=1),pacdm='Unchanged original stopping/acquisition; warm 8 corrector iterations then original acquisition fallback',
        scope='No new closed-loop rollout/contact/grasp/hardware experiment. Virtual tool coordinates are not physical actuators.'))
    with threadpool_limits(limits=1):
        save_json(out/'environment.json',environment());structure(comp,out)
        try:
            for stage in stages:
                log('starting '+stage);start=perf_counter_ns()
                if stage=='pipeline':result=pipeline(source,ref,cfg,out)
                elif stage=='warm':result=warm(comp,ref,cfg,out)
                elif stage=='local':result=local_updates(comp,ref,cfg,out)
                elif stage=='derivatives':result=derivatives(comp,ref,cfg,out)
                elif stage in ['dynamics','native']:result=dynamics(comp,ref,cfg,out,native=native)
                else:raise ValueError('Unknown stage '+stage)
                summary[stage]=result;summary[stage+'_stage_seconds']=(perf_counter_ns()-start)*1e-9;save_json(out/'summary.json',summary)
                log(stage+' completed')
            good=True
            if 'pipeline'in summary:good&=summary['pipeline']['passed']==summary['pipeline']['witness_checks']
            for stage in ['warm','local','derivatives']:
                if stage in summary:good&=all(r['accepted']==r['attempts']for r in summary[stage])
            for stage in ['dynamics','native']:
                if stage in summary:good&=summary[stage]['accepted']==summary[stage]['witnesses']and all(r['accepted']==r['witnesses']for r in summary[stage]['curvature'])
            summary['status']='completed_all_declared_checks_passed'if good else 'completed_with_recorded_failures'
        except Exception:
            summary['status']='execution_error';summary['exception']=traceback.format_exc();save_json(out/'summary.json',summary);raise
        save_json(out/'source_hashes.json',source_hashes());save_json(out/'summary.json',summary)
        if not (out/'native_status.json').exists():save_json(out/'native_status.json',status)
        from .reporting import generate_report
        generate_report(out)
        save_json(out/'RUN_MANIFEST.json',{p.relative_to(out).as_posix():sha(p)for p in out.rglob('*')if p.is_file()and p.name!='RUN_MANIFEST.json'})
    log(summary['status']+'; results at '+str(out))
    return 0 if good else 2
