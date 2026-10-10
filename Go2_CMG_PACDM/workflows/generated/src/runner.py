"""Reproducible Go2 experiment stages. Raw attempts, failures and states retained."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import argparse, json, sys, traceback
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from . import bootstrap
from .bootstrap import ROOT
from .compiler import compile_graph,variant_inputs,support_sets,export_physical,ModelError,components
from .task_graph import GeneratedTaskGraph,CompiledStanceGraph,ModularSolver,create_solver,TRFOracle
from .physics import PhysicalReference,mapping_curvature,dynamics_witness,normmax,relative
from .io_utils import save_json,save_csv,sha,environment,timing_summary
from vendor.pacdm_original import PACDM,rank
from go2.contact import ContactGraph
from go2.task import target_trajectory

METHODS=['original_monolithic','compiled_monolithic','compiled_modular','modular_no_predictor','modular_no_reuse','trf_compiled_predictor','trf_modular_predictor']
DERIVATIVE_METHODS=['analytic_dense','fd3_dense','analytic_sparse','fd3_colored']
PROFILES={
 'smoke':dict(route_samples=41,repeats=1,variants=2,witnesses=2,derivative_cases=4,increment_blocks=1,increment_steps=4,dynamics_states=6,curvature_states=3),
 'standard':dict(route_samples=1301,repeats=1,variants=12,witnesses=5,derivative_cases=24,increment_blocks=2,increment_steps=12,dynamics_states=24,curvature_states=6),
 'full':dict(route_samples=2601,repeats=3,variants=12,witnesses=5,derivative_cases=48,increment_blocks=4,increment_steps=12,dynamics_states=48,curvature_states=12)}


def original_cmg():return json.loads((ROOT/'original/data/go2_cmg.json').read_text())

def supplied_reference():
    with np.load(ROOT/'original/data/reference.npz',allow_pickle=False) as r:return {k:r[k].copy()for k in r.files}


def initialize(comp,active):
    # One common feasible initial state; acquisition is outside timing for ALL methods.
    q=np.array(comp.cmg['q_reference']);g=ContactGraph(comp.cmg,q)
    return g.solve_feet(active[:6],active[6:].reshape(4,3),seed=q)[0]


def acceptance(ref,comp,q,N,active):
    geo=ref.feet_geometry(q);gap=float(np.max(np.linalg.norm(geo['points']-active[6:].reshape(4,3),axis=1)))
    records={j['id']:j for j in comp.cmg['joints']};lo=np.array([records[k]['limits']['lower']for k in comp.cmg['coordinate_ids']]);hi=np.array([records[k]['limits']['upper']for k in comp.cmg['coordinate_ids']])
    J=geo['jacobian'].reshape(12,18);res=np.c_[np.zeros((12,6)),np.eye(12)]
    tangent=normmax(J@N-res);r=int(rank(J[:,6:]));base=normmax(q[:6]-active[:6]);eye=normmax(N[:6]-np.c_[np.eye(6),np.zeros((6,12))])
    return dict(max_gap_m=gap,base_error=base,tangent_residual=tangent,rank=r,
        success=bool(np.all(np.isfinite(q)) and np.all(np.isfinite(N)) and gap<=1e-8 and base<=1e-12 and eye<1e-12 and tangent<1e-8 and r==12 and np.all(q>=lo-1e-12) and np.all(q<=hi+1e-12)))


def pipeline_stage(source,cfg,out):
    rows=[];compiles=[];witness_states=[];rng=np.random.default_rng(2026092501)
    for variant,physical in variant_inputs(source)[:cfg['variants']]:
        start=perf_counter();comp=compile_graph(physical);ms=(perf_counter()-start)*1000
        directory=out/'generated'/variant
        save_json(directory/'physical_graph.json',physical);save_json(directory/'compiled_cmg.json',comp.cmg);save_json(directory/'plan.json',comp.plan)
        modes=[('foot_tasks',None)]+[(f"support_{''.join(map(str,s))}",s)for s in support_sets()]
        save_json(directory/'support_plans.json',{name:comp.support_plan(s)for name,s in modes if s is not None})
        compiles.append(dict(variant=variant,compilation_ms=ms,mode_plans=len(modes),physical_cycles=0,modules=4,max_dependent_block=3))
        ref=PhysicalReference(comp.cmg);task=GeneratedTaskGraph(comp)
        for k in range(cfg['witnesses']):
            q=np.array(comp.cmg['q_reference']);q[:3]+=rng.uniform(-.06,.06,3);q[3:6]+=rng.uniform(-.14,.14,3);q[6:]+=rng.uniform(-.08,.08,12)
            p,J=task.points_jacobian(q);geom=ref.feet_geometry(q);qaug=task.lift(q);N,info=PACDM(task).mapping(qaug)
            Nm=ModularSolver(comp,q).step(np.r_[q[:6],p.ravel()])[1]
            h=2e-6;Jfd=np.zeros((4,3,18))
            for j in range(18):
                d=np.eye(18)[j]*h
                Jfd[:,:,j]=(ref.feet_geometry(q+d)['points']-ref.feet_geometry(q-d)['points'])/(2*h)
            point_error=normmax(p-geom['points']);jac_error=normmax(J-geom['jacobian']);fd_error=normmax(J-Jfd)
            sparsity=np.asarray(comp.plan['passive_sparsity']);actual=task.residual(qaug)[1][:,task.passive]
            zero_error=normmax(actual[sparsity==0]);M,b,U=ref.mass_bias(q,np.zeros(18))
            core_good=point_error<1e-10 and jac_error<1e-10 and fd_error<1e-7 and zero_error<1e-13
            for mode,sites in modes:
                if sites is None:
                    current=N[:18];independent=current.shape[1];r=int(rank(J.reshape(12,18)[:,6:]));tangent=normmax(J.reshape(12,18)@current-np.c_[np.zeros((12,6)),np.eye(12)])
                    mapping_error=normmax(current-Nm);passed=info['success'];mrmin=float(np.linalg.eigvalsh(current.T@M@current).min())
                else:
                    g=CompiledStanceGraph(comp,q,sites);current,si=PACDM(g).mapping(q);Ji=geom['jacobian'][sites].reshape(-1,18)
                    independent=len(g.active);r=int(rank(Ji));Ni=np.zeros((18,independent));Ni[g.active]=np.eye(independent);Ni[g.passive]=-np.linalg.solve(Ji[:,g.passive],Ji[:,g.active])
                    mapping_error=normmax(current-Ni);tangent=normmax(Ji@current);passed=si['success'];mrmin=float(np.linalg.eigvalsh(current.T@M@current).min())
                rows.append(dict(variant=variant,witness=k,mode=mode,support_count=0 if sites is None else len(sites),
                    rank=r,mobility_or_task_coordinates=independent,point_discrepancy_m=point_error,jacobian_discrepancy=jac_error,
                    independent_fd_jacobian_error=fd_error,sparsity_zero_error=zero_error,mapping_discrepancy=mapping_error,
                    tangent_residual=tangent,min_reduced_inertia_eigenvalue=mrmin,
                    success=bool(core_good and passed and mapping_error<1e-9 and tangent<1e-9 and mrmin>0)))
            witness_states.append(dict(variant=variant,witness=k,q=q))
    save_csv(out/'pipeline_raw.csv',rows);save_csv(out/'compilation_times.csv',compiles);save_json(out/'pipeline_states.json',witness_states)
    return dict(variants=len(compiles),physical_models=len(compiles),mode_plans=sum(r['mode_plans']for r in compiles),checks=len(rows),passed=sum(r['success']for r in rows),
                median_compilation_ms=float(np.median([r['compilation_ms']for r in compiles])),
                note='One task and eleven ideal two/three/four-foot support modes per variant. Known-feasible configuration checks, not cold starts.')


def warm_stage(comp,cfg,out):
    time=np.linspace(0,26,cfg['route_samples']);active,_,_,stance=target_trajectory(time)
    q0=initialize(comp,active[0]);ref=PhysicalReference(comp.cmg);rng=np.random.default_rng(2026092502)
    raw=[];state_records=[]
    save_json(out/'warm_inputs.json',dict(time=time,active=active,q_initial=q0,methods=METHODS))
    for repeat in range(cfg['repeats']):
        solvers={name:create_solver(name,comp,q0)for name in METHODS}
        for i,t in enumerate(time):
            moving=bool(i and normmax(active[i]-active[i-1])>1e-14)
            for name in rng.permutation(METHODS):
                row=dict(method=str(name),repeat=repeat,sample=i,time_s=float(t),moving=moving,success=False,error='')
                t0=perf_counter()
                try:
                    q,N,info=solvers[name].step(active[i]);row['time_ms']=(perf_counter()-t0)*1000
                    check=acceptance(ref,comp,q,N,active[i]);row.update(info);row.update(check)
                    state_records.append(dict(method=str(name),repeat=repeat,sample=i,q=q,N=N))
                except Exception as e:
                    row['time_ms']=(perf_counter()-t0)*1000;row['error']=type(e).__name__+': '+str(e)
                raw.append(row)
            if i%500==0:print(f'warm repeat {repeat+1}/{cfg["repeats"]}, sample {i}/{len(time)}',flush=True)
    save_csv(out/'warm_raw.csv',raw);summary=timing_summary(raw);save_csv(out/'warm_summary.csv',summary)
    save_csv(out/'warm_moving_summary.csv',timing_summary([r for r in raw if r['moving']]))
    save_csv(out/'warm_per_repeat.csv',timing_summary(raw,('method','repeat')))
    # Compact binary states retain every measured accepted solution/map for auditing.
    np.savez_compressed(out/'warm_states.npz',method=np.array([r['method']for r in state_records]),
        repeat=np.array([r['repeat']for r in state_records]),sample=np.array([r['sample']for r in state_records]),
        q=np.array([r['q']for r in state_records]),N=np.array([r['N']for r in state_records]))
    return dict(route_samples=len(time),repeats=cfg['repeats'],duration_s=26.,attempts=len(raw),passed=sum(r['success']for r in raw),methods=summary)


def derivative_stage(comp,cfg,out):
    source=supplied_reference();n=cfg['derivative_cases'];indices=np.linspace(210,2370,n).astype(int)
    rng=np.random.default_rng(2026092503);raw=[];inputs=[]
    for case,k in enumerate(indices):
        q=source['q'][k];qa=source['active'][k+3];g=GeneratedTaskGraph(comp);x=g.lift(q);N,info=PACDM(g).mapping(x)
        pred=x[g.passive]+N[g.passive]@(qa-x[g.active]);clipped=np.any(pred<=g.lower[g.passive]) or np.any(pred>=g.upper[g.passive])
        pred=np.clip(pred,g.lower[g.passive]+1e-10,g.upper[g.passive]-1e-10)
        inputs.append(dict(case=case,source_index=int(k),q_seed=q,active_target=qa,passive_prediction=pred,prediction_clipped=bool(clipped)))
        for repeat in range(cfg['repeats']):
            for method in rng.permutation(DERIVATIVE_METHODS):
                analytic=method.startswith('analytic');sparse=method.endswith('sparse') or method.endswith('colored')
                graph=GeneratedTaskGraph(comp);oracle=TRFOracle(graph,qa,analytic=analytic,sparse=sparse)
                jac=oracle.jac if analytic else '3-point';kwargs={}
                if not analytic and sparse:kwargs['jac_sparsity']=csr_matrix(comp.plan['passive_sparsity'])
                row=dict(method=str(method),case=case,repeat=repeat,source_index=int(k),success=False,error='')
                start=perf_counter()
                try:
                    res=least_squares(oracle.fun,pred.copy(),jac=jac,method='trf',bounds=(graph.lower[graph.passive],graph.upper[graph.passive]),
                        tr_solver='lsmr' if sparse else 'exact',ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=1500,x_scale=1.,**kwargs)
                    row['time_ms']=(perf_counter()-start)*1000;row['evaluations']=oracle.evaluations
                    full=oracle.full(res.x);nn,mi=PACDM(graph).mapping(full);check=acceptance(PhysicalReference(comp.cmg),comp,full[:18],nn[:18],qa)
                    row.update(check);row['success']=bool(check['success'] and mi['success'] and res.success);row['scipy_nfev']=int(res.nfev)
                except Exception as e:
                    row.setdefault('time_ms',(perf_counter()-start)*1000);row['evaluations']=oracle.evaluations;row['error']=type(e).__name__+': '+str(e)
                raw.append(row)
    save_json(out/'derivative_inputs.json',inputs);save_csv(out/'derivative_raw.csv',raw);s=timing_summary(raw);save_csv(out/'derivative_summary.csv',s)
    return dict(cases=n,repeats=cfg['repeats'],attempts=len(raw),passed=sum(r['success']for r in raw),methods=s)


def incremental_stage(comp,cfg,out):
    source=supplied_reference();ref=PhysicalReference(comp.cmg);rng=np.random.default_rng(2026092504)
    raw=[];inputs=[];names=['original_monolithic','compiled_monolithic','compiled_modular','modular_no_reuse','trf_compiled_predictor','trf_modular_predictor']
    starts=np.linspace(0,1700,cfg['increment_blocks']).astype(int)
    for count in (1,2,4):
        for block,k in enumerate(starts):
            q0=source['q'][k].copy();truth=q0.copy();targets=[]
            for step in range(cfg['increment_steps']):
                sites=[step%4] if count==1 else ([step%4,(step+2)%4] if count==2 else list(range(4)))
                for i in sites:
                    dep=comp.plan['point_paths'][i]['dependent_indices'];delta=.025*np.sin(.65*(step+1)+.3*i)
                    truth[dep]+=delta*np.array([.8,1.,-.6])
                target=np.r_[q0[:6],ref.feet_geometry(truth)['points'].ravel()]
                targets.append((target,truth.copy(),sites))
                inputs.append(dict(changed_feet=count,block=block,step=step,q_initial=q0,truth=truth.copy(),active=target,sites=sites))
            for repeat in range(cfg['repeats']):
                solvers={n:create_solver(n,comp,q0)for n in names}
                for step,(target,known,sites) in enumerate(targets):
                    for name in rng.permutation(names):
                        row=dict(method=str(name),changed_feet=count,block=block,step=step,repeat=repeat,success=False,error='')
                        start=perf_counter()
                        try:
                            q,N,info=solvers[name].step(target);row['time_ms']=(perf_counter()-start)*1000
                            check=acceptance(ref,comp,q,N,target);row.update(info);row.update(check);row['known_branch_error_rad']=normmax(q[6:]-known[6:]);row['success']=bool(check['success'] and row['known_branch_error_rad']<2e-6)
                        except Exception as e:row['time_ms']=(perf_counter()-start)*1000;row['error']=type(e).__name__+': '+str(e)
                        raw.append(row)
    save_json(out/'incremental_inputs.json',inputs);save_csv(out/'incremental_raw.csv',raw);s=timing_summary(raw,('changed_feet','method'));save_csv(out/'incremental_summary.csv',s)
    return dict(attempts=len(raw),passed=sum(r['success']for r in raw),methods=s,note='Fixed-base task edits, not dynamic fixed-base robot assumption. Unchanged targets are reused exactly.')


def dynamics_stage(comp,cfg,out,native):
    source=supplied_reference();indices=np.linspace(40,2560,cfg['dynamics_states']).astype(int);rng=np.random.default_rng(2026092505)
    sets=support_sets();by_count={m:[s for s in sets if len(s)==m]for m in (2,3,4)}
    raw=[];states=[];ablation=[];avinputs=[]
    for case,k in enumerate(indices):
        m=(2,3,4)[case%3];sites=by_count[m][(case//3)%len(by_count[m])];q=source['q'][k].copy()
        g=CompiledStanceGraph(comp,q,sites);va=rng.uniform(-.12,.12,len(g.active));motors=rng.uniform(-8,8,12);wrench=rng.uniform([-15,-15,-10,-2,-2,-2],[15,15,10,2,2,2])
        row=dict(case=case,source_index=int(k),time_s=float(source['time'][k]),support_count=m,support_sites=''.join(map(str,sites)),success=False,error='')
        try:
            result,state=dynamics_witness(comp,q,sites,va,motors,wrench,native=native);row.update(result);state.update(case=case,source_index=int(k));states.append(state)
            if case<cfg['curvature_states']:
                aactive=rng.uniform(-.3,.3,len(g.active))
                for speed in (.5,1.,2.,4.):
                    N,v,c,info=mapping_curvature(g,q,va*speed);geo=PhysicalReference(comp.cmg).feet_geometry(q,v)
                    J=geo['jacobian'][sites].reshape(-1,18);gamma=geo['gamma'][sites].ravel()
                    full=normmax(J@(N@aactive+c)+gamma);omitted=normmax(J@(N@aactive)+gamma)
                    ablation.append(dict(case=case,support_count=m,speed_scale=speed,full_residual_m_s2=full,omitted_residual_m_s2=omitted,success=bool(full<2e-6)))
                avinputs.append(dict(case=case,q=q,sites=sites,base_active_velocity=va,active_acceleration=aactive))
        except Exception as e:row['error']=type(e).__name__+': '+str(e)
        raw.append(row)
    save_csv(out/'dynamics_raw.csv',raw);save_json(out/'dynamics_states.json',states);save_csv(out/'curvature_ablation.csv',ablation);save_json(out/'curvature_inputs.json',avinputs)
    cols=['relative_acceleration_difference','absolute_acceleration_difference','point_acceleration_residual_m_s2','mapping_discrepancy','tangent_residual','virtual_power_defect_W','reduced_inertia_relative','kinematic_curvature_constraint','kkt_force_balance']
    if native:cols+=['native_acceleration_relative','native_mass_inf','native_bias_inf','native_point_jacobian_inf','native_point_bias_inf']
    maxima={c:max([r[c]for r in raw if r.get(c) is not None],default=None)for c in cols}
    ds=[]
    for m in (2,3,4):
        rr=[r for r in raw if r['support_count']==m];ds.append(dict(support_count=m,attempts=len(rr),passed=sum(r['success']for r in rr),
            expected_rank=3*m,expected_mobility=18-3*m,acceleration_relative=max((r.get('relative_acceleration_difference',0)for r in rr),default=0),
            constraint_residual_m_s2=max((r.get('point_acceleration_residual_m_s2',0)for r in rr),default=0)))
    save_csv(out/'support_dynamics_summary.csv',ds)
    cur=[]
    for speed in (.5,1.,2.,4.):
        rr=[r for r in ablation if r['speed_scale']==speed]
        cur.append(dict(speed_scale=speed,configurations=len(rr),full_max_m_s2=max((r['full_residual_m_s2']for r in rr),default=0),
            omitted_max_m_s2=max((r['omitted_residual_m_s2']for r in rr),default=0),passed=sum(r['success']for r in rr)))
    save_csv(out/'curvature_summary.csv',cur)
    status=dict(status='completed' if native else 'not_run',version=states[0]['native_version'] if native and states else None,
        reason=None if native else 'Native Pinocchio was not enabled or not importable; no native measurements asserted.')
    save_json(out/'native_status.json',status)
    return dict(attempts=len(raw),passed=sum(r['success']for r in raw),maxima=maxima,support_modes=ds,
        curvature_attempts=len(ablation),curvature_passed=sum(r['success']for r in ablation),native=status,
        note='Rigid-body, ideal fixed-anchor support witnesses; no contact transitions, impacts or locomotion rollout.')


def fault_stage(source,out):
    raw=[]
    def reject(name,mutate):
        s=deepcopy(source);mutate(s);message='';detected=False
        try:compile_graph(s)
        except ModelError as e:detected=True;message=str(e)
        raw.append(dict(test=name,expected='reject',success=detected,message=message))
    reject('duplicate_body',lambda s:s['bodies'].append(deepcopy(s['bodies'][0])))
    reject('duplicate_joint',lambda s:s['joints'].append(deepcopy(s['joints'][0])))
    reject('negative_mass',lambda s:s['bodies'][0].update(mass_kg=-1))
    reject('invalid_inertia',lambda s:s['bodies'][0].update(inertia_kg_m2=np.diag([1,1,-1]).tolist()))
    reject('missing_joint_endpoint',lambda s:s['joints'][0].update(body_a='missing'))
    reject('disconnected_graph',lambda s:s['joints'].pop())
    reject('nonunit_axis',lambda s:s['joints'][0].update(axis=[2,0,0]))
    reject('improper_transform',lambda s:s['joints'][0].update(T_AJ=np.zeros((4,4)).tolist()))
    reject('reversed_limits',lambda s:s['joints'][0].update(limits=dict(lower=1,upper=-1)))
    reject('unsupported_joint',lambda s:s['joints'][0].update(type='spherical'))
    reject('duplicate_actuator',lambda s:s['actuators'].append(deepcopy(s['actuators'][0])))
    reject('missing_actuator',lambda s:s['actuators'].pop())
    reject('invalid_site',lambda s:s['point_sites'][0].update(body='missing'))
    reject('preauthored_modules',lambda s:s.update(modules=[[0],[1],[2],[3]]))
    reject('singular_base_chart_seed',lambda s:s['seed']['base_pose'].__setitem__(4,np.pi/2))
    reject('missing_named_seed',lambda s:s['seed']['joints'].pop(next(iter(s['seed']['joints']))))
    reject('shared_dependent_branches_unsupported',lambda s:s['point_sites'][1].update(body=s['point_sites'][0]['body']))
    comp=compile_graph(source);q=np.array(comp.cmg['q_reference']);g=GeneratedTaskGraph(comp);p,_=g.points_jacobian(q);qa=np.r_[q[:6],p.ravel()]
    solver=ModularSolver(comp,q);_,_,i=solver.step(qa)
    raw.append(dict(test='unchanged_blocks_reused',expected='four exact reuses',success=i['skipped_modules']==4,message=str(i)))
    target=qa.copy();target[0]+=.001;qq,NN,i=solver.step(target)
    raw.append(dict(test='base_change_invalidates_all_four',expected='four modules recomputed',success=i['solved_modules']==4 and acceptance(PhysicalReference(comp.cmg),comp,qq,NN,target)['success'],message=str(i)))
    supports=[p['dependent_indices']+[0]for p in comp.plan['point_paths']]
    raw.append(dict(test='shared_base_dependent_merges_task_support',expected='one coupled candidate module',success=len(components(supports))==1,message=str(components(supports))))
    s=deepcopy(source)
    for j in s['joints']:
        j['body_a'],j['body_b']=j['body_b'],j['body_a'];j['T_AJ'],j['T_BJ']=j['T_BJ'],j['T_AJ'];j['axis']=(-np.array(j['axis'])).tolist()
    cr=compile_graph(s);pr,jr=GeneratedTaskGraph(cr).points_jacobian(q);po,jo=g.points_jacobian(q)
    raw.append(dict(test='reverse_edge_storage_preserves_kinematics',expected='same physical points and derivatives',success=normmax(pr-po)<1e-12 and normmax(jr-jo)<1e-12,message='valid reversed edge input'))
    save_csv(out/'verification_tests.csv',raw)
    return dict(tests=len(raw),passed=sum(r['success']for r in raw),note='Designed compiler/precondition/component checks; not a real-world failure detection probability.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=list(PROFILES),default='full');p.add_argument('--out',default='results_local')
    p.add_argument('--stages',nargs='+',choices=['all','pipeline','warm','derivatives','incremental','dynamics','native','checks'],default=['all'])
    p.add_argument('--native',choices=['off','auto','required'],default='off')
    a=p.parse_args(argv);out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory is not empty. Use a fresh --out name; previous results will not be overwritten.')
    native=False;import_error=None
    if a.native!='off':
        try:
            import pinocchio as pin
            if not hasattr(pin,'constraintDynamics'):raise ImportError('Imported module lacks robotics constraintDynamics')
            native=True
        except ImportError as e:
            import_error=str(e)
            if a.native=='required':p.error('Native Pinocchio required but unavailable: '+str(e)+'. Activate an environment with robotics Pinocchio installed, or install conda-forge pinocchio=3.8.0.')
    if 'native' in a.stages and not native:p.error('--stages native requires --native required or an available --native auto')
    stages=['pipeline','checks','warm','derivatives','incremental','dynamics'] if 'all' in a.stages else a.stages
    out.mkdir(parents=True,exist_ok=True);cfg=PROFILES[a.profile].copy();source=json.loads((ROOT/'inputs/physical_graph.json').read_text());comp=compile_graph(source)
    protocol=dict(profile=a.profile,configuration=cfg,stages=stages,native_requested=a.native,native_import_error=import_error,
        methods=METHODS,derivative_methods=DERIVATIVE_METHODS,seeds=[2026092501,2026092502,2026092503,2026092504,2026092505],
        physical_gap_gate_m=1e-8,rank_rcond_gate=1e-10,trf_tolerances=1e-11,blas_threads=1,
        derivative_lookahead_s=.03,curvature_fd_step='1e-5 / max(1, max(abs(v)))',
        timing_scope='warm/incremental: correction + output map; derivative: corrector only; setup and independent audit excluded equally',
        note='Predefined finite suite; all attempts and unfavorable baselines retained. New compiler/evaluator/scheduler extension; PACDM original bytes unchanged.')
    save_json(out/'protocol.json',protocol);save_json(out/'compiled_cmg.json',comp.cmg);save_json(out/'compiled_plan.json',comp.plan)
    save_json(out/'source_hashes.json',{str(f.relative_to(ROOT)).replace('\\','/'):sha(f)for f in ROOT.rglob('*')if f.is_file() and (f.suffix=='.py' or f.parent==ROOT/'inputs') and '__pycache__' not in str(f) and not f.is_relative_to(out)})
    summary=dict(status='running',profile=a.profile,stages={},protocol_sha256=sha(out/'protocol.json'))
    with threadpool_limits(limits=1):
        save_json(out/'environment.json',environment());start_all=perf_counter()
        for stage in stages:
            print('STAGE',stage,flush=True);start=perf_counter()
            try:
                if stage=='pipeline':r=pipeline_stage(source,cfg,out)
                elif stage=='checks':r=fault_stage(source,out)
                elif stage=='warm':r=warm_stage(comp,cfg,out)
                elif stage=='derivatives':r=derivative_stage(comp,cfg,out)
                elif stage=='incremental':r=incremental_stage(comp,cfg,out)
                elif stage in ('dynamics','native'):r=dynamics_stage(comp,cfg,out,native)
                r['stage_wall_s']=perf_counter()-start;summary['stages'][stage]=r
                save_json(out/'summary.json',summary);print('DONE',stage,'seconds',round(r['stage_wall_s'],2),flush=True)
            except Exception as e:
                summary.update(status='error',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());save_json(out/'summary.json',summary);raise
        summary['total_wall_s']=perf_counter()-start_all
    failures=[]
    for name,r in summary['stages'].items():
        denom=r.get('attempts',r.get('checks',r.get('tests',0)))
        if r.get('passed',0)!=denom:failures.append(name)
        if r.get('curvature_passed',0)!=r.get('curvature_attempts',0):failures.append(name+'_curvature')
    summary.update(status='completed' if not failures else 'completed_with_failures',failed_stages=failures)
    save_json(out/'summary.json',summary)
    from .reporting import generate_report
    generate_report(out)
    save_json(out/'RUN_MANIFEST.json',{str(f.relative_to(out)).replace('\\','/'):sha(f)for f in out.rglob('*')if f.is_file() and f.name!='RUN_MANIFEST.json'})
    print('RESULT:',summary['status'],'OUTPUT:',out,flush=True)
    return 0 if not failures else 2
