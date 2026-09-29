#!/usr/bin/env python3
"""One entry point: python run_comparison.py --profile full --out results.

Uses the unchanged PACDM source. Freshly executed NumPy/SciPy tests and
optional native Pinocchio tests are reported separately; the URDF+ framework
is discussed but not run as a baseline.
"""
from __future__ import annotations
import os
# Set before NumPy/SciPy import; never compare runs with different BLAS settings.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import ast
import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'original'))
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from vendor.pacdm_original import PointGraph, PACDM, select
from stewart.model import inverse_seed, rotation
from numpy_backend import NumpyTree

# Execute ONLY the original pure task_pose function, avoiding its optional
# Pinocchio imports. Its AST is copied verbatim from the bundled source.
reference_ast = ast.parse((ROOT/'original/stewart/reference.py').read_text())
pose_node = next(n for n in reference_ast.body if isinstance(n,ast.FunctionDef) and n.name=='task_pose')
namespace = {'np':np}
exec(compile(ast.Module(body=[pose_node],type_ignores=[]), 'original_task_pose','exec'), namespace)
task_pose = namespace['task_pose']

WARM_METHODS = ['PACDM','PACDM_no_homotopy','PACDM_no_predictor',
                'TRF_augmented','TRF_augmented_predictor','TRF_points']
COLD_METHODS = ['PACDM','PACDM_no_homotopy','TRF_augmented','TRF_points']
PROFILES = {
 'smoke': dict(warm_samples=31,cold_targets=2,cold_levels=2,dynamics_states=4,timing_repeats=1),
 'standard': dict(warm_samples=221,cold_targets=6,cold_levels=4,dynamics_states=24,timing_repeats=2),
 'full': dict(warm_samples=1101,cold_targets=12,cold_levels=4,dynamics_states=64,timing_repeats=3)}
LEVELS = [('1mm_0.005rad',.001,.005),('5mm_0.025rad',.005,.025),
          ('20mm_0.1rad',.02,.1),('50mm_0.25rad',.05,.25)]


def clean(obj):
    if isinstance(obj, dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean(x) for x in obj]
    if isinstance(obj,np.ndarray): return clean(obj.tolist())
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(float,np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        Path(path).write_text('',encoding='utf-8'); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(clean(v)) if isinstance(v,(dict,list,np.ndarray)) else clean(v)
                        for k,v in row.items()})


class EvaluationLimit(RuntimeError): pass


class CountedGraph:
    def __init__(self, graph, limit):
        self.graph=graph;self.calls=0;self.limit=limit
    def __getattr__(self,name):return getattr(self.graph,name)
    def residual(self,*args,**kwargs):
        if self.calls>=self.limit:raise EvaluationLimit(f'Oracle evaluation budget {self.limit} exhausted')
        self.calls+=1
        return self.graph.residual(*args,**kwargs)


def assessment(graph, backend, x, qa, truth):
    """Independent physical gate; truth is used ONLY to label the root."""
    if not np.all(np.isfinite(x)):
        return dict(physical_success=False,complete_success=False,root_match=False,
                    point_gap_m=np.inf,augmented_residual=np.inf,active_error_m=np.inf,
                    target_position_error_m=np.inf,target_rotation_error_rad=np.inf,
                    physical_rank=0,passive_rcond=0.,within_bounds=False)
    geom=backend.geometry(x[:graph.nt]);J=geom['jacobian']
    gap=float(np.max(np.linalg.norm(geom['point_difference'].reshape(-1,3),axis=1)))
    active_error=float(np.max(abs(x[graph.active]-qa)))
    bounds=bool(np.all(x>=graph.lower-1e-10) and np.all(x<=graph.upper+1e-10))
    fullrank=int(np.linalg.matrix_rank(J,tol=1e-10))
    rc=1./np.linalg.cond(J[:,backend.passive],p=1)
    try:
        r,_,D=graph.residual(x)
        augmented=float(np.max(abs(r)))
        augmented_ok=PACDM.physical_ok(r,D)
    except Exception:
        augmented=np.inf;augmented_ok=False
    pos=float(np.linalg.norm(x[:3]-truth[:3]))
    angle=float(np.linalg.norm(Rotation.from_matrix(rotation(truth[:6]).T@rotation(x[:6])).as_rotvec()))
    physical=bool(gap<=1e-8 and active_error<=1e-12 and bounds and fullrank==18 and rc>=1e-10)
    return dict(physical_success=physical,complete_success=bool(physical and augmented_ok),
                root_match=bool(physical and pos<=1e-5 and angle<=1e-5),
                point_gap_m=gap,augmented_residual=augmented,active_error_m=active_error,
                target_position_error_m=pos,target_rotation_error_rad=angle,
                physical_rank=fullrank,passive_rcond=float(rc),within_bounds=bounds)


def trial(method, graph, backend, qa, initial, truth, budget, cold=False, previous_map=None, previous_info=None):
    counter=CountedGraph(graph,budget);solver=PACDM(counter)
    x=initial.copy();x[graph.active]=qa
    meta=dict(internal_success=False,message='',fallback_count=0,solver_iterations=None,
              scipy_nfev=None,scipy_njev=None,accepted_homotopy_steps=0,rejected_homotopy_steps=0)
    point_calls=0
    start=time.perf_counter_ns()
    if not cold and method in ('PACDM','PACDM_no_homotopy','TRF_augmented_predictor'):
        x[graph.passive]=initial[graph.passive]+previous_map[graph.passive]@(qa-initial[graph.active])
    try:
        if method.startswith('PACDM'):
            if cold and method!='PACDM_no_homotopy':
                x,info=solver.acquire(qa,initial)
            else:
                if cold:
                    rows,rc,rk=select(counter.residual(x)[1][:,graph.passive])
                    if rk!=len(graph.passive) or rc<1e-10:raise ValueError('Invalid initial passive chart')
                else:
                    rows=np.asarray(previous_info['rows'])
                x,info=solver.correct(qa,x[graph.passive],None,rows,maxiter=100 if cold else 8)
                if not info['success'] and method!='PACDM_no_homotopy':
                    meta['fallback_count']=1
                    x,info=solver.acquire(qa,initial)
            meta.update(internal_success=bool(info['success']),message=info.get('message',''),
                        solver_iterations=info.get('iterations'),
                        accepted_homotopy_steps=info.get('accepted_steps',0),
                        rejected_homotopy_steps=info.get('rejected_steps',0))
        else:
            point_mode=method=='TRF_points'
            passive=backend.passive if point_mode else graph.passive
            lower=graph.lower[passive];upper=graph.upper[passive]
            cache_x=None;cache_result=None
            def evaluate(p):
                nonlocal cache_x,cache_result,point_calls
                if cache_x is None or not np.array_equal(cache_x,p):
                    q=x.copy();q[passive]=p
                    if point_mode:
                        if point_calls>=budget:raise EvaluationLimit('Point oracle evaluation budget exhausted')
                        point_calls+=1
                        g=backend.geometry(q[:graph.nt])
                        cache_result=(g['point_difference'],g['jacobian'][:,passive])
                    else:
                        r,J,_=counter.residual(q)
                        cache_result=(r,J[:,passive])
                    cache_x=p.copy()
                return cache_result
            # The warm predictor can leave bounds; symmetric handling is no
            # projection for either augmented predictor method: record failure.
            result=least_squares(lambda p:evaluate(p)[0],x[passive],jac=lambda p:evaluate(p)[1],
                                 bounds=(lower,upper),method='trf',tr_solver='exact',
                                 ftol=1e-12,xtol=1e-12,gtol=1e-12,x_scale=1.,
                                 max_nfev=budget,loss='linear')
            x[passive]=result.x
            if point_mode:x=graph.lift(x[:graph.nt])
            meta.update(internal_success=bool(result.success),message=str(result.message),
                        scipy_nfev=int(result.nfev),scipy_njev=int(result.njev))
    except (EvaluationLimit,ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        meta['message']=type(exc).__name__+': '+str(exc)
    solve_ns=time.perf_counter_ns()-start
    # Every warm method returns a map suitable for continuation. The point
    # baseline uses its OWN physical Jacobian; no PACDM map is used by it.
    map_start=time.perf_counter_ns();N=None;mi={}
    try:
        if method=='TRF_points':
            NP,geom=backend.mapping(x[:graph.nt])
            N=np.zeros((graph.n,len(graph.active)));N[:graph.nt]=NP
            mi=dict(success=True,rows=[])
        else:
            N,mi=PACDM(graph).mapping(x)
    except Exception as exc:
        mi=dict(success=False,message=str(exc))
    mapping_ns=time.perf_counter_ns()-map_start
    check=assessment(graph,backend,x,qa,truth)
    successful=bool(check['complete_success'] and mi.get('success',False))
    record=dict(method=method,success=successful,**meta,**check,
                solve_ms=solve_ns/1e6,mapping_ms=mapping_ns/1e6,
                total_ms=(solve_ns+mapping_ns)/1e6,
                combined_residual_jacobian_evaluations=counter.calls+point_calls,
                point_evaluations=point_calls,augmented_evaluations=counter.calls)
    return record,x,N,mi


def pacdm_terms(graph, state, active_velocity):
    solver=PACDM(graph)
    N,info=solver.mapping(state)
    if not info['success']:raise RuntimeError(str(info))
    v=N@active_velocity
    h=1e-5/max(1.,np.linalg.norm(v))
    J=graph.residual(state)[1]
    dJ=(graph.residual(state+h*v)[1]-graph.residual(state-h*v)[1])/(2*h)
    rows=np.asarray(info['rows'])
    c=np.zeros(graph.n)
    c[graph.passive]=-np.linalg.solve(J[np.ix_(rows,graph.passive)],(dJ@v)[rows])
    return N[:graph.nt],v[:graph.nt],c[:graph.nt],info


def self_tests(cmg, graph, backend, seed):
    tests=[]
    def check(name,value,limit,relation='<='):
        passed=bool(np.isfinite(value) and (value<=limit if relation=='<=' else value>=limit))
        tests.append(dict(name=name,value=float(value),limit=float(limit),relation=relation,passed=passed))
    truth=inverse_seed(cmg,task_pose(6.1));x=graph.lift(truth)
    rng=np.random.default_rng(seed+19)
    geom=backend.geometry(truth)
    PP,_=graph.poses(x)
    err=0.
    for name,s in geom['states'].items():
        T=np.eye(4);T[:3,:3]=s['R'];T[:3,3]=s['p']
        err=max(err,float(np.max(abs(T-PP[name]))))
    check('Independent_world_tree_vs_PACDM_FK',err,1e-12)
    check('Known_pose_physical_closure',np.max(abs(geom['point_difference'])),1e-12)
    # Off-manifold derivative checks, not only a zero-residual configuration.
    off=x.copy();off[:6]+=rng.uniform(-.01,.01,6)
    direction=rng.normal(size=graph.n);direction[graph.active]=0.;direction/=np.linalg.norm(direction)
    h=1e-6
    r,J,_=graph.residual(off)
    dr=(graph.residual(off+h*direction)[0]-graph.residual(off-h*direction)[0])/(2*h)
    check('Augmented_analytic_J_directional_FD',np.max(abs(dr-J@direction)),2e-7)
    q=off[:graph.nt];d=direction[:graph.nt]
    G=backend.geometry(q)
    dp=(backend.geometry(q+h*d)['point_difference']-backend.geometry(q-h*d)['point_difference'])/(2*h)
    check('Point_analytic_J_directional_FD',np.max(abs(dp-G['jacobian']@d)),2e-8)
    N,geom=backend.mapping(truth)
    v=N@rng.uniform(-.055,.055,6)
    G=backend.geometry(truth,v)
    dJ=(backend.geometry(truth+h*v)['jacobian']-backend.geometry(truth-h*v)['jacobian'])/(2*h)
    check('Exact_classical_gamma_vs_directional_FD',np.max(abs(G['acceleration_bias']-dJ@v)),2e-7)
    M,b,V=backend.mass_bias(truth,v)
    check('Mass_symmetry',np.max(abs(M-M.T)),1e-12)
    check('Positive_tree_mass',np.linalg.eigvalsh(M)[0],1e-7,'>=')
    check('Positive_reduced_mass',np.linalg.eigvalsh(N.T@M@N)[0],1e-5,'>=')
    # Independent Lagrange identity: b_i=(dM/dq_j v_j)v_i - d(T)/dq_i+dV/dq_i.
    derivatives=[];dV=[];basis=np.eye(backend.n)
    for e in basis:
        Mp,_,Vp=backend.mass_bias(truth+h*e,np.zeros(backend.n))
        Mm,_,Vm=backend.mass_bias(truth-h*e,np.zeros(backend.n))
        derivatives.append((Mp-Mm)/(2*h));dV.append((Vp-Vm)/(2*h))
    dM=np.asarray(derivatives)
    b_lagrange=np.einsum('kij,k,j->i',dM,v,v)-.5*np.einsum('ijk,j,k->i',dM,v,v)+np.array(dV)
    check('Classical_bias_vs_Lagrange_FD',np.max(abs(b-b_lagrange)),3e-6)
    P,v2,c,info=pacdm_terms(graph,x,rng.uniform(-.04,.04,6))
    check('PACDM_vs_independent_physical_mapping',np.max(abs(P-N)),1e-8)
    check('Physical_J_times_PACDM_map',np.max(abs(geom['jacobian']@P)),1e-9)
    good=assessment(graph,backend,x,truth[graph.active],truth)
    check('Valid_configuration_is_accepted',int(good['complete_success']),1,'>=')
    bad=x.copy();bad[0]+=.001
    check('Negative_control_displaced_platform_rejected',int(not assessment(graph,backend,bad,truth[graph.active],truth)['complete_success']),1,'>=')
    wrong_command=truth[graph.active].copy();wrong_command[0]+=.001
    check('Negative_control_wrong_command_rejected',int(not assessment(graph,backend,x,wrong_command,truth)['complete_success']),1,'>=')
    bad=x.copy();bad[0]=np.nan
    check('Negative_control_nonfinite_rejected',int(not assessment(graph,backend,bad,truth[graph.active],truth)['complete_success']),1,'>=')
    bad=x.copy();bad[0]=graph.upper[0]+.001
    check('Negative_control_joint_limit_rejected',int(not assessment(graph,backend,bad,truth[graph.active],truth)['complete_success']),1,'>=')
    # Package integrity is a correctness prerequisite.
    source_manifest=json.loads((ROOT/'ORIGINAL_SHA256.json').read_text())
    for rel,digest in source_manifest.items():
        check('Original_source_hash:'+rel,int(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest),1,'>=')
    return tests


def aggregate(rows,group_keys):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in group_keys)].append(row)
    out=[]
    for key,group in groups.items():
        good=[r for r in group if r['success']]
        moving=[r for r in group if r.get('moving',True)]
        values=np.array([r['total_ms'] for r in group])
        out.append(dict(zip(group_keys,key),attempts=len(group),successful=len(good),
            success_rate=len(good)/len(group),target_root_matches=sum(r['root_match'] for r in group),
            median_total_ms=float(np.median(values)),p95_total_ms=float(np.percentile(values,95)),
            median_solve_ms=float(np.median([r['solve_ms'] for r in group])),
            median_mapping_ms=float(np.median([r['mapping_ms'] for r in group])),
            median_moving_total_ms=float(np.median([r['total_ms'] for r in moving])) if moving else None,
            max_success_point_gap_m=max([r['point_gap_m'] for r in good],default=None),
            max_success_target_position_error_m=max([r['target_position_error_m'] for r in good],default=None),
            median_evaluations=float(np.median([r['combined_residual_jacobian_evaluations'] for r in group])),
            total_fallbacks=sum(r['fallback_count'] for r in group),
            total_homotopy_rejections=sum(r['rejected_homotopy_steps'] for r in group)))
    return out


def run_warm(cmg,graph,backend,config,args,out):
    times=np.linspace(0.,22.,config['warm_samples'])
    targets=np.array([inverse_seed(cmg,task_pose(t)) for t in times])
    initial=graph.lift(targets[0]);N0,mi0=PACDM(graph).mapping(initial)
    allrows=[];stored={};rng=np.random.default_rng(args.seed+1)
    order=list(WARM_METHODS);rng.shuffle(order)
    for method in order:
        x=initial.copy();N=N0.copy();mi=mi0.copy();states=[]
        print('Warm continuation:',method,flush=True)
        for i,(t,truth) in enumerate(zip(times,targets)):
            qa=truth[graph.active]
            row,candidate,Nnew,minew=trial(method,graph,backend,qa,x,truth,args.budget,
                                         previous_map=N,previous_info=mi)
            row.update(experiment='warm',index=i,time_s=float(t),
                       moving=bool(i>0 and np.max(abs(qa-targets[i-1,graph.active]))>1e-12))
            allrows.append(row);states.append(candidate.copy())
            if row['success']:x,N,mi=candidate,Nnew,minew
            if i and i%250==0:print(f'  {i}/{len(times)-1}',flush=True)
        stored[method]=np.asarray(states)
        save_csv(out/'warm_raw.csv',allrows)
        recent=[r for r in allrows if r['method']==method]
        print('  accepted',sum(r['success'] for r in recent),'/',len(recent),flush=True)
    np.savez_compressed(out/'warm_states.npz',time=times,truth=targets,**stored)
    return allrows


def run_cold(cmg,graph,backend,config,args,out):
    rng=np.random.default_rng(args.seed+2)
    rows=[];allseeds=[];truths=[];solved=[]
    nominal=np.asarray(cmg['geometry']['nominal_pose'])
    for target_index in range(config['cold_targets']):
        pose=nominal+rng.uniform([-.045,-.045,-.035,-.14,-.11,-.11],
                                [.045,.045,.045,.14,.11,.11])
        truth=inverse_seed(cmg,pose)
        for label,pos_noise,ang_noise in LEVELS[:config['cold_levels']]:
            physical=truth.copy()
            physical[:3]+=rng.uniform(-pos_noise,pos_noise,3)
            physical[3:6]+=rng.uniform(-ang_noise,ang_noise,3)
            for k in range(6):physical[6+3*k:8+3*k]+=rng.uniform(-ang_noise,ang_noise,2)
            before=physical.copy()
            physical=np.clip(physical,graph.lower[:graph.nt]+1e-10,graph.upper[:graph.nt]-1e-10)
            initial=graph.lift(physical)
            initgap=np.max(np.linalg.norm(backend.geometry(physical)['point_difference'].reshape(-1,3),axis=1))
            case_id=len(allseeds);allseeds.append(initial);truths.append(truth)
            order=list(COLD_METHODS);rng.shuffle(order)
            for method in order:
                row,x,_,_=trial(method,graph,backend,truth[graph.active],initial,truth,args.budget,cold=True)
                row.update(experiment='cold',case_id=case_id,target_index=target_index,level=label,
                           translation_perturbation_halfwidth_m=pos_noise,
                           angular_perturbation_halfwidth_rad=ang_noise,
                           initial_point_gap_m=float(initgap),seed_clipped=bool(np.any(before!=physical)))
                rows.append(row);solved.append(x)
            save_csv(out/'cold_raw.csv',rows)
        print('Cold-start target',target_index+1,'/',config['cold_targets'],flush=True)
    np.savez_compressed(out/'cold_states.npz',initial_states=np.array(allseeds),truth=np.array(truths),solutions=np.array(solved))
    return rows


def run_dynamics(cmg,graph,backend,config,args,out):
    rng=np.random.default_rng(args.seed+3);rows=[];native_status={};native=None
    if args.native!='off':
        try:
            import pinocchio as pin
            from stewart.pin_checks import PinConstraintOracle
            native=PinConstraintOracle(cmg)
            native_status=dict(status='available',version=pin.__version__,engine='Original native CONTACT_3D oracle')
        except Exception as exc:
            native_status=dict(status='not_run',reason=type(exc).__name__+': '+str(exc))
            if args.native=='required':raise RuntimeError('Native Pinocchio required but unavailable: '+str(exc)) from exc
    else:native_status=dict(status='not_run',reason='Disabled by --native off')
    for i in range(config['dynamics_states']):
        pose=np.asarray(cmg['geometry']['nominal_pose'])+rng.uniform(
            [-.045,-.045,-.035,-.14,-.11,-.11],[.045,.045,.045,.14,.11,.11])
        q=inverse_seed(cmg,pose);x=graph.lift(q)
        NP,geom=backend.mapping(q);active_v=rng.uniform(-.055,.055,6)
        velocity=NP@active_v
        M,_,_=backend.mass_bias(q,np.zeros(24))
        _,gravity,_=backend.mass_bias(q,np.zeros(24))
        force=NP.T@gravity+rng.uniform(-35.,35.,6)
        wrench=rng.uniform([-14.,-14.,-10.,-3.,-3.,-3.],[14.,14.,10.,3.,3.,3.])
        tau=np.zeros(24);tau[graph.active]=force;tau+=geom['platform_jacobian'].T@wrench
        def reduced():
            P,v,c,mi=pacdm_terms(graph,x,active_v)
            mass,bias,_=backend.mass_bias(q,velocity)
            # External generalized effort is held identical across methods.
            aa=np.linalg.solve(P.T@mass@P,P.T@(tau-bias-mass@c))
            return P@aa+c,dict(P=P,v=v,c=c,mass=mass,bias=bias)
        def kkt():
            g=backend.geometry(q,velocity);mass,bias,_=backend.mass_bias(q,velocity)
            J=g['jacobian'];gamma=g['acceleration_bias']
            K=np.block([[mass,J.T],[J,np.zeros((18,18))]])
            solution=np.linalg.solve(K,np.r_[tau-bias,-gamma])
            return solution[:24],dict(J=J,gamma=gamma,mass=mass,bias=bias,lam=solution[24:])
        funcs={'PACDM_reduced_NumPy':reduced,'Full_KKT_NumPy':kkt}
        if native is not None:
            funcs['Pinocchio_constraintDynamics']=lambda:(native.acceleration(q,velocity,force,wrench),{})
        values={};timings={k:[] for k in funcs}
        # Untimed first calls warm caches; repeat randomized method order.
        for name,func in funcs.items():values[name]=func()
        for rep in range(config['timing_repeats']):
            order=list(funcs);rng.shuffle(order)
            for name in order:
                t=time.perf_counter_ns();values[name]=funcs[name]();timings[name].append((time.perf_counter_ns()-t)/1e6)
        reference,ki=values['Full_KKT_NumPy'];pa,pi=values['PACDM_reduced_NumPy']
        row=dict(index=i,acceleration_abs_inf=float(np.max(abs(pa-reference))),
                 acceleration_relative_inf=float(np.max(abs(pa-reference))/max(1.,np.max(abs(reference)))),
                 pacdm_constraint_acceleration_inf=float(np.max(abs(ki['J']@pa+ki['gamma']))),
                 kkt_constraint_acceleration_inf=float(np.max(abs(ki['J']@reference+ki['gamma']))),
                 kkt_force_balance_inf=float(np.max(abs(ki['mass']@reference+ki['bias']-tau+ki['J'].T@ki['lam']))),
                 mapping_difference_inf=float(np.max(abs(pi['P']-NP))),
                 same_input_velocity_difference_inf=float(np.max(abs(pi['v']-velocity))),
                 mass_min_eigenvalue=float(np.linalg.eigvalsh(ki['mass'])[0]),
                 reduced_mass_min_eigenvalue=float(np.linalg.eigvalsh(pi['P'].T@ki['mass']@pi['P'])[0]),
                 physical_point_gap_m=float(np.max(np.linalg.norm(geom['point_difference'].reshape(-1,3),axis=1))),
                 physical_rank=int(np.linalg.matrix_rank(ki['J'])),
                 reduced_ms=float(np.median(timings['PACDM_reduced_NumPy'])),
                 kkt_ms=float(np.median(timings['Full_KKT_NumPy'])),
                 q=q.tolist(),velocity=velocity.tolist(),force_N=force.tolist(),wrench=wrench.tolist())
        if native is not None:
            native_a=values['Pinocchio_constraintDynamics'][0]
            ng=native.geometry(q,velocity)
            nm=native.backend.mass(q);nb=native.backend.bias(q,velocity)
            row.update(native_vs_kkt_acceleration_inf=float(np.max(abs(native_a-reference))),
                       native_vs_pacdm_acceleration_inf=float(np.max(abs(native_a-pa))),
                       native_vs_numpy_mass_inf=float(np.max(abs(nm-ki['mass']))),
                       native_vs_numpy_bias_inf=float(np.max(abs(nb-ki['bias']))),
                       native_vs_numpy_J_inf=float(np.max(abs(ng['jacobian']-ki['J']))),
                       native_vs_numpy_gamma_inf=float(np.max(abs(ng['acceleration_bias']-ki['gamma']))),
                       native_ms=float(np.median(timings['Pinocchio_constraintDynamics'])))
        row['passed']=bool(row['acceleration_abs_inf']<=2e-6 and row['mapping_difference_inf']<=1e-8
                           and row['pacdm_constraint_acceleration_inf']<=2e-7 and row['kkt_force_balance_inf']<=1e-8
                           and row['physical_rank']==18 and row['mass_min_eigenvalue']>0
                           and (native is None or (row['native_vs_kkt_acceleration_inf']<=2e-6
                                and row['native_vs_numpy_mass_inf']<=1e-8 and row['native_vs_numpy_bias_inf']<=1e-7)))
        rows.append(row)
        if (i+1)%16==0:print('Dynamics states',i+1,'/',config['dynamics_states'],flush=True)
    save_csv(out/'dynamics_raw.csv',rows)
    return rows,native_status


def table_md(rows,columns):
    headers=[label for key,label in columns]
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in rows:
        items=[]
        for key,_ in columns:
            value=row.get(key)
            if isinstance(value,float):text=f'{value:.6g}'
            elif value is None:text='Not available'
            else:text=str(value)
            items.append(text.replace('|','/'))
        lines.append('| '+' | '.join(items)+' |')
    return '\n'.join(lines)


def report(out,summary):
    warm=summary['warm'];cold=summary['cold'];ds=summary['dynamics']
    lines=['# Stewart CMG/PACDM comparison — freshly executed benchmark','',
      f"Profile: **{summary['profile']}**. Seed: **{summary['seed']}**. Status: **{summary['status']}**.",'',
      '## Scope',
      'The unchanged PACDM implementation is compared with bounded SciPy TRF. '
      'A separate NumPy classical rigid-body evaluator supports a full KKT dynamics reference. '
      'The NumPy reference is NOT Pinocchio, URDF+, or generalized_rbda. This run is an assembly '
      'and dynamics benchmark; it does not rerun the closed-loop control simulations.','',
      'Native Pinocchio: '+json.dumps(summary['native'],ensure_ascii=False)+'.',
      'URDF+ / generalized_rbda: **not implemented and not run**. No measurements for that framework are asserted.','',
      '## Warm continuation',
      'Each method follows the entire original 22-second mission using its own last accepted state. '
      'All methods start from the same known nominal state. No target solution is fed to a solver. '
      'The standard/full sampling intervals are 100/20 ms; these are reference sampling intervals, not physics steps. '
      'Times include assembly plus output mapping, but exclude the common independent validation. '
      'All attempts (not only successful ones) enter timing summaries. Hold samples are included; '
      'moving-only timing is also reported in CSV.','',
      table_md(warm,[('method','Method'),('attempts','Attempts'),('successful','Success'),
                    ('median_total_ms','Median ms'),('p95_total_ms','P95 ms'),
                    ('max_success_point_gap_m','Max accepted gap (m)'),('total_fallbacks','Fallbacks')]),'',
      '## Cold starts',
      'Each target is independently generated from a feasible geometric inverse solution. Only the '
      'commanded lengths and a perturbed seed are passed to solvers. Physical seeds, limits and '
      'acceptance rules are matched. Perturbations apply independently to xyz, platform angles '
      'and the twelve universal-joint angles. Seeds are clipped to shared limits; clipping is logged. '
      'The same target is reused across perturbation levels, so these are not independent hardware trials.','',
      table_md(cold,[('level','Perturbation'),('method','Method'),('attempts','Attempts'),('successful','Success'),
                    ('median_total_ms','Median ms'),('p95_total_ms','P95 ms'),
                    ('max_success_point_gap_m','Max accepted gap (m)')]),'',
      '## Dynamics',json.dumps(clean(ds),indent=2),'',
      'PACDM uses its original SE(3) mapping and directional finite-difference curvature. '
      'The KKT route uses independent physical point Jacobians and exact classical acceleration bias. '
      'The two NumPy routes share mass/bias evaluation, inertias, q, velocity, actuator forces and external wrench. '
      'Comparisons use feasible states, zero stabilization and zero regularization. '
      'Acceleration infinity norms mix scalar linear/angular coordinates; the dimensionless relative error is also reported. '
      'These are Python implementation timings, not proof of algorithmic complexity or real-time capability.','',
      '## Fairness and interpretation',
      'TRF_augmented uses the SAME 36 residual rows, 36 dependent coordinates, bounds, analytical '
      'Jacobian and unit numerical scaling as PACDM. TRF_augmented_predictor additionally receives '
      'the same tangent prediction as PACDM, with map computation counted. It is a shared-predictor '
      'baseline, not an independent differential-mapping method. TRF_points solves 18 physical '
      'point residuals in 18 physical dependent coordinates using the independent NumPy Jacobian; '
      'this is a formulation-plus-solver comparison, not a solver-only comparison.','',
      'PACDM_no_homotopy disables acquisition/fallback homotopy; PACDM_no_predictor disables '
      'the warm tangent predictor but retains recovery. PACDM itself is not edited. '
      'The existing implementation makes one global passive solve for this Stewart model; '
      'these tests do not establish module-level speedup or automatic compiler superiority.','',
      'The common external budget is '+str(summary['budget'])+' residual/Jacobian oracle evaluations per attempt. '
      'PACDM retains its internal corrector limits (100 cold / 8 warm iterations) and acquisition schedule. '
      'TRF tolerances are 1e-12; PACDM retains its original tolerances. Thus stopping rules are not identical, '
      'and extra TRF accuracy is not credited as a speed improvement. Each TRF residual/Jacobian pair is cached '
      'at identical inputs. Evaluation counts exclude output mapping and independent auditing; those overheads '
      'are separated in the raw timing columns.','',
      'Physical acceptance requires all six anchor gaps <=1e-8 m, exact commanded lengths within 1e-12 m, '
      'joint bounds, full physical rank 18 and a passive reciprocal condition number >=1e-10. '
      'The original full augmented closure gate is also applied. Root matching is separately checked '
      'against the known target pose with 1e-5 m and 1e-5 rad thresholds; it is not a global branch enumeration.','',
      'A solver failure is a benchmark observation, not automatically a harness failure. '
      'Significance tests and global convergence are outside this scope. Timing should be rerun on the same '
      'local hardware for a publication. Fully successful ablations do not demonstrate that recovery is necessary.','',
      '## Integrity and reproducibility',
      f"Self-tests: {summary['self_tests_passed']}/{summary['self_tests_total']}. "
      'Source hashes, dependency versions, platform, seeds, configuration, failures and all raw states are saved. '
      'Saved simulation metrics from earlier runs are not substituted for new measurements.','',
      '## Sources',
      '- Source package (Pinocchio.zip): original/stewart/*.py, original/vendor/pacdm_original.py and original/data/stewart.cmg.json.',
      '- SciPy 1.17.0 API (the tested version): https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html',
      '- Native Pinocchio example: https://github.com/stack-of-tasks/pinocchio/blob/v3.8.0/examples/simulation-closed-kinematic-chains.py',
      '- URDF+ associated library, not benchmarked here: https://github.com/ROAM-Lab-ND/generalized_rbda','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Portable HTML viewer; no plotting or additional packages required.
    text='<!doctype html><html><head><meta charset="utf-8"><title>Stewart comparison</title><style>body{font:16px system-ui;margin:40px;max-width:1300px}pre{white-space:pre-wrap;line-height:1.45}h1{font-size:28px}</style></head><body><h1>Stewart comparison</h1><pre>'+html.escape('\n'.join(lines))+'</pre></body></html>'
    (out/'REPORT.html').write_text(text,encoding='utf-8')
    tex=['% Generated results. Requires booktabs. Times in ms; gaps in metres.',
         '% Native Pinocchio status: '+summary['native']['status'],
         '% URDF+/generalized_rbda NOT IMPLEMENTED / NOT RUN.',
         '\\begin{table*}[t]','\\centering','\\caption{Stewart warm-continuation assembly comparison.}',
         '\\begin{tabular}{lrrrrr}','\\toprule',
         'Method & Success/attempts & Median (ms) & P95 (ms) & Max. gap (m) & Fallbacks \\\\', '\\midrule']
    for r in warm:
        gap=r['max_success_point_gap_m'];gapstr='--' if gap is None else f"{gap:.3e}"
        tex.append(r['method'].replace('_',r'\_')+f" & {r['successful']}/{r['attempts']} & {r['median_total_ms']:.3f} & {r['p95_total_ms']:.3f} & {gapstr} & {r['total_fallbacks']} \\\\")
    tex+=['\\bottomrule','\\end{tabular}','\\end{table*}','']
    (out/'tables.tex').write_text('\n'.join(tex),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile',choices=PROFILES,default='full')
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--seed',type=int,default=20260925)
    parser.add_argument('--budget',type=int,default=1500)
    parser.add_argument('--native',choices=['auto','required','off'],default='auto')
    parser.add_argument('--skip-warm',action='store_true')
    parser.add_argument('--skip-cold',action='store_true')
    parser.add_argument('--skip-dynamics',action='store_true')
    parser.add_argument('--self-test-only',action='store_true')
    args=parser.parse_args()
    if args.budget<10:parser.error('--budget must be at least 10')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();config=dict(PROFILES[args.profile])
    cmg=json.loads((ROOT/'original/data/stewart.cmg.json').read_text())
    q0=inverse_seed(cmg,cmg['geometry']['nominal_pose']);graph=PointGraph(cmg,q0);backend=NumpyTree(cmg)
    env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
             machine=platform.machine(),processor=platform.processor(),cpu_count=os.cpu_count(),
             seed=args.seed,profile=args.profile,configuration=config,
             command=sys.argv,threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
             measurement_origin='Fresh execution of this harness, not previously saved metrics')
    try:env['cpu_model']=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except Exception:pass
    save_json(out/'environment.json',env)
    print('Checking independent backend, Jacobians, acceptance gates and original source hashes...',flush=True)
    tests=self_tests(cmg,graph,backend,args.seed)
    save_json(out/'self_tests.json',tests)
    if not all(t['passed'] for t in tests):
        print(json.dumps(clean([t for t in tests if not t['passed']]),indent=2));return 2
    print('Self-tests passed:',len(tests),flush=True)
    if args.self_test_only:return 0
    warm=[] if args.skip_warm else run_warm(cmg,graph,backend,config,args,out)
    cold=[] if args.skip_cold else run_cold(cmg,graph,backend,config,args,out)
    if args.skip_dynamics:dyn=[];native=dict(status='not_run',reason='Dynamics skipped')
    else:dyn,native=run_dynamics(cmg,graph,backend,config,args,out)
    ws=aggregate(warm,['method']);cs=aggregate(cold,['level','method']);ca=aggregate(cold,['method'])
    ws.sort(key=lambda x:WARM_METHODS.index(x['method']))
    cs.sort(key=lambda x:([v[0] for v in LEVELS].index(x['level']),COLD_METHODS.index(x['method'])))
    ca.sort(key=lambda x:COLD_METHODS.index(x['method']))
    save_csv(out/'warm_summary.csv',ws);save_csv(out/'cold_summary.csv',cs);save_csv(out/'cold_overall.csv',ca)
    ds=dict(states=len(dyn),passed_states=sum(r['passed'] for r in dyn),
            maxima={k:max(r[k] for r in dyn) for k in dyn[0] if k.endswith('_inf')} if dyn else {},
            median_reduced_ms=float(np.median([r['reduced_ms'] for r in dyn])) if dyn else None,
            median_kkt_ms=float(np.median([r['kkt_ms'] for r in dyn])) if dyn else None,
            native_timing_ms=float(np.median([r['native_ms'] for r in dyn])) if native['status']=='available' else None)
    complete=not(args.skip_warm or args.skip_cold or args.skip_dynamics)
    passed=all(r['passed'] for r in dyn)
    summary=dict(status='COMPLETED' if complete and passed else ('PARTIAL_BY_REQUEST' if passed else 'DYNAMICS_CHECK_FAILED'),
                 profile=args.profile,seed=args.seed,budget=args.budget,warm=ws,cold=cs,cold_overall=ca,
                 dynamics=ds,native=native,urdf_plus=dict(status='not_implemented',reason='No robot port or measured generalized_rbda execution'),
                 self_tests_total=len(tests),self_tests_passed=sum(t['passed'] for t in tests),
                 elapsed_s=time.perf_counter()-started,config=config)
    save_json(out/'summary.json',summary);report(out,summary)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json')
           and 'results' not in str(p.relative_to(ROOT)) and p.name not in ('RUN_MANIFEST.json',)}
    save_json(out/'RUN_MANIFEST.json',dict(source_hashes=files,outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
         for p in out.iterdir() if p.is_file() and p.name!='RUN_MANIFEST.json'}))
    print('Results:',out,flush=True)
    print('Native Pinocchio:',native,flush=True)
    print('URDF+/generalized_rbda: NOT IMPLEMENTED / NOT RUN',flush=True)
    print('Dynamics passed:',ds['passed_states'],'/',ds['states'],flush=True)
    return 0 if passed else 3


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc();sys.exit(1)
