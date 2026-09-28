"""Production experiments. Every case/method is retained, including failures."""
from __future__ import annotations
import csv, hashlib, importlib.metadata, json, platform, time, traceback
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_array
from .bootstrap import ROOT,PRIOR
from .compiler import compile_graph,from_original_cmg,canonical_hash,ModelError
from .experiments import *
from vendor.pacdm_original import PointGraph,PACDM,select
from run_comparison import clean,save_json,save_csv,pacdm_terms


def median(values):return float(np.median(values))

def summarize(rows,keys,fields):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for group,rr in groups.items():
        x=dict(zip(keys,group));x.update(attempts=len(rr),success=sum(bool(r.get('success',True)) for r in rr))
        for f in fields:
            vals=[r[f] for r in rr if r.get(f) is not None and np.isfinite(r[f])]
            x[f+'_median']=median(vals) if vals else None
            x[f+'_p95']=float(np.quantile(vals,.95)) if vals else None
            x[f+'_max']=float(max(vals)) if vals else None
        out.append(x)
    return out


def environment():
    versions={}
    for k in ('numpy','scipy','pin','pinocchio'):
        try:versions[k]=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:versions[k]=None
    return dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),
                packages=versions,processor=platform.processor(),blas_threads=1,
                timing_note='Wall-clock implementation timings; no claim of hardware independence or real-time execution')


def pipeline_stage(nominal,config,protocol,out):
    raw=[];compile_rows=[];examples=out/'generated';examples.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(protocol['seed']+1)
    for vindex,(name,cmg) in enumerate(variants(nominal)[:config['variants']]):
        source=from_original_cmg(cmg);reference=NumpyTree(cmg)
        spherical=sorted(j['id'] for j in source['joints'] if j['type']=='spherical')
        for ti,chosen in enumerate(spherical[:config['tree_choices']]):
            elapsed=[]
            for repeat in range(3):
                start=time.perf_counter_ns();comp=compile_graph(source,chosen);elapsed.append((time.perf_counter_ns()-start)/1e6)
            rec=dict(variant=name,spherical_tree_edge=chosen,compile_ms=median(elapsed),
                physical_cycles=comp.plan['physical_cycles'],physical_bodies=comp.plan['physical_bodies'],
                physical_joints=comp.plan['physical_joints'],physical_tree_nq=len(comp.cmg['coordinate_ids']),
                augmented_nq=comp.plan['augmented_coordinates'],augmented_rows=comp.plan['augmented_residual_rows'],
                point_rows=comp.plan['point_residual_rows'],candidate_modules=len(comp.plan['candidate_modules']),
                manual_tree_labels=0,manual_constraint_functions=0,input_sha256=canonical_hash(source),model_sha256=canonical_hash(comp.cmg))
            compile_rows.append(rec)
            dest=examples/name/chosen;dest.mkdir(parents=True,exist_ok=True)
            save_json(dest/'physical_graph.json',source);save_json(dest/'topology_plan.json',comp.plan);save_json(dest/'compiled_model.json',comp.cmg)
            qzero,qoldzero,_=make_state(comp,cmg,reference,0.)
            graph=comp.graph(qzero);backend=NumpyTree(comp.cmg);oldgraph=PointGraph(cmg,qoldzero)
            for sample,t in enumerate(protocol['pipeline_probe_times_s'][:config['pipeline_samples']]):
                row=dict(variant=name,spherical_tree_edge=chosen,sample=sample,time_s=t,success=False,error='')
                try:
                    q,qold,W=make_state(comp,cmg,reference,t);x=graph.lift(q);xold=oldgraph.lift(qold)
                    N,mi=PACDM(graph).mapping(x);No,mio=PACDM(oldgraph).mapping(xold)
                    if not mi['success'] or not mio['success']:raise ValueError('Numerical coordinate rank/closure gate failed')
                    P=N[:graph.nt];Po=No[:oldgraph.nt]
                    check=physical_acceptance(graph,backend,source,x,q[graph.active],W)
                    M,b,U=backend.mass_bias(q,np.zeros(graph.nt));Mo,bo,Uo=reference.mass_bias(qold,np.zeros(oldgraph.nt))
                    reduced_error=float(np.max(abs(P.T@M@P-Po.T@Mo@Po))/max(1.,np.max(abs(Po.T@Mo@Po))))
                    task=backend.geometry(q)['platform_jacobian']@P;task_old=reference.geometry(qold)['platform_jacobian']@Po
                    task_error=float(np.max(abs(task-task_old)))
                    off=x.copy();off[graph.passive]+=.002*rng.uniform(-1,1,len(graph.passive))
                    direction=rng.normal(size=graph.n);direction/=np.linalg.norm(direction);h=1e-6
                    f=ResidualOnly(graph);r,J,_=graph.residual(off);mask=structural_sparsity(graph)
                    residual_match=float(np.max(abs(f(off)-r)))
                    derivative_error=float(np.max(abs((f(off+h*direction)-f(off-h*direction))/(2*h)-J@direction)))
                    structural_zero=float(np.max(abs(J[~mask]))) if np.any(~mask) else 0.
                    row.update(check,reduced_mass_relative_error=reduced_error,task_mapping_inf=task_error,
                        residual_only_match_inf=residual_match,off_manifold_derivative_inf=derivative_error,
                        outside_generated_sparsity_inf=structural_zero,potential_difference_J=float(abs(U-Uo)))
                    row['success']=bool(check['success'] and check['pose_error']<1e-10 and reduced_error<1e-9
                          and task_error<1e-9 and residual_match<1e-11 and derivative_error<1e-7 and structural_zero<1e-12 and abs(U-Uo)<1e-9)
                except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
                raw.append(row)
        print(f'Pipeline {vindex+1}/{config["variants"]}: {name}',flush=True)
    save_csv(out/'pipeline_raw.csv',raw);save_csv(out/'compilation_raw.csv',compile_rows)
    summary=summarize(raw,['variant'],['gap_m','pose_error','reduced_mass_relative_error','task_mapping_inf','off_manifold_derivative_inf'])
    save_csv(out/'pipeline_summary.csv',summary)
    return dict(attempts=len(raw),passed=sum(r['success'] for r in raw),models=len(compile_rows),summary=summary,
                median_compile_ms=median([r['compile_ms'] for r in compile_rows]))


class Counter:
    def __init__(self,g):self.g=g;self.calls=0
    def __getattr__(self,k):return getattr(self.g,k)
    def residual(self,*args,**kwargs):self.calls+=1;return self.g.residual(*args,**kwargs)


def warm_step(method,g,qa,prev,Nprev,info_prev):
    counter=Counter(g);solver=PACDM(counter);x=prev.copy();x[g.active]=qa
    start=time.perf_counter_ns();recovery=False;message='';success_internal=False
    try:
        if method!='PACDM_compiled_no_predictor':x[g.passive]+=Nprev[g.passive]@(qa-prev[g.active])
        if method=='TRF_compiled_analytic_predictor':
            cache_x=None;cache_value=None
            def evaluate(p):
                nonlocal cache_x,cache_value
                if cache_x is None or not np.array_equal(cache_x,p):
                    z=x.copy();z[g.passive]=p;r,J,_=counter.residual(z);cache_value=(r,J[:,g.passive]);cache_x=p.copy()
                return cache_value
            res=least_squares(lambda p:evaluate(p)[0],x[g.passive],jac=lambda p:evaluate(p)[1],
                bounds=(g.lower[g.passive],g.upper[g.passive]),method='trf',tr_solver='exact',
                ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale=1.,max_nfev=120)
            x[g.passive]=res.x;message=str(res.message);success_internal=bool(res.success)
        else:
            x,info=solver.correct(qa,x[g.passive],None,np.asarray(info_prev['rows']),maxiter=8)
            if not info['success']:
                recovery=True;x,info=solver.acquire(qa,prev)
            success_internal=bool(info['success']);message=info.get('message','')
        map_start=time.perf_counter_ns();N,mi=solver.mapping(x);map_ms=(time.perf_counter_ns()-map_start)/1e6
        success_internal=success_internal and bool(mi['success'])
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        message=type(exc).__name__+': '+str(exc);N=None;mi={};map_ms=0.
    elapsed=(time.perf_counter_ns()-start)/1e6
    return x,N,mi,dict(total_ms=elapsed,mapping_ms=map_ms,oracle_evaluations=counter.calls,
                      recovery=recovery,internal_success=success_internal,message=message)


def warm_stage(nominal,config,protocol,out):
    source=from_original_cmg(nominal);comp=compile_graph(source);ref=NumpyTree(nominal);back=NumpyTree(comp.cmg)
    times=np.linspace(0,22,config['warm_samples']);targets=[];world=[];commands=[]
    for t in times:
        qold=inverse_seed(nominal,task_pose(t));targets.append(qold);world.append(world_poses(ref,qold))
        commands.append(qold[[nominal['coordinate_ids'].index(k) for k in nominal['independent_ids']]])
    qnew0=comp.seed_from_body_poses(dict(zip(nominal['coordinate_ids'],targets[0])),world[0])
    methods=['PACDM_original_chart','PACDM_compiled_tree','PACDM_compiled_no_predictor','TRF_compiled_analytic_predictor']
    graphs={m:(PointGraph(nominal,targets[0]) if m=='PACDM_original_chart' else PointGraph(comp.cmg,qnew0)) for m in methods}
    backs={m:(ref if m=='PACDM_original_chart' else back) for m in methods}
    raw=[];allstates={};rng=np.random.default_rng(protocol['seed']+2)
    for m,g in graphs.items():
        x=g.lift(targets[0] if m=='PACDM_original_chart' else qnew0);N,mi=PACDM(g).mapping(x)
        warm_step(m,g,commands[0],x,N,mi)
    for repeat in range(config['warm_repeats']):
        previous={};maps={};infos={};states={m:[] for m in methods}
        for m,g in graphs.items():
            previous[m]=g.lift(targets[0] if m=='PACDM_original_chart' else qnew0);maps[m],infos[m]=PACDM(g).mapping(previous[m])
        for sample,(t,qa) in enumerate(zip(times,commands)):
            moving=bool(sample>0 and np.max(abs(qa-commands[sample-1]))>1e-12)
            for m in rng.permutation(methods):
                g=graphs[m];x,N,mi,info=warm_step(m,g,np.asarray(qa),previous[m],maps[m],infos[m])
                try:check=physical_acceptance(g,backs[m],source,x,qa,world[sample])
                except Exception as exc:check=dict(success=False,gap_m=np.inf,pose_error=np.inf,validation_error=str(exc))
                check['success']=bool(check['success'] and N is not None and check['pose_error']<=1e-7)
                raw.append(dict(method=str(m),repeat=repeat,sample=sample,time_s=float(t),moving=moving,**info,**check))
                states[m].append(x)
                if check['success']:previous[m]=x;maps[m]=N;infos[m]=mi
            if sample and sample%250==0:print(f'Continuation repeat {repeat+1}: {sample}/{len(times)}',flush=True)
        for m in methods:allstates[f'{m}_repeat_{repeat}']=np.asarray(states[m])
        print(f'Continuation repeat {repeat+1}/{config["warm_repeats"]} complete',flush=True)
    save_csv(out/'warm_raw.csv',raw)
    summary=summarize(raw,['method'],['total_ms','mapping_ms','oracle_evaluations','gap_m','pose_error'])
    moving_summary=summarize([r for r in raw if r['moving']],['method'],['total_ms','oracle_evaluations','gap_m'])
    repeat_summary=summarize(raw,['method','repeat'],['total_ms','gap_m'])
    save_csv(out/'warm_summary.csv',summary);save_csv(out/'warm_moving_summary.csv',moving_summary);save_csv(out/'warm_repeat_summary.csv',repeat_summary)
    np.savez_compressed(out/'warm_states.npz',time=times,commanded_lengths=np.asarray(commands),**allstates)
    return dict(attempts=len(raw),passed=sum(r['success'] for r in raw),summary=summary,moving_summary=moving_summary,repeat_summary=repeat_summary)


def derivative_trial(method,graph,initial,qa,protocol):
    x=initial.copy();x[graph.active]=qa;counter=Counter(graph);f=ResidualOnly(graph);calls=0;cached_x=None;cached=None
    settings=protocol['derivative_comparison'];mask=structural_sparsity(graph)[:,graph.passive]
    def evaluate(p):
        nonlocal cached_x,cached,calls
        if cached_x is None or not np.array_equal(cached_x,p):
            if calls>=settings['common_oracle_budget']:raise RuntimeError('Oracle budget exhausted')
            calls+=1;z=x.copy();z[graph.passive]=p
            if 'analytic' in method:
                r,J,_=counter.residual(z);cached=(r,J[:,graph.passive])
            else:cached=(f(z),None)
            cached_x=p.copy()
        return cached
    kwargs={}
    if 'analytic' in method:
        if method.endswith('sparse'):jac=lambda p:csr_array(evaluate(p)[1]);kwargs['tr_solver']='lsmr'
        else:jac=lambda p:evaluate(p)[1];kwargs['tr_solver']='exact'
    else:
        jac='3-point'
        if method.endswith('colored'):kwargs['jac_sparsity']=csr_array(mask);kwargs['tr_solver']='lsmr'
        else:kwargs['tr_solver']='exact'
    if kwargs['tr_solver']=='lsmr':kwargs['tr_options']=dict(atol=1e-12,btol=1e-12,maxiter=150)
    start=time.perf_counter_ns();message='';internal=False;nfev=njev=0
    try:
        result=least_squares(lambda p:evaluate(p)[0],x[graph.passive],jac=jac,
            bounds=(graph.lower[graph.passive],graph.upper[graph.passive]),method='trf',
            ftol=settings['trf_tolerances'],xtol=settings['trf_tolerances'],gtol=settings['trf_tolerances'],
            x_scale=1.,max_nfev=settings['max_nfev'],**kwargs)
        x[graph.passive]=result.x;message=str(result.message);internal=bool(result.success);nfev=int(result.nfev);njev=int(result.njev)
    except Exception as exc:message=type(exc).__name__+': '+str(exc)
    elapsed=(time.perf_counter_ns()-start)/1e6
    return x,dict(corrector_ms=elapsed,oracle_evaluations=calls,combined_analytic_calls=counter.calls,
                  residual_only_calls=f.calls,scipy_nfev=nfev,scipy_njev=njev,internal_success=internal,message=message)


def derivative_stage(nominal,config,protocol,out):
    source=from_original_cmg(nominal);comp=compile_graph(source);ref=NumpyTree(nominal);backend=NumpyTree(comp.cmg)
    q0,_,_=make_state(comp,nominal,ref,0.);graph=comp.graph(q0);raw=[];saved=[]
    times=np.linspace(.50,21.5,config['derivative_samples']);rng=np.random.default_rng(protocol['seed']+3)
    methods=protocol['derivative_comparison']['methods']
    for sample,t in enumerate(times):
        qprev,_,_=make_state(comp,nominal,ref,t-protocol['derivative_comparison']['shared_previous_reference_offset_s'])
        qtarget,_,truth=make_state(comp,nominal,ref,t);qa=qtarget[graph.active]
        prev=graph.lift(qprev);N,mi=PACDM(graph).mapping(prev)
        initial=prev.copy();initial[graph.passive]+=N[graph.passive]@(qa-prev[graph.active]);initial[graph.active]=qa
        for repeat in range(config['derivative_repeats']):
            for m in rng.permutation(methods):
                x,info=derivative_trial(str(m),graph,initial,qa,protocol)
                try:check=physical_acceptance(graph,backend,source,x,qa,truth)
                except Exception as exc:check=dict(success=False,gap_m=np.inf,pose_error=np.inf,validation_error=str(exc))
                check['success']=bool(check['success'] and check['pose_error']<=1e-7)
                raw.append(dict(method=str(m),sample=sample,repeat=repeat,time_s=float(t),**info,**check));saved.append(x)
        if (sample+1)%8==0:print(f'Derivative comparison {sample+1}/{len(times)}',flush=True)
    save_csv(out/'derivative_raw.csv',raw)
    summary=summarize(raw,['method'],['corrector_ms','oracle_evaluations','gap_m','pose_error'])
    save_csv(out/'derivative_summary.csv',summary);np.savez_compressed(out/'derivative_states.npz',states=np.array(saved),times=times)
    return dict(attempts=len(raw),passed=sum(r['success'] for r in raw),summary=summary,
                timing_scope='Corrector only; same upstream predictor and downstream map/audit excluded equally')


def dynamics_stage(nominal,config,protocol,out,native='off'):
    source=from_original_cmg(nominal);comp=compile_graph(source);reference=NumpyTree(nominal);backend=NumpyTree(comp.cmg)
    qzero,qoldzero,_=make_state(comp,nominal,reference,0.);graph=comp.graph(qzero);oldgraph=PointGraph(nominal,qoldzero)
    rng=np.random.default_rng(protocol['seed']+4);times=np.linspace(.4,21.6,config['dynamics_samples']);raw=[];negative=[];saved=[]
    native_oracle=None;native_status=dict(status='disabled')
    if native!='off':
        try:
            import pinocchio as pin
            from stewart.pin_checks import PinConstraintOracle
            native_oracle=PinConstraintOracle(comp.cmg)
            native_status=dict(status='running',version=pin.__version__)
        except Exception as exc:
            native_status=dict(status='not_run',reason=type(exc).__name__+': '+str(exc),code='prior_benchmark/original/stewart/pin_checks.py and src/runner.py dynamics_stage',installation='conda install -c conda-forge pinocchio')
            if native=='required':raise RuntimeError('Native Pinocchio requested but unavailable: '+native_status['reason']) from exc
    for sample,t in enumerate(times):
        q,qold,W=make_state(comp,nominal,reference,t);x=graph.lift(q);xold=oldgraph.lift(qold)
        va=rng.uniform(-.10,.10,6);forces=rng.uniform(-100,180,6);wrench=rng.uniform([-40,-40,-30,-8,-8,-8],[40,40,30,8,8,8])
        row=dict(sample=sample,time_s=float(t),success=False,error='')
        try:
            a,red=reduced_acceleration(graph,backend,x,va,forces,wrench)
            aK,kkt=kkt_acceleration(backend,q,red['v'],forces,wrench)
            No,io=PACDM(oldgraph).mapping(xold);vo=No[:oldgraph.nt]@va
            aOld,kktOld=kkt_acceleration(reference,qold,vo,forces,wrench)
            physical=[b['id'] for b in source['bodies']]
            verr,aerr=body_motion_errors(backend,q,red['v'],a,reference,qold,vo,aOld,physical)
            J=red['geometry']['jacobian'];gamma=red['geometry']['acceleration_bias']
            constraint=float(np.max(abs(J@a+gamma)))
            rel=float(np.max(abs(a-aK))/max(1.,np.max(abs(aK))))
            tau=red['tau'];power=float(abs(tau@red['v']-(red['N'].T@tau)@va))
            MrOld=No[:oldgraph.nt].T@kktOld['M']@No[:oldgraph.nt]
            mrerr=float(np.max(abs(red['Mr']-MrOld))/max(1.,np.max(abs(MrOld))))
            row.update(relative_acceleration_difference=rel,body_velocity_difference=verr,body_acceleration_difference=aerr,
                physical_acceleration_constraint_m_s2=constraint,virtual_power_defect_W=power,
                reduced_mass_relative=mrerr,min_reduced_mass_eigenvalue=float(np.linalg.eigvalsh(red['Mr']).min()),
                force_balance_inf=kkt['force_residual'])
            row['success']=bool(rel<1e-8 and aerr<2e-6 and verr<1e-9 and constraint<2e-6 and power<1e-9 and mrerr<1e-9 and row['min_reduced_mass_eigenvalue']>0)
            if native_oracle is not None:
                an=native_oracle.acceleration(q,red['v'],forces,wrench)
                native_error=float(np.max(abs(an-a))/max(1.,np.max(abs(an))))
                native_mass=native_oracle.backend.mass(q);native_bias=native_oracle.backend.bias(q,red['v'])
                row.update(native_acceleration_relative=native_error,native_mass_inf=float(np.max(abs(native_mass-red['M']))),
                           native_bias_inf=float(np.max(abs(native_bias-red['b']))))
                row['success']=bool(row['success'] and native_error<1e-7 and row['native_mass_inf']<1e-8 and row['native_bias_inf']<1e-7)
            if sample<min(12,len(times)):
                for speed in (.5,1.,2.,4.):
                    N,v,c,info=pacdm_terms(graph,x,va*speed);geom=backend.geometry(q,v);qa_dd=rng.uniform(-.15,.15,6)
                    good=float(np.max(abs(geom['jacobian']@(N@qa_dd+c)+geom['acceleration_bias'])))
                    omitted=float(np.max(abs(geom['jacobian']@(N@qa_dd)+geom['acceleration_bias'])))
                    negative.append(dict(sample=sample,speed_scale=speed,full_curvature_residual_m_s2=good,
                        omitted_curvature_residual_m_s2=omitted,success=bool(good<2e-6),
                        label='Explicit acceleration-curvature ablation; NOT an external algorithm baseline'))
            saved.append(dict(q=q,q_original=qold,active_velocity=va,forces=forces,wrench=wrench,acceleration=a))
        except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
        raw.append(row)
    if native_oracle is not None:native_status['status']='completed'
    save_csv(out/'dynamics_raw.csv',raw);save_csv(out/'curvature_ablation.csv',negative)
    save_json(out/'dynamics_states.json',saved);save_json(out/'native_status.json',native_status)
    maxima={key:max([r[key] for r in raw if key in r],default=None) for key in ['relative_acceleration_difference','body_velocity_difference','body_acceleration_difference','physical_acceleration_constraint_m_s2','virtual_power_defect_W','reduced_mass_relative']}
    return dict(attempts=len(raw),passed=sum(r['success'] for r in raw),maxima=maxima,native=native_status,
                curvature_ablations=len(negative),curvature_full_passed=sum(r['success'] for r in negative))


def fault_stage(nominal,out):
    source=from_original_cmg(nominal);cases=[]
    def case(name,fn,expected):
        s=deepcopy(source);fn(s);detected=False;message=''
        try:compile_graph(s)
        except ModelError as exc:detected=True;message=str(exc)
        cases.append(dict(test=name,expected=expected,detected=detected,success=detected,message=message))
    case('duplicate_body',lambda s:s['bodies'].append(deepcopy(s['bodies'][0])),'schema rejection')
    case('duplicate_joint',lambda s:s['joints'].append(deepcopy(s['joints'][0])),'schema rejection')
    case('missing_endpoint',lambda s:s['joints'][0].update(body_a='does_not_exist'),'schema rejection')
    case('negative_mass',lambda s:s['bodies'][0].update(mass_kg=-1.),'physical parameter rejection')
    case('invalid_inertia',lambda s:s['bodies'][0].update(inertia_kg_m2=np.diag([1,1,-1]).tolist()),'physical parameter rejection')
    case('malformed_transform',lambda s:s['joints'][0].update(T_AJ=np.zeros((4,4)).tolist()),'rigid-transform rejection')
    def bad_axis(s):next(j for j in s['joints'] if j['type']=='revolute')['axis']=[2.,0.,0.]
    case('nonunit_axis',bad_axis,'joint-axis rejection')
    def bad_bounds(s):next(j for j in s['joints'] if j['type']=='prismatic')['limits'].update(lower=2,upper=1)
    case('reversed_limits',bad_bounds,'joint-bound rejection')
    case('duplicate_actuator',lambda s:s['actuators'].append(deepcopy(s['actuators'][0])),'actuation rejection')
    case('invalid_actuator_reference',lambda s:s['actuators'][0].update(joint_id='absent'),'actuation rejection')
    case('disconnected_graph',lambda s:s['joints'].__setitem__(slice(None),[j for j in s['joints'] if j['id']!='payload_mount']),'connectivity rejection')
    case('preauthored_tree_not_physical_input',lambda s:s.update(tree=[]),'input-contract rejection')
    s=deepcopy(source);s['actuators']=s['actuators'][:-1];comp=compile_graph(s);ref=NumpyTree(nominal)
    q,_,_=make_state(comp,nominal,ref,0.);graph=comp.graph(q);N,info=PACDM(graph).mapping(graph.lift(q))
    cases.append(dict(test='five_actuators_not_full_independent_chart',expected='numerical rank rejection',detected=not info['success'],success=not info['success'],message=info['message']))
    s=deepcopy(source)
    for j in s['joints']:
        if j['type'] in ('revolute','prismatic'):
            j['body_a'],j['body_b']=j['body_b'],j['body_a'];j['T_AJ'],j['T_BJ']=j['T_BJ'],j['T_AJ'];j['axis']=(-np.array(j['axis'])).tolist()
    c=compile_graph(s);q,old,W=make_state(c,nominal,ref,5.7);qzero,_,_=make_state(c,nominal,ref,0.);g=c.graph(qzero)
    check=physical_acceptance(g,NumpyTree(c.cmg),s,g.lift(q),q[g.active],W)
    cases.append(dict(test='reversed_edge_storage_preserves_physics',expected='valid model accepted',detected=None,success=bool(check['success'] and check['pose_error']<1e-10),message=str(check)))
    save_csv(out/'verification_tests.csv',cases)
    return dict(tests=len(cases),passed=sum(r['success'] for r in cases),fault_cases=13,
                note='Designed unit/precondition tests, not an estimated real-world fault detection probability')
