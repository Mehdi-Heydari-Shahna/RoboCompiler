#!/usr/bin/env python
"""Miniforge entry point. No implicit package installation or license acceptance."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,uuid,zipfile
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
# Avoid 76x76/140x140 linear algebra spawning dozens of BLAS threads. Set before
# any NumPy/SciPy import. The user can explicitly set a different value.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')


def arguments():
    p=argparse.ArgumentParser(description='Kangaroo CMG/PACDM native Isaac Sim port candidate')
    from kangaroo_isaac.version import VERSION
    p.add_argument('--version',action='version',version='Kangaroo Isaac Sim '+VERSION)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--preflight',action='store_true',help='Inspect the ACTIVE environment, do not launch Isaac')
    g.add_argument('--offline-test',action='store_true',help='Run CPU tests and 401 PACDM sample audits, NOT Isaac')
    g.add_argument('--suite',action='store_true',help='Run the native task, robustness cases, ablation, and negative controls')
    g.add_argument('--verify-native',action='store_true',help='Full nominal, then dt/2 and doubled-iteration checks; stop at the first failure')
    g.add_argument('--case',default=None,choices=['nominal','refined','solver_check','low_friction','higher_drop_push','slow_actuators','no_feedforward','no_contact','no_loops','passive'])
    p.add_argument('--headless',action='store_true',help='Disable the GUI (not a GPU-free Isaac runtime)')
    p.add_argument('--no-visuals',action='store_true',help='Omit CAD meshes; retain all masses, joints, and sole collisions')
    from kangaroo_isaac.control import SOLVER_PROFILES
    p.add_argument('--solver-profile',choices=list(SOLVER_PROFILES),default='pgs_64_8')
    p.add_argument('--contact-mode',choices=['contact_envelope','force_threshold'],default='contact_envelope',
        help='Feedforward gate: measured sole/ground geometry (default), or legacy instantaneous force threshold')
    from kangaroo_isaac.contact_model import CONTACT_MODELS
    p.add_argument('--contact-model',choices=list(CONTACT_MODELS),default='source_compliance',
        help='Sole/floor contact law: source MuJoCo compliance as a PhysX compliant contact (default), or legacy rigid')
    p.add_argument('--physx-threads',type=int,default=0,help='CPU PhysX worker threads: 0 is synchronous main-thread execution')
    p.add_argument('--dt',type=float,default=.000025,help='Physics seconds, default source-faithful 25 microseconds')
    p.add_argument('--duration',type=float,default=None,help='Short diagnostic only; never eligible for full task validation')
    p.add_argument('--output',type=Path,default=ROOT/'results',help='Parent directory for a unique run folder')
    p.add_argument('--timeout',type=float,default=0.,help='Optional wall-clock subprocess timeout seconds; 0 disables')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    import math
    if not math.isfinite(a.dt) or a.dt<=0:p.error('--dt must be positive and finite')
    if not math.isfinite(a.timeout) or a.timeout<0:p.error('--timeout must be nonnegative and finite')
    if a.duration is not None and (not math.isfinite(a.duration) or not 0<a.duration<=10):p.error('--duration must be in (0,10]')
    if (a.suite or a.verify_native) and a.duration is not None:p.error('--duration cannot be used with --suite or --verify-native')
    if not 0<=a.physx_threads<=255:p.error('--physx-threads must be in [0,255]')
    if a.verify_native and a.worker:p.error('--verify-native cannot be a worker request')
    if a.verify_native and a.dt!=.000025:p.error('--verify-native preserves the 25-microsecond source timestep')
    if a.verify_native and SOLVER_PROFILES[a.solver_profile]['solver_position_iterations']>127:
        p.error('--verify-native needs an iteration profile that can be doubled without clamping')
    return a


def bundle(folder):
    folder=Path(folder);path=folder/'results_bundle.zip'
    # Exclude scene export (reproducible from source) and nested bundles.
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f!=path and f.suffix not in ('.usda','.usdc','.zip','.tmp'):
                z.write(f,f.relative_to(folder))
    return path


def main():
    args=arguments()
    from kangaroo_isaac.version import VERSION
    print(f'Kangaroo Isaac Sim {VERSION} | project: {ROOT}',flush=True)
    from kangaroo_isaac.preflight import inspect_environment
    from kangaroo_isaac.io_utils import write_json
    if args.worker:
        from dataclasses import replace
        from kangaroo_isaac.control import case_config
        from kangaroo_isaac.native import run
        cfg=case_config(args.case or 'nominal',args.dt,args.solver_profile,args.physx_threads,args.contact_mode,args.contact_model)
        if args.duration is not None:cfg=replace(cfg,duration_s=args.duration).validate()
        return run(args.case or 'nominal',cfg,args.output,args.headless,not args.no_visuals)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    folder=args.output.resolve()/run_id;folder.mkdir(parents=True,exist_ok=False)
    if args.preflight:
        info=inspect_environment();write_json(folder/'preflight.json',info)
        print(json.dumps(info,indent=2));print('\nSaved:',folder)
        print('Diagnostics:',bundle(folder))
        return 0 if info['status']=='READY_FOR_NATIVE_ATTEMPT' else 2
    if args.offline_test:
        from kangaroo_isaac.offline import audit_reference
        report=audit_reference(folder)
        command=[sys.executable,'-m','pytest',str(ROOT/'tests'),'-q','-p','no:cacheprovider','--junitxml='+str(folder/'pytest.xml')]
        # Run only pytest's built-in plugins and this project's tests.
        # Unrelated site-installed plugins must not change or hang the audit.
        test_env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,env=test_env)
        (folder/'pytest.log').write_text(test.stdout+'\n'+test.stderr,encoding='utf-8')
        print(test.stdout);print(test.stderr,file=sys.stderr)
        good=test.returncode==0 and report['status']=='PASS'
        write_json(folder/'offline_status.json',{'status':'PASS' if good else 'FAIL','pytest_exit_code':test.returncode,
            'isaac_sim_execution':'NOT_RUN','certified_ready':False})
        print('Offline results:',bundle(folder))
        return 0 if good else 1
    info=inspect_environment();write_json(folder/'preflight.json',info)
    if info['status']!='READY_FOR_NATIVE_ATTEMPT':
        for error in info['errors']:print('BLOCKED:',error)
        write_json(folder/'native_execution.json',{'status':'NOT_RUN','reason':'environment preflight failed','certified_ready':False})
        print('Diagnostics:',bundle(folder));return 2
    # Verify source PACDM/reference before starting the expensive native app.
    from kangaroo_isaac.offline import audit_reference
    audit=audit_reference(folder/'reference_audit')
    if audit['status']!='PASS':print('BLOCKED: PACDM/reference audit failed');return 2
    from kangaroo_isaac.control import CASES,case_config
    from kangaroo_isaac.report import make_case_report,suite_report
    from kangaroo_isaac.runtime_reporting import classify_physics_log,make_failure_report,authoritative_exit_code
    cases=('nominal','refined','solver_check') if args.verify_native else (CASES if args.suite else (args.case or 'nominal',))
    if args.suite and args.duration is not None:raise SystemExit('--duration cannot be used with --suite')
    codes=[];raw_codes=[]
    for case in cases:
        # Validate flags before launch and isolate every case in a fresh process.
        case_config(case,args.dt,args.solver_profile,args.physx_threads,args.contact_mode,args.contact_model)
        out=folder/case;out.mkdir()
        cmd=[sys.executable,str(ROOT/'run_isaac.py'),'--worker','--case',case,'--dt',str(args.dt),'--output',str(out),'--solver-profile',args.solver_profile,'--physx-threads',str(args.physx_threads),'--contact-mode',args.contact_mode,'--contact-model',args.contact_model]
        if args.headless:cmd.append('--headless')
        if args.no_visuals or args.headless:cmd.append('--no-visuals')
        if args.duration is not None:cmd+=['--duration',str(args.duration)]
        print('\nLaunching',case,'; log:',out/'console.log',flush=True)
        (out/'launch_command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        # Tee output in a bounded-memory streaming thread. Child remains a
        # foreground operation; this function does not return until it finishes.
        import threading
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              bufsize=0)
        def tee():
            import codecs
            decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
            with (out/'console.log').open('w',encoding='utf-8') as log:
                while True:
                    chunk=os.read(proc.stdout.fileno(),8192)
                    if not chunk:break
                    text=decoder.decode(chunk)
                    log.write(text);log.flush();print(text,end='',flush=True)
                tail=decoder.decode(b'',final=True)
                if tail:log.write(tail);print(tail,end='',flush=True)
        thread=threading.Thread(target=tee,daemon=True);thread.start()
        try:code=proc.wait(timeout=args.timeout if args.timeout>0 else None)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait();code=124
            write_json(out/'supervisor_error.json',{'status':'TIMEOUT','certified_ready':False})
        except KeyboardInterrupt:
            proc.terminate();proc.wait();code=130
            write_json(out/'supervisor_error.json',{'status':'INTERRUPTED','certified_ready':False})
        thread.join(timeout=10)
        raw_codes.append(code)
        result_path=out/'result.json'
        if not result_path.exists():
            write_json(out/'supervisor_result.json',{'status':'NOT_COMPLETED','exit_code':code,'certified_ready':False})
            code=code or 2
            make_failure_report(out,case,code)
        else:
            result=json.loads(result_path.read_text())
            # A shorter diagnostic must not pass as the declared 10-second task.
            if args.duration is not None:
                result['validation']['functional_status']='DIAGNOSTIC_ONLY_NOT_FULL_TASK'
                result['validation']['scope']='user-requested shortened diagnostic'
            log=(out/'console.log').read_text(encoding='utf-8',errors='replace')
            classified=classify_physics_log(log,
                high_tgs_iterations=result['configuration'].get('solver_type','TGS')=='TGS' and result['configuration']['solver_velocity_iterations']>4)
            result['physics_log']=classified
            result['physics_log_warnings']=classified['failures']
            # The exact version-change notice is retained for review; it is
            # not a tensor binding failure or convergence evidence. All other
            # current PhysX warnings/errors remain failure conditions.
            if classified['failures'] or code not in (0,3):
                result['validation']['functional_status']='NATIVE_RUNTIME_OR_PHYSICS_LOG_FAILURE'
                code=code or 3
            code=authoritative_exit_code(result,code)
            result['supervisor']={'raw_process_exit_code':raw_codes[-1],'effective_exit_code':code,
                'policy':'Native completion, full-task gate values and status are authoritative; raw exit 0 alone is insufficient.'}
            write_json(result_path,result)
            try:make_case_report(out)
            except Exception as e:
                write_json(out/'report_error.json',{'status':'ERROR','error':str(e),'certified_ready':False})
                code=code or 4
        codes.append(code)
        if args.verify_native and code!=0:break
        if code in (2,124,130):break  # Do not repeat app/environment failures nine times.
    summary=suite_report(folder,cases)
    case_codes=codes.copy()
    if (args.suite or args.verify_native) and summary['refinement']['status']!='KINEMATIC_AND_WORK_GATES_PASSED':codes.append(3)
    if args.verify_native and summary['solver_convergence']['status']!='KINEMATIC_AND_WORK_GATES_PASSED':codes.append(3)
    summary['subprocess_exit_codes']=raw_codes
    summary['effective_exit_codes']=case_codes
    summary['convergence_exit_codes']=codes[len(case_codes):]
    summary['requested_verification']=bool(args.verify_native)
    summary['verification_status']=('NATIVE_TASK_AND_CONVERGENCE_PASSED' if codes and all(x==0 for x in codes)
        else 'NATIVE_VERIFICATION_FAILED_OR_INCOMPLETE') if args.verify_native else 'NOT_REQUESTED'
    if args.verify_native:
        print('Verification:',summary['verification_status'],flush=True)
    write_json(folder/'suite_summary.json',summary)
    print('\nResults bundle:',bundle(folder))
    print('Report:',folder/'REPORT.md')
    print('Native outcomes and incomplete reaction/energy checks are reported separately; no full certification is claimed.')
    return max([abs(x) for x in codes] or [2])

if __name__=='__main__':
    raise SystemExit(main())
