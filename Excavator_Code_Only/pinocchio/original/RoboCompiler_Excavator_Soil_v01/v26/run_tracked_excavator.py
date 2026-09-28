"""Reproduce or audit the articulated-track excavator release."""
from project import *
import argparse
import hashlib
import platform
import zipfile
from importlib.metadata import version
import numpy as np

CASES = {
    'coarse': dict(dt=.001),
    'nominal': dict(dt=.0005),
    'fine': dict(dt=.00025),
    'low_traction': dict(dt=.0005, traction=.55),
    'heavy_payload': dict(dt=.0005, density_scale=1.25),
    'no_drive': dict(dt=.0005, duration=6., soil=False, drive_enabled=False),
    'no_bucket_contact': dict(dt=.0005, duration=16., bucket_contact=False),
}

def installed_version(name):
    try:return version(name)
    except Exception:
        if name=='pin':
            import pinocchio
            return pinocchio.__version__
        raise

def source_integrity():
    hashes=json.loads((SOURCE/'SHA256SUMS.json').read_text())
    mismatches=[]
    for rel,expected in hashes.items():
        p=SOURCE/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            mismatches.append(rel)
    return dict(passed=not mismatches,files=len(hashes),mismatches=mismatches)

def recorded_hashes_match(base, hashes):
    """Require nonempty, project-contained provenance and current file hashes."""
    if not isinstance(hashes,dict) or not hashes:
        return False
    base=base.resolve()
    for relative,expected in hashes.items():
        if not isinstance(relative,str) or not isinstance(expected,str):return False
        candidate=base/relative
        if Path(relative).is_absolute() or not candidate.resolve().is_relative_to(base):return False
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest()!=expected:return False
    return True

def reproduce_native_parity():
    """Regenerate the three registered stepping fixtures with native MuJoCo."""
    import subprocess
    fixtures=[
        ('loaded_machine_current','results/assets/coarse.xml',
         'outputs/native_step_validation/loaded_machine_current/loaded_checkpoint.npz'),
        ('tracks_exact','outputs/track_compiler_exact/model.xml',None),
        ('activation','outputs/native_step_validation/activation/model.xml',None),
    ]
    for label,xml,checkpoint in fixtures:
        command=[sys.executable,'native_step.py','--xml',xml,
                 '--output','outputs/native_step_validation/'+label]
        if checkpoint:command+=['--checkpoint',checkpoint]
        subprocess.run(command,cwd=ROOT,check=True)

def evaluate_suite(names=None):
    names=list(CASES) if names is None else list(names)
    checks=[];reports={};missing=[]
    def gate(name,value,limit,case=None,scope=''):
        finite=bool(np.isfinite(value))
        checks.append(dict(name=name,value=float(value) if finite else None,limit=float(limit),
            passed=bool(finite and value<=limit),case=case,scope=scope))
    integrity=source_integrity()
    gate('Original v21 source files remain byte-identical',len(integrity['mismatches']),0,scope='Source provenance')
    for filename,label in [('track_drive_component_validation.json','Hydraulic travel drives'),
                           ('tracked_arm_component_validation.json','Online PACDM floating arm'),
                           ('travel_servo_validation.json','Low-inertia travel speed regulation')]:
        p=RESULTS/filename
        if p.exists():
            item=json.loads(p.read_text())
            gate(label+' component suite passes',0 if item.get('passed') else 1,0,scope='Component verification')
            if filename=='travel_servo_validation.json':
                gate(label+' source hashes are fresh',0 if recorded_hashes_match(ROOT,item.get('source_sha256')) else 1,0,scope='Component provenance')
            elif filename=='tracked_arm_component_validation.json':
                gate(label+' source hashes are fresh',0 if recorded_hashes_match(SOURCE,item.get('source_sha256')) else 1,0,scope='Component provenance')
                gate(label+' controller hash is fresh',0 if recorded_hashes_match(ROOT,{'tracked_arm.py':item.get('controller_sha256')}) else 1,0,scope='Component provenance')
        else:missing.append(str(p.relative_to(ROOT)))
    p=ROOT/'outputs/track_validated/assessment.json'
    if p.exists():
        item=json.loads(p.read_text())
        track_passed=item.get('status')=='PASS' and all(g['passed'] for g in item['gates'])
        gate('Explicit track component suite passes',0 if track_passed else 1,0,scope='Isolated full-mass ballast test')
    else:missing.append(str(p.relative_to(ROOT)))
    for label in ['loaded_machine_current','tracks_exact','activation']:
        p=ROOT/'outputs/native_step_validation'/label/'report.json'
        if p.exists():
            item=json.loads(p.read_text())
            gate('Native integrator parity '+label,0 if item.get('status')=='PASS' else 1,0,scope='Native stepping equivalence')
            gate('Native integrator parity hashes are fresh '+label,0 if recorded_hashes_match(ROOT,item.get('input_sha256')) else 1,0,scope='Component provenance')
        else:missing.append(str(p.relative_to(ROOT)))
    p=ROOT/'outputs/native_solver_accuracy/report.json'
    if p.exists():
        item=json.loads(p.read_text())
        gate('Residual-controlled native solver component',0 if item.get('passed') is True else 1,0,scope='Native solve accuracy; no mechanical state correction')
        gate('Native solver component input hashes are fresh',0 if recorded_hashes_match(ROOT,item.get('input_sha256')) else 1,0,scope='Component provenance')
    else:missing.append(str(p.relative_to(ROOT)))
    for name in names:
        p=RESULTS/f'{name}.json'
        if not p.exists():missing.append(str(p.relative_to(ROOT)));continue
        r=json.loads(p.read_text());reports[name]=r
        from mobile_mission import BOUNDARIES
        gate('Registered mission duration',abs(r['duration_s']-CASES[name].get('duration',float(BOUNDARIES[-1]))),1e-9,name)
        trajectory_path=RESULTS/f'{name}.npz'
        trajectory_ok=False
        try:
            with zipfile.ZipFile(trajectory_path) as archive:trajectory_ok=archive.testzip() is None
            with np.load(trajectory_path) as trajectory:
                times=trajectory['time']
                trajectory_ok=trajectory_ok and len(times)>1 and abs(times[0])<1e-12 and abs(times[-1]-r['duration_s'])<1e-9
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):pass
        gate('Complete CRC-checked trajectory',0 if trajectory_ok else 1,0,name)
        gate('Completed requested native integration',0 if r['completed'] else 1,0,name)
        gate('Native warnings',r['warning_count'],0,name)
        ns=r.get('numerical_solver',{})
        gate('Recorded native force-accuracy policy',0 if ns.get('policy')=='residual_controlled_native_warmstart_retry' else 1,0,name)
        gate('Accepted native solve target',ns.get('maximum_final_residual',float('inf')),1e-8,name)
        for key,limit in [('arm_gap',1e-4),('track_gap',1e-3),('tracking',.2),('force_balance',1e-6),
                          ('port_power_identity',1e-6),('arm_fluid_identity',1e-6),('drive_fluid_identity',1e-6),('pressure_bound',1e-6)]:
            gate(key,r['peak'][key],limit,name)
        for key in ['native_state_projection','base_propulsion_wrench','q22_actuated']:
            gate(key,int(r[key]),0,name)
        ctl=r['arm_controller']
        # Use the controller's independently registered diagnostic checks when
        # present; all-row closure is additionally tested by component replay.
        for key in ['pacdm_closure_max','pacdm_tangent_max','pacdm_point_tangent_difference_max']:
            if key in ctl:gate('PACDM '+key,ctl[key],1e-8,name)
        for key,limit in [('mass_reconstruction_error_kg',1e-9),('com_reconstruction_error_m',1e-10),('inertia_reconstruction_error_kg_m2',1e-8)]:
            gate('Undercarriage '+key,r['scene']['tracks'][key],limit,name)
        ap=RESULTS/f'{name}_audit.json'
        if ap.exists():
            audits=json.loads(ap.read_text())
            from tracked_audit import AUDIT_REVISION
            gate('Complete native global-wrench audit revision',0 if audits.get('revision')==AUDIT_REVISION else 1,0,name,
                 'Finite-gap constraint effects accounted; raw ideal-closure residuals retained')
            gate('Fresh dynamics audit inputs',0 if recorded_hashes_match(ROOT,audits.get('input_sha256')) else 1,0,name,
                 'Trace, model, oracle, native solver, replay code and preserved original metadata')
            gate('Audit replay leaves physical trace unchanged',0 if audits.get('physical_trace_unchanged') is True else 1,0,name)
            for snap in audits['snapshots']:
                for metric,value in snap['metrics'].items():
                    gate(f"t={snap['time_s']:g}s {metric}",value,snap['limits'][metric],name,'Independent Pinocchio/native/contact audit')
            gate('Rigid-frame covariance',0 if audits['frame_covariance']['passed'] else 1,0,name)
        else:missing.append(str(ap.relative_to(ROOT)))
        full=name not in ['no_drive','no_bucket_contact']
        if full:
            gate('Mechanical energy balance',r['mechanical_balance_relative'],.005 if name=='fine' else .01,name)
            gate('Total energy balance (gross activity normalization)',r['total_balance_relative'],.005,name)
            gate('Terminal position',r['final_position_error_m'],.15,name)
            gate('Terminal heading',r['final_heading_error_rad'],.1,name)
            gate('At least one metre of loaded base travel',1-r['loaded_base_displacement_m'],0,name)
            grain_mass=1800.*r['scene']['density_scale']*4*np.pi*.09**3/3
            gate('At least three particles carried',3-r['carried_mass_kg']/grain_mass,0,name)
            gate('At least two carried particles deposited',2-r['delivered_mass_kg']/grain_mass,0,name)
            task_path=RESULTS/f'{name}_task_audit.json'
            if task_path.exists():
                task=json.loads(task_path.read_text())
                gate('Independent particle-identity transport audit',0 if task.get('passed') else 1,0,name)
                for key,path in [('input_trace_sha256',RESULTS/f'{name}.npz'),('input_model_sha256',RESULTS/'assets'/f'{name}.xml'),('input_report_sha256',RESULTS/f'{name}.json')]:
                    current=hashlib.sha256(path.read_bytes()).hexdigest()
                    gate('Fresh task audit '+key,0 if task.get(key)==current else 1,0,name)
            else:missing.append(str(task_path.relative_to(ROOT)))
        elif name=='no_drive':
            traveled=np.linalg.norm(np.asarray(r['final_pose'])[:2]-np.asarray(r['initial_pose'])[:2])
            gate('Disconnected drive does not complete 0.5m approach',traveled,.10,name)
            gate('Disconnected drive applies zero torque',r['peak']['drive_torque'],0,name)
        elif name=='no_bucket_contact':
            gate('Disabled bucket captures no material',r['carried_mass_kg'],1e-9,name)
        if not full:
            task_path=RESULTS/f'{name}_task_audit.json'
            if task_path.exists():
                task=json.loads(task_path.read_text())
                gate('Negative-control body and energy accounting audit',0 if task.get('technical_checks_passed') else 1,0,name)
                for key,path in [('input_trace_sha256',RESULTS/f'{name}.npz'),('input_model_sha256',RESULTS/'assets'/f'{name}.xml'),('input_report_sha256',RESULTS/f'{name}.json')]:
                    current=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                    gate('Fresh task audit '+key,0 if current is not None and task.get(key)==current else 1,0,name)
            else:missing.append(str(task_path.relative_to(ROOT)))
        if task_path.exists():
            losses=task.get('energy_decomposition',{}).get('physical_losses',{})
            gate('Nonnegative accumulated physical losses',0 if losses.get('all_accumulated_nonnegative') is True else 1,0,name,
                 'Recorded physical loss work terms each >= -1e-7 J; no numerical loss relabelling')
            gate('Nonnegative sampled physical losses',0 if losses.get('all_sampled_nonnegative') is True else 1,0,name,
                 'Saved physical loss powers each >= -1e-7 W; samples are 0.02s apart')
    if all(k in reports for k in ['coarse','nominal','fine']):
        gate('Fine absolute mechanical defect improves over coarse',abs(reports['fine']['mechanical_balance_defect_J'])/max(1.,abs(reports['coarse']['mechanical_balance_defect_J'])),1.)
        gate('Fine absolute combined energy defect improves over coarse',abs(reports['fine']['total_balance_defect_J'])/max(1.,abs(reports['coarse']['total_balance_defect_J'])),1.)
        task_paths=[RESULTS/f'{name}_task_audit.json' for name in ['coarse','fine']]
        if all(path.exists() for path in task_paths):
            energy=[json.loads(path.read_text()).get('energy_decomposition',{}) for path in task_paths]
            if all(item.get('available') for item in energy):
                defects=[item['arm_fluid']['absolute_balance_defect_J'] for item in energy]
                gate('Fine absolute arm fluid integration defect improves over coarse',defects[1]/max(1.,defects[0]),1.)
    gate('All required evidence files exist',len(missing),0)
    result=dict(passed=all(x['passed'] for x in checks),passed_count=sum(x['passed'] for x in checks),
                check_count=len(checks),checks=checks,cases=names,missing=missing,source_integrity=integrity,
                versions={k:installed_version(k) for k in ['numpy','scipy','mujoco','pin','jsonschema']},python=sys.version,platform=platform.platform(),
                scope='Simulation verification of explicit tracks with engineering assumptions. PACDM closes the source arm; MuJoCo enforces new belt/gear constraints. No hardware/soil calibration or certified real-time claim.')
    save_json(RESULTS/'validation.json',result)
    print(('PASS' if result['passed'] else 'FAIL'),f"{result['passed_count']}/{result['check_count']} gates")
    for item in checks:
        if not item['passed']:print('FAILED',item)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=list(CASES));p.add_argument('--audit-existing',action='store_true')
    p.add_argument('--components-only',action='store_true');p.add_argument('--render',action='store_true')
    args=p.parse_args()
    if args.case:
        from mobile_simulation import run_case
        run_case(args.case,**CASES[args.case]);return
    if not args.audit_existing:
        import subprocess
        subprocess.run([sys.executable,'track_drive.py','--validate',str(RESULTS/'track_drive_component_validation.json')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'tracked_arm_validation.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'travel_servo_validation.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'native_solve.py','--model','outputs/force_residual_current/model.xml',
                        '--checkpoint','outputs/force_residual_current/checkpoint.npz',
                        '--trace','outputs/native_solver_accuracy/validation_states.npz',
                        '--output','outputs/native_solver_accuracy'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'track_smoke.py','--assess','--output','outputs/track_validated'],cwd=ROOT,check=True)
        if args.components_only:
            # The complete package bundles the coarse model for this mode.
            reproduce_native_parity()
            return
        from mobile_simulation import run_case
        for name,config in CASES.items():run_case(name,**config)
        # Full reproduction uses its newly exported coarse plant; component-only
        # reproduction above uses the bundled model with the same frozen plant.
        reproduce_native_parity()
        # Recreate the complete-ledger audit and its portable provenance before
        # task reports bind their input hash to final audit metadata.
        for name in CASES:
            subprocess.run([sys.executable,'replay_audits.py',name],cwd=ROOT,check=True)
        # Detailed task/body replay is separate so it can be repeated without
        # rerunning the expensive native trajectories.
        from tracked_task_audit import audit_task
        for name in CASES:audit_task(name)
    result=evaluate_suite()
    from tracked_report import make_report
    make_report(list(CASES))
    if args.render:
        from tracked_video import render_video
        render_video('nominal')
    if not result['passed']:sys.exit(1)

if __name__=='__main__':main()
