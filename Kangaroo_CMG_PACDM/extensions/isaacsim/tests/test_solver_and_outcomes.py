"""Regression of observed 23.0.2 failure and 23.0.3 control flow.

No fixture supplies native physics, and no historical trace is a new native run.
"""
from dataclasses import asdict,replace
from pathlib import Path
import copy,json,subprocess,sys
import numpy as np
import pytest
from kangaroo_isaac.control import Config,case_config,SOLVER_PROFILES
from kangaroo_isaac.gates import assess
from kangaroo_isaac.model import ROOT
from kangaroo_isaac.runtime_reporting import authoritative_exit_code
from kangaroo_isaac.report import convergence_check,suite_report
from kangaroo_isaac.solver_settings import author_solver_settings,read_solver_settings,app_configuration,expected_settings
from kangaroo_isaac.timing import StepTiming

FIXTURE=ROOT/'tests/fixtures/uploaded_v23_0_2'


def uploaded():return json.loads((FIXTURE/'result.json').read_text())


def test_real_uploaded_failed_task_cannot_exit_successfully():
    r=uploaded()
    assert r['completed'] is True
    assert r['validation']['functional_status']=='FUNCTIONAL_GATES_FAILED'
    assert authoritative_exit_code(r,0)==3
    failed={g['name'] for g in r['validation']['gates'] if g['status']!='PASS'}
    assert failed=={'maximum_final_base_speed_m_s','maximum_foot_drift_after_landing_m'}


def test_uploaded_metrics_still_fail_identical_acceptance_after_repair():
    r=uploaded();c=case_config('nominal',solver_profile='legacy_128_32')
    v=assess('nominal',c,r['metrics'],{'status':'PASS'},True)
    assert v['gates']==r['validation']['gates']
    assert v['functional_status']=='FUNCTIONAL_GATES_FAILED'
    assert v['unavailable_source_gates']==r['validation']['unavailable_source_gates']


def test_recorded_vertical_oscillation_is_visible_in_positions():
    with np.load(FIXTURE/'observations.npz',allow_pickle=False) as a:
        t=a['time'];m=t>=9.5
        dzdt=np.gradient(a['base'][:,2],t)
        assert np.corrcoef(dzdt[m][1:-1],a['base_velocity_origin'][m,2][1:-1])[0,1]>.99
        speed=np.linalg.norm(a['base_velocity_origin'][m],axis=1)
        assert speed.max()>.08
        assert speed.max()<=uploaded()['metrics']['maximum_final_base_speed_m_s']
        assert np.ptp(a['base'][m,2])>.004


@pytest.mark.parametrize('profile',list(SOLVER_PROFILES))
def test_profiles_change_only_declared_numerical_settings(profile):
    source=uploaded()['configuration'];c=asdict(case_config('nominal',solver_profile=profile))
    numerical={'solver_position_iterations','solver_velocity_iterations','solver_type',
               'external_forces_every_iteration','solver_profile','physx_threads'}
    assert all(c[k]==v for k,v in source.items() if k not in numerical)
    assert c['dt_s']==.000025 and c['duration_s']==10.


def test_source_and_nominal_refinement_are_exact_and_full_duration():
    a=case_config('nominal');b=case_config('refined');c=case_config('solver_check')
    assert b.dt_s==a.dt_s/2 and c.dt_s==a.dt_s
    assert c.solver_position_iterations==2*a.solver_position_iterations
    assert all(x.duration_s==10. for x in (a,b,c))


@pytest.mark.parametrize('bad',[dict(solver_velocity_iterations=-1),dict(solver_position_iterations=0),
    dict(solver_velocity_iterations=True),dict(physx_threads=-1),dict(physx_threads=True),
    dict(solver_type='newton'),dict(external_forces_every_iteration=1),dict(solver_profile='invented')])
def test_invalid_numerical_settings_fail_closed(bad):
    with pytest.raises(ValueError):replace(Config(),**bad).validate()


def test_zero_velocity_iterations_are_valid_not_silently_clamped():
    c=case_config('nominal',solver_profile='tgs_64_0')
    assert c.solver_velocity_iterations==0
    assert expected_settings(c)['MinVelocityIterationCount']==0
    assert expected_settings(c)['MaxVelocityIterationCount']==0


class Attribute:
    def __init__(self,value):self.value=value
    def Get(self):return self.value


class SceneSchemaFixture:
    """Explicitly a method-contract fixture, not a USD/PhysX installation."""
    def __init__(self):self.attributes={}
    def __getattr__(self,key):
        if key.startswith('Create') and key.endswith('Attr'):
            def setter(value):
                self.attributes[key[6:-4]]=Attribute(value)
                return self.attributes[key[6:-4]]
            return setter
        if key.startswith('Get') and key.endswith('Attr'):
            return lambda:self.attributes[key[3:-4]]
        raise AttributeError(key)


@pytest.mark.parametrize('profile',list(SOLVER_PROFILES))
def test_all_solver_settings_authored_and_read_back(profile):
    c=case_config('nominal',solver_profile=profile);scene=SceneSchemaFixture()
    r=author_solver_settings(scene,c)
    assert r['actual']==expected_settings(c) and r['status']=='PASS'
    scene.attributes['MinVelocityIterationCount'].value+=1
    with pytest.raises(RuntimeError,match='readback mismatch'):read_solver_settings(scene,c)


def test_absent_schema_method_is_not_silent_default():
    with pytest.raises(RuntimeError,match='no solver fallback'):author_solver_settings(object(),Config())


@pytest.mark.parametrize('headless',[True,False])
def test_viewport_and_threads_do_not_change_step_clock(headless):
    a=app_configuration(Config(),headless)
    assert a['extra_args']==['--/persistent/physics/numThreads=0']
    assert a.get('disable_viewport_updates',False)==headless
    assert 'physics_dt' not in a


def passing_result(case='nominal'):
    # Synthetic metric fixture used only to exercise control flow.
    from test_control_and_reporting import good_metrics
    c=case_config(case);m=good_metrics(c)
    m.update(final_base_position_m=[0.,0.,.7],final_motor_work_J=[0.]*12)
    return {'engine':'Isaac Sim / PhysX','completed':True,'case':case,'configuration':asdict(c),
            'metrics':m,'validation':assess(case,c,m,{'status':'PASS'},True)}


@pytest.mark.parametrize('code',[0,3,2,124,130,-11])
def test_raw_process_failure_never_erased(code):
    assert authoritative_exit_code(passing_result(),code)==abs(code)


@pytest.mark.parametrize('defect',['missing_gates','failed_gate','failed_status','missing_status',
 'missing_engine','incomplete','physics_error','diagnostic'])
def test_inconsistent_success_status_rejected(defect):
    r=passing_result()
    if defect=='missing_gates':r['validation']['gates']=[]
    if defect=='failed_gate':r['validation']['gates'][0]['status']='FAIL'
    if defect=='failed_status':r['validation']['functional_status']='FUNCTIONAL_GATES_FAILED'
    if defect=='missing_status':r['validation'].pop('functional_status')
    if defect=='missing_engine':r.pop('engine')
    if defect=='incomplete':r['completed']=False
    if defect=='physics_error':r['physics_log']={'failures':['SDK failure']}
    if defect=='diagnostic':r['validation']['functional_status']='DIAGNOSTIC_ONLY_NOT_FULL_TASK'
    assert authoritative_exit_code(r,0)!=0


def test_convergence_requires_both_native_case_passes():
    assert convergence_check(uploaded(),passing_result('refined'))['status']=='BLOCKED_CASE_NOT_PASSED'
    assert convergence_check(None,passing_result())['status']=='NOT_RUN'


@pytest.mark.parametrize('varying,case',[('dt_s','refined'),('solver_position_iterations','solver_check')])
def test_guarded_convergence_uses_original_position_and_work_limits(varying,case):
    a=passing_result();b=passing_result(case)
    r=convergence_check(a,b,varying)
    assert r['status']=='KINEMATIC_AND_WORK_GATES_PASSED'
    assert r['position_threshold_m']==.005 and r['work_threshold_J']==1.
    b['metrics']['final_motor_work_J'][0]=1.01
    assert convergence_check(a,b,varying)['status']=='FAILED'
    b['metrics']['final_motor_work_J'][0]=0.
    b['metrics']['final_base_position_m'][0]=.00501
    assert convergence_check(a,b,varying)['status']=='FAILED'


@pytest.mark.parametrize('defect',['nan','missing','shape','friction','duration','not_halved','process_failure'])
def test_invalid_convergence_input_cannot_pass(defect):
    a=passing_result();b=passing_result('refined')
    if defect=='nan':b['metrics']['final_motor_work_J'][0]=float('nan')
    if defect=='missing':b['metrics'].pop('final_base_position_m')
    if defect=='shape':b['metrics']['final_base_position_m']=[0.]
    if defect=='friction':b['configuration']['friction']=1.
    if defect=='duration':b['configuration']['duration_s']=.1
    if defect=='not_halved':b['configuration']['dt_s']=a['configuration']['dt_s']
    if defect=='process_failure':b['supervisor']={'raw_process_exit_code':3}
    assert convergence_check(a,b)['status'].startswith('BLOCKED')


def test_actual_failure_is_not_obscured_by_skipped_refinement(tmp_path):
    p=tmp_path/'nominal';p.mkdir();(p/'result.json').write_text(json.dumps(uploaded()))
    r=suite_report(tmp_path,['nominal','refined','solver_check'])
    assert r['overall_status']=='NATIVE_FUNCTIONAL_GATES_FAILED_OR_DIAGNOSTIC_ONLY'
    assert r['refinement']['status']==r['solver_convergence']['status']=='NOT_RUN'
    assert not r['certified_ready']


@pytest.mark.parametrize('args',[['--verify-native','--duration','1'],['--verify-native','--worker'],
    ['--verify-native','--dt','.0001'],['--physx-threads','-1'],['--verify-native','--solver-profile','legacy_128_32']])
def test_invalid_launch_arguments_rejected_before_native_import(args):
    p=subprocess.run([sys.executable,str(ROOT/'run_isaac.py'),*args],capture_output=True,text=True)
    assert p.returncode==2 and 'error:' in p.stderr
    assert 'Simulation App Starting' not in p.stdout


def test_timing_is_measured_not_estimated():
    values=iter([1.,1.2,1.5,2.,2.4]);t=StepTiming(clock=lambda:next(values))
    t.begin();t.mark('a');t.mark('b');t.begin();t.mark('a')
    r=t.report()
    assert r['seconds']['a']==pytest.approx(.6) and r['calls']['a']==2
    assert r['seconds']['b']==pytest.approx(.3)
    assert not r['native_speedup_claim']


def supervisor_fixture(monkeypatch,tmp_path,failed_case=None,work_offset=0.):
    """Exercise the actual supervisor with real subprocesses but NO physics.

    Child processes write clearly synthetic metric fixtures. SDK startup,
    reference audit and case telemetry rendering are stubbed explicitly here.
    """
    import importlib.util
    spec=importlib.util.spec_from_file_location('runner_under_test',ROOT/'run_isaac.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr('kangaroo_isaac.preflight.inspect_environment',lambda:{'status':'READY_FOR_NATIVE_ATTEMPT','errors':[],'origin':'SOFTWARE_FIXTURE'})
    monkeypatch.setattr('kangaroo_isaac.offline.audit_reference',lambda output:{'status':'PASS','origin':'SOFTWARE_FIXTURE'})
    monkeypatch.setattr('kangaroo_isaac.report.make_case_report',lambda output:None)
    original_popen=subprocess.Popen;commands=[]
    def child(cmd,**kwargs):
        commands.append(cmd)
        case=cmd[cmd.index('--case')+1];out=cmd[cmd.index('--output')+1]
        r=uploaded() if case==failed_case else passing_result(case)
        r['case']=case
        if case=='refined' and case!=failed_case:r['metrics']['final_motor_work_J'][0]=work_offset
        script="import pathlib,json; p=pathlib.Path(%r); (p/'result.json').write_text(%r); print('SOFTWARE SUBPROCESS FIXTURE -- NO ISAAC/PHYSX EXECUTION',flush=True)" % (out,json.dumps(r))
        return original_popen([sys.executable,'-c',script],**kwargs)
    monkeypatch.setattr(mod.subprocess,'Popen',child)
    monkeypatch.setattr(sys,'argv',['run_isaac.py','--verify-native','--headless','--no-visuals','--output',str(tmp_path)])
    return mod,commands


def test_actual_supervisor_fixes_observed_zero_exit_and_stops_on_failure(monkeypatch,tmp_path):
    mod,cmds=supervisor_fixture(monkeypatch,tmp_path,failed_case='nominal')
    assert mod.main()==3 and len(cmds)==1
    f=next(tmp_path.iterdir());r=json.loads((f/'suite_summary.json').read_text())
    assert r['subprocess_exit_codes']==[0] and r['effective_exit_codes']==[3]
    assert r['verification_status']=='NATIVE_VERIFICATION_FAILED_OR_INCOMPLETE'
    assert not (f/'refined').exists() and (f/'results_bundle.zip').is_file()
    assert json.loads((f/'nominal/result.json').read_text())['supervisor']['effective_exit_code']==3


def test_actual_supervisor_propagates_all_settings_and_verifies_three_cases(monkeypatch,tmp_path):
    mod,cmds=supervisor_fixture(monkeypatch,tmp_path)
    assert mod.main()==0 and len(cmds)==3
    assert [cmd[cmd.index('--case')+1] for cmd in cmds]==['nominal','refined','solver_check']
    for cmd in cmds:
        assert cmd[cmd.index('--solver-profile')+1]=='pgs_64_8'
        assert cmd[cmd.index('--physx-threads')+1]=='0'
        assert cmd[cmd.index('--dt')+1]=='2.5e-05'
    f=next(tmp_path.iterdir());r=json.loads((f/'suite_summary.json').read_text())
    assert r['verification_status']=='NATIVE_TASK_AND_CONVERGENCE_PASSED'
    assert r['certified_ready'] is False
    assert r['overall_status']=='BLOCKED_INCOMPLETE_REACTION_AUDIT'


def test_actual_supervisor_rejects_work_nonconvergence_even_when_all_cases_pass(monkeypatch,tmp_path):
    mod,cmds=supervisor_fixture(monkeypatch,tmp_path,work_offset=1.01)
    assert mod.main()==3
    f=next(tmp_path.iterdir());r=json.loads((f/'suite_summary.json').read_text())
    assert r['refinement']['status']=='FAILED'
    assert r['verification_status']=='NATIVE_VERIFICATION_FAILED_OR_INCOMPLETE'
    assert r['overall_status']=='NATIVE_CONVERGENCE_FAILED'
