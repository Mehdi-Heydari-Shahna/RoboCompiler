import ast,json,subprocess,sys,xml.etree.ElementTree as ET
from dataclasses import replace
import numpy as np
import pytest
from kangaroo_isaac.model import ROOT,finite,axis_frame
from kangaroo_isaac.control import Config,Controller,case_config,CASES,external_push
from kangaroo_isaac.gates import assess,check,SOURCE,UNAVAILABLE
from kangaroo_isaac.io_utils import name_map,sha256,write_json
from kangaroo_isaac.report import compare_supplied_mujoco,suite_report
from kangaroo_isaac.observations import Recorder


@pytest.mark.parametrize('name',CASES)
def test_case_definitions(name):
    cfg=case_config(name)
    assert cfg.validate() is cfg
    assert cfg.dt_s==(.0000125 if name=='refined' else .000025)
    assert cfg.duration_s==(.3 if name in ('no_contact','no_loops') else 1. if name=='passive' else 10.)


@pytest.mark.parametrize('change',[{'dt_s':0},{'dt_s':.000027},{'duration_s':11},{'friction':-1},
    {'actuator_time_constant_s':0},{'drop_height_m':-1},{'contact_offset_m':0},
    {'kp_N_m':-1},{'command_slew_N_s':-1},{'render_period_s':0},
    {'solver_position_iterations':256},{'solver_velocity_iterations':1.5},{'dt_s':float('nan')}])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):replace(Config(),**change).validate()


def test_causal_filter_and_1000hz_command_hold(model,ref):
    cfg=Config();c=Controller(model,ref,cfg)
    assert np.all(c.force==0)
    q=ref.u(0)-.1;qd=ref.u(0,1)
    cmd=c.update_command(0,q,qd,False)
    np.testing.assert_allclose(cmd,np.full(12,1500.))
    assert np.all(c.force==0) # first native step really receives zero filter state
    for k in range(1,40):
        c.advance_filter();before=c.command.copy()
        c.update_command(k,q+100,qd,True)
        np.testing.assert_array_equal(c.command,before)
    c.advance_filter()
    np.testing.assert_allclose(c.force,cmd*(1-np.exp(-.001/.002)),atol=2e-12)
    assert c.update_count==1


@pytest.mark.parametrize('sign',[-1,1])
def test_per_motor_saturation_and_slew(model,ref,sign):
    c=Controller(model,ref,Config())
    for k in range(240):
        prev=c.command.copy()
        c.update_command(k,ref.u(0)-sign*10,ref.u(0,1),True)
        assert np.max(abs(c.command-prev))<=1500+1e-10
        c.advance_filter()
        assert np.all(c.force>=model.bounds[:,0]) and np.all(c.force<=model.bounds[:,1])
    np.testing.assert_allclose(c.command,model.bounds[:,1]*sign,atol=1e-8)
    assert c.saturation_updates>0


def test_actual_contact_gate_and_no_feedforward_ablation(model,ref):
    cfg=replace(Config(),command_slew_N_s=1e12)
    a=Controller(model,ref,cfg);b=Controller(model,ref,cfg)
    q=ref.u(0);qd=ref.u(0,1)
    a.update_command(0,q,qd,False);b.update_command(0,q,qd,True)
    np.testing.assert_allclose(a.last_raw,0,atol=1e-12)
    np.testing.assert_allclose(b.last_raw,ref.force(0),atol=1e-10)
    c=Controller(model,ref,replace(cfg,no_feedforward=True))
    c.update_command(0,q,qd,True);np.testing.assert_allclose(c.last_raw,0,atol=1e-12)


def test_passive_has_zero_control(model,ref):
    c=Controller(model,ref,case_config('passive'))
    c.update_command(0,np.zeros(12),np.ones(12)*50,True)
    np.testing.assert_array_equal(c.advance_filter(),np.zeros(12))


def test_disturbance_only_declared_interval_and_axis():
    c=Config()
    np.testing.assert_array_equal(external_push(0,c),np.zeros(3))
    np.testing.assert_array_equal(external_push(7,c),np.zeros(3))
    np.testing.assert_allclose(external_push(6.825,c),[0,50,0],atol=1e-10)


@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),-.1])
def test_failed_or_missing_measurement_cannot_pass(value):
    result=check('margin',value,0.,'>=')
    assert result['status']!='PASS'


def good_metrics(cfg):
    m={k:0. for k in SOURCE['positive_trials'] if k not in UNAVAILABLE}
    m.update(observed_duration_s=cfg.duration_s,steps=round(cfg.duration_s/cfg.dt_s),
        force_limit_violation=False,achieved_crouch_m=.12,achieved_lateral_excursion_m=.09,
        minimum_slide_margin_m=.005,minimum_hinge_margin_rad=.005,
        final_weight_error_N=0.,maximum_foot_drift_after_landing_m=[0.,0.])
    return m


def test_functional_pass_cannot_become_full_certification():
    c=Config();a=assess('nominal',c,good_metrics(c),{'status':'PASS'},True)
    assert a['functional_status']=='FUNCTIONAL_GATES_PASSED'
    assert not a['certified_ready'] and a['full_source_validation_status'].startswith('BLOCKED')
    assert len(a['unavailable_source_gates'])==3


@pytest.mark.parametrize('defect',['incomplete','steps','missing','limit','initialization','nan'])
def test_invalid_native_result_fails(defect):
    c=Config();m=good_metrics(c);completed=True;audit={'status':'PASS'}
    if defect=='incomplete':completed=False
    if defect=='steps':m['steps']-=1
    if defect=='missing':m.pop('maximum_loop_gap_m')
    if defect=='limit':m['force_limit_violation']=True
    if defect=='initialization':audit={'status':'NOT_RUN'}
    if defect=='nan':m['final_tilt_deg']=float('nan')
    assert assess('nominal',c,m,audit,completed)['functional_status']=='FUNCTIONAL_GATES_FAILED'


@pytest.mark.parametrize('case', ['no_contact','no_loops','passive'])
def test_negative_control_is_not_positive_validation(case):
    c=case_config(case);m=good_metrics(c);m.update(base_fall_m=.2,maximum_loop_gap_m=.03)
    a=assess(case,c,m,{'status':'PASS'},True)
    assert a['functional_status']=='EXPECTED_FAILURE_OBSERVED'
    assert a['full_source_validation_status']=='NOT_APPLICABLE' and not a['certified_ready']
    m.update(base_fall_m=0.,maximum_loop_gap_m=0.)
    assert assess(case,c,m,{'status':'PASS'},True)['functional_status']=='NEGATIVE_CONTROL_FAILED'


def test_name_mapping_uses_ids_not_native_order(model):
    native=list(reversed(model.ids));idx=name_map(model.ids,native)
    assert [native[i] for i in idx]==model.ids
    with pytest.raises(RuntimeError):name_map(model.ids,native[:-1])
    with pytest.raises(RuntimeError):name_map(model.ids,native[:-1]+[native[0]])


def test_source_and_core_hashes():
    p=json.loads((ROOT/'data/port_provenance.json').read_text())
    for f,digest in p['source_files'].items():assert sha256(ROOT/f)==digest,f
    assert sha256(ROOT/'kangaroo_isaac/pacdm.py')==p['preserved_pacdm_sha256']
    assert (ROOT/'kangaroo_isaac/source_dynamics.py').read_bytes()==(ROOT/'vendor_v22/source_dynamics.py').read_bytes()


def test_atomic_json_and_nonfinite_rejection(tmp_path):
    f=tmp_path/'nested/a.json';write_json(f,{'a':np.arange(3),'b':np.float64(.4)})
    assert json.loads(f.read_text())=={'a':[0,1,2],'b':.4}
    assert not f.with_suffix('.json.tmp').exists()
    with pytest.raises(ValueError):write_json(f,{'x':float('nan')})
    assert 'a' in json.loads(f.read_text())


def test_cross_engine_coordinate_reordering_is_exact(model):
    # Construct a TEST FIXTURE from the recorded baseline, NOT an Isaac result.
    with np.load(ROOT/'reference_mujoco/landing_nominal.npz',allow_pickle=False) as r:a={k:r[k] for k in r.files}
    tree=ET.parse(ROOT/'reference_mujoco/landing_nominal.xml').getroot()
    names=[x.attrib['name'] for x in tree.find('worldbody').iter('joint')]
    mi=[names.index(n) for n in model.ids]
    acts=[x.attrib['joint'] for x in tree.find('actuator')]
    ai=[acts.index(n) for n in model.c['independent_ids']]
    pose=np.zeros((len(a['time']),78,7));pose[:,:,6]=1
    pose[:,model.root,3:]=a['q'][:,3:7][:,[1,2,3,0]]
    mock=dict(time=a['time'],base=a['q'][:,:3],q=a['q'][:,7:][:,mi],motor_force=a['act'][:,ai],
        motor_work=a['motor_work'],body_pose_xyzw=pose)
    v=compare_supplied_mujoco('nominal',mock)
    assert not v['equivalence_claim']
    for k,value in v.items():
        if k.endswith(('_m','_rad','_N','_J','_deg')):assert abs(value)<1e-12,(k,value)


def test_empty_suite_cannot_pass(tmp_path):
    r=suite_report(tmp_path,['nominal','refined'])
    assert not r['certified_ready']
    assert all(x['functional_status']=='NOT_COMPLETED' for x in r['cases'])
    assert r['refinement']['status']=='NOT_RUN'


def test_recorder_work_is_measured_subtotal_not_full_ledger(model,ref):
    cfg=Config();rec=Recorder(model,ref,cfg)
    q=ref.data['q'][0];P=model.fk(q,ref.data['base'][0]);V=np.zeros((78,6));v=np.zeros(76)
    pose=np.zeros((78,7));pose[:,6]=1
    rec.observe(0,P,V,q,v,np.zeros(12),np.zeros(12),np.zeros((2,3)),np.zeros(3),pose,save=True)
    force=np.ones(12)*100;v[model.active]=.1
    rec.integrate_step(force,v,v,np.zeros(3),V,V)
    np.testing.assert_allclose(rec.motor_work,100*.1*cfg.dt_s,atol=1e-15)
    assert rec.passive_work<0
    assert rec.summarize()['complete_energy_ledger_available'] is False


@pytest.mark.parametrize('version',[(3,11),(3,12)])
def test_all_new_python_files_parse_for_target_versions(version):
    paths=list((ROOT/'kangaroo_isaac').glob('*.py'))+[ROOT/'run_isaac.py']
    for p in paths:ast.parse(p.read_text(encoding='utf-8'),filename=str(p),feature_version=version)


def test_cli_help_without_isaac_import():
    r=subprocess.run([sys.executable,str(ROOT/'run_isaac.py'),'--help'],capture_output=True,text=True)
    assert r.returncode==0 and '--offline-test' in r.stdout


def test_preflight_only_path_has_no_numpy_requirement():
    code='import sys;sys.path.insert(0,sys.argv[1]);import kangaroo_isaac.preflight;import kangaroo_isaac.io_utils;assert "numpy" not in sys.modules'
    r=subprocess.run([sys.executable,'-S','-c',code,str(ROOT)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


@pytest.mark.parametrize('flags',[['--dt','0'],['--timeout','-1'],['--duration','0'],['--suite','--duration','1']])
def test_bad_cli_options_fail_without_native_launch(flags):
    r=subprocess.run([sys.executable,str(ROOT/'run_isaac.py'),*flags],capture_output=True,text=True)
    assert r.returncode==2 and 'error:' in r.stderr
