"""Contact-logic regressions; recorded 23.0.3 replay is NOT a new PhysX run."""
from dataclasses import replace,asdict
from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
import pytest
from kangaroo_isaac.control import Config,Controller,case_config
from kangaroo_isaac.contact_gate import ContactGate,ControllerTrace
from kangaroo_isaac.scene_paths import GROUND_CENTER_M,GROUND_SIZE_M
from kangaroo_isaac.runtime_reporting import authoritative_exit_code
from kangaroo_isaac.gates import assess
from kangaroo_isaac.model import ROOT

FIX=Path(__file__).parent/'fixtures/native_23_0_3_contact'

def sole_points(left=0.,right=0.,x=0.):
    corners=np.array([[a,b,c] for a in (-.1,.1) for b in (-.05,.05) for c in (0.,.02)])
    return np.array([corners+[x,-.15,left],corners+[x,.15,right]])

def load(total=0.):
    f=np.zeros((2,3));f[:,2]=total/2;return f

@pytest.mark.parametrize('h',[-1e-6,0.,2.3e-6,.001,.002])
def test_zero_force_within_existing_envelope_keeps_contact(h):
    d=ContactGate(Config()).evaluate(sole_points(h,h),load())
    assert d.selected and d.envelope_gate and not d.force_gate
    np.testing.assert_allclose(d.sole_min_height_m,[h,h],atol=1e-15)

@pytest.mark.parametrize('h',[.002001,.003,.05,.08])
def test_actual_separation_clears_contact_without_latch(h):
    gate=ContactGate(Config())
    assert gate.evaluate(sole_points(),load(414.)).selected
    assert not gate.evaluate(sole_points(h,h),load()).selected

@pytest.mark.parametrize('left,right',[(0.,.05),(.05,0.),(.002,.05)])
def test_either_sole_can_provide_contact(left,right):
    assert ContactGate(Config()).evaluate(sole_points(left,right),load()).selected

@pytest.mark.parametrize('x',[10.,10.2,-10.,-10.2])
def test_floor_edge_or_outside_is_not_inferred_support(x):
    d=ContactGate(Config()).evaluate(sole_points(x=x),load(400.))
    assert not d.selected and not d.foot_inside_floor_xy.any()

def test_completely_below_ground_box_is_not_inferred_support():
    assert not ContactGate(Config()).evaluate(sole_points(-.2,-.2),load()).selected

@pytest.mark.parametrize('mode',['contact_envelope','force_threshold'])
def test_no_contact_negative_control_forces_gate_off(mode):
    c=replace(Config(),no_contact=True,feedforward_contact_mode=mode)
    assert not ContactGate(c).evaluate(sole_points(),load(414.)).selected

@pytest.mark.parametrize('force',[0.,.999,1.,1.001,414.])
def test_legacy_mode_preserves_instantaneous_threshold_exactly(force):
    c=replace(Config(),feedforward_contact_mode='force_threshold')
    assert ContactGate(c).evaluate(sole_points(),load(force)).selected==(force>1.)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('where',['geometry','force'])
def test_nonfinite_sensor_inputs_stop(bad,where):
    p=sole_points();f=load()
    (p if where=='geometry' else f).flat[0]=bad
    with pytest.raises((ValueError,FloatingPointError)):
        ContactGate(Config()).evaluate(p,f)

@pytest.mark.parametrize('p,f',[(np.zeros((2,4,3)),load()),(sole_points(),np.zeros((1,3)))])
def test_wrong_shape_is_not_silently_accepted(p,f):
    with pytest.raises((ValueError,FloatingPointError)):ContactGate(Config()).evaluate(p,f)

def test_unknown_mode_rejected():
    with pytest.raises(ValueError):replace(Config(),feedforward_contact_mode='automatic_tuning').validate()

def test_geometry_is_shared_with_authoring_and_unchanged():
    assert tuple(GROUND_CENTER_M)==(0.,0.,-.05) and tuple(GROUND_SIZE_M)==(20.,20.,.1)
    source=(ROOT/'kangaroo_isaac/usd_builder.py').read_text()
    assert 'xform(floor.GetPrim(),GROUND_CENTER_M,np.eye(3),GROUND_SIZE_M)' in source
    assert 'p.CreateContactOffsetAttr(cfg.contact_offset_m)' in source
    d=ContactGate(Config()).description()
    assert d['contact_distance_m']==.002 and not d['force_injection']
    assert not d['native_manifold_equivalence_verified']

@pytest.mark.parametrize('case',['nominal','refined','solver_check','low_friction','higher_drop_push','no_feedforward','no_contact','passive'])
def test_mode_propagates_without_changing_any_other_configuration(case):
    a=asdict(case_config(case));b=asdict(case_config(case,feedforward_contact_mode='force_threshold'))
    assert a.pop('feedforward_contact_mode')=='contact_envelope'
    assert b.pop('feedforward_contact_mode')=='force_threshold'
    assert a==b

def test_new_configuration_keeps_all_previous_native_physics_and_gains():
    # 23.0.5 changes ONLY the sole/floor contact law (three explicit fields) and
    # the named numerical solver profile (five explicit fields, still one of the
    # declared SOLVER_PROFILES); 23.0.4 changed only the feedforward gate. Every
    # physical setting, gain, gate and clock equals 23.0.3.
    from kangaroo_isaac.control import SOLVER_PROFILES
    old=json.loads((FIX/'recorded_result.json').read_text())['configuration']
    new=asdict(Config());new.pop('feedforward_contact_mode')
    added={k:new.pop(k) for k in ('contact_model','contact_stiffness_per_s2','contact_damping_per_s')}
    solver={k:new.pop(k) for k in ('solver_profile','solver_type','solver_position_iterations',
                                   'solver_velocity_iterations','external_forces_every_iteration')}
    for k in solver:old.pop(k)
    assert new==old
    assert added['contact_model']=='source_compliance'
    assert solver=={'solver_profile':'pgs_64_8',**SOLVER_PROFILES['pgs_64_8']}


def test_actual_5ms_trace_replay_does_not_claim_new_native_motion():
    p=json.loads((FIX/'provenance.json').read_text())
    assert hashlib.sha256((FIX/'recorded_contact_trace.npz').read_bytes()).hexdigest()==p['derived_fixture_sha256']
    assert p['new_physics_simulated'] is False
    with np.load(FIX/'recorded_contact_trace.npz',allow_pickle=False) as a:
        c=ContactGate(Config());g=np.array([c.evaluate(x,f).selected for x,f in zip(a['sole_points_world'],a['foot_force'])])
        t=a['time'];tail=t>=9.5;fn=a['foot_force'][:,:,2].sum(1)
        assert len(t)==2001 and tail.sum()==101
        assert np.count_nonzero(tail&(fn==0))==19
        assert np.count_nonzero(np.diff((fn>1)[tail]))==32
        heights=a['sole_points_world'][:,:,:,2].min(axis=(1,2))
        off=tail&(fn<=1.)
        assert heights[off].min()>-1e-6 and heights[off].max()<2.3e-6
        assert g[tail].all() and not g[0]
        assert np.linalg.norm(a['base_velocity_origin'][tail],axis=1).max()>.06
        assert np.ptp(a['base'][tail,2])>.0023
    # The same recorded trajectory remains a failed task. Replay never changes it.
    r=json.loads((FIX/'recorded_result.json').read_text())
    assert authoritative_exit_code(r,0)==3
    failed=[g['name'] for g in r['validation']['gates'] if g['status']=='FAIL']
    assert failed==['maximum_final_base_speed_m_s']

@pytest.mark.parametrize('mode',['contact_envelope','force_threshold'])
@pytest.mark.parametrize('disable',['none','no_feedforward','passive'])
def test_exact_tick_logging_filter_force_and_feedforward_semantics(model,ref,tmp_path,mode,disable):
    c=replace(Config(),duration_s=.003,feedforward_contact_mode=mode,
              **({disable:True} if disable!='none' else {}))
    gate=ContactGate(c);ctrl=Controller(model,ref,c);trace=ControllerTrace(c)
    for k in range(round(c.duration_s/c.dt_s)):
        q=ctrl.targets[min(k//ctrl.control_stride,len(ctrl.targets)-1)]
        qd=ctrl.veltargets[min(k//ctrl.control_stride,len(ctrl.targets)-1)]
        if k%ctrl.control_stride==0:
            d=gate.evaluate(sole_points(),load())
            ctrl.update_command(k,q,qd,d.selected);trace.record(k,d,q,qd,ctrl)
        ctrl.advance_filter()
    a=trace.arrays();enabled=(mode=='contact_envelope' and disable=='none')
    np.testing.assert_array_equal(a['step'],[0,40,80])
    np.testing.assert_allclose(a['time_s'],[0.,.001,.002])
    assert (a['feedforward_enabled']==enabled).all()
    assert (a['legacy_force_gate']==False).all()
    np.testing.assert_array_equal(a['submitted_motor_force_N'][0],np.zeros(12))
    np.testing.assert_allclose(a['raw_command_N'],a['feedforward_N'],atol=1e-12)
    assert (np.abs(np.diff(np.vstack([np.zeros((1,12)),a['command_N']]),axis=0))<=c.command_slew_N_s*c.control_period_s+1e-9).all()
    trace.save(tmp_path,completed=True)
    summary=json.loads((tmp_path/'contact_gate_summary.json').read_text())
    assert summary['all_control_ticks_recorded'] and summary['recorded_control_ticks']==3
    assert summary['status']=='TELEMETRY_ONLY_NOT_A_TASK_PASS'
    assert not summary['native_contact_count_measured']
    trace.save(tmp_path,completed=False)
    assert (tmp_path/'partial_controller_trace.npz').exists()

@pytest.mark.parametrize('k',[1,39,40,-1,.5])
def test_trace_rejects_missed_or_noncontrol_first_tick(model,ref,k):
    c=Config();ctrl=Controller(model,ref,c);trace=ControllerTrace(c)
    d=ContactGate(c).evaluate(sole_points(),load())
    with pytest.raises(ValueError):trace.record(k,d,np.zeros(12),np.zeros(12),ctrl)

def test_trace_requires_controller_update_first(model,ref):
    c=Config();ctrl=Controller(model,ref,c);trace=ControllerTrace(c)
    with pytest.raises(ValueError,match='matching controller update'):
        trace.record(0,ContactGate(c).evaluate(sole_points(),load()),np.zeros(12),np.zeros(12),ctrl)

def test_empty_partial_trace_can_be_saved(tmp_path):
    trace=ControllerTrace(Config());trace.save(tmp_path,completed=False)
    s=json.loads((tmp_path/'partial_contact_gate_summary.json').read_text())
    assert s['recorded_control_ticks']==0 and not s['all_control_ticks_recorded']

def test_cli_exposes_legacy_control_and_defaults_to_geometry():
    p=subprocess.run([sys.executable,str(ROOT/'run_isaac.py'),'--help'],capture_output=True,text=True)
    assert p.returncode==0 and '--contact-mode {contact_envelope,force_threshold}' in p.stdout
    source=(ROOT/'run_isaac.py').read_text()
    assert "'--contact-mode',args.contact_mode" in source


def test_worker_writes_actual_causal_trace_with_api_fixture(monkeypatch,model,ref,tmp_path):
    # Uses controlled API arrays, NOT PhysX. Existing test also guards this status.
    from test_native_stage_binding import install_worker_fixture
    from kangaroo_isaac.native import run
    art,view,worlds=install_worker_fixture(monkeypatch,model,ref,tmp_path)
    code=run('nominal',replace(Config(),duration_s=.002),tmp_path,headless=True,visuals=False)
    assert code==3
    with np.load(tmp_path/'controller_trace.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['step'],[0,40])
        np.testing.assert_allclose(a['time_s'],[0.,.001])
        assert not a['feedforward_enabled'].any()  # fixture starts 5 cm above floor
    assert worlds[0].current_time_step_index==80 and len(art.wrenches)==80
    r=json.loads((tmp_path/'result.json').read_text())
    assert r['metrics']['feedforward_contact_gate']['recorded_control_ticks']==2
    assert not r['certified_ready']
