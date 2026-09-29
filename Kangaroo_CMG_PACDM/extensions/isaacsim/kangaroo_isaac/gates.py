"""No unavailable measurement, partial run, or negative control becomes a PASS."""
from __future__ import annotations
import json,math
from .model import ROOT

SOURCE=json.loads((ROOT/'data/contact_acceptance.json').read_text())
UNAVAILABLE={
    'maximum_energy_ledger_error_J':'Independent native loop/contact/friction/limit reaction-work ledger is not implemented.',
    'maximum_constraint_power_decomposition_error_W':'External loop reactions are not independently resolved by this adapter.',
    'maximum_actuator_virtual_work_error_W':'Only geometric port-power reconstruction is available, not independent native motor wrench measurement.',
}

def check(name,value,threshold,comparison='<='):
    if value is None:return {'name':name,'status':'NOT_MEASURED','value':None,'threshold':threshold,'comparison':comparison}
    good=isinstance(value,(int,float)) and math.isfinite(value)
    if good:good=value<=threshold if comparison=='<=' else value>=threshold
    return {'name':name,'status':'PASS' if good else 'FAIL','value':value if isinstance(value,(int,float)) and math.isfinite(value) else None,'threshold':threshold,'comparison':comparison}


def assess(case,cfg,metrics,native_audit,completed):
    observed=metrics.get('observed_duration_s',0.)
    complete=bool(completed and metrics.get('steps',-1)==round(cfg.duration_s/cfg.dt_s)
                  and abs(observed-cfg.duration_s)<cfg.dt_s/10)
    gates=[{'name':'native_initialization_audit','status':native_audit.get('status','NOT_RUN')},
           {'name':'complete_declared_duration','status':'PASS' if complete else 'FAIL'},
           {'name':'physical_per_port_force_limits','status':'FAIL' if metrics.get('force_limit_violation',True) else 'PASS'}]
    negative=case in ('no_contact','no_loops','passive')
    if negative:
        if case=='no_contact':gates.append(check('expected_fall_without_contact_m',metrics.get('base_fall_m'),.1,'>='))
        elif case=='no_loops':gates.append(check('expected_gap_without_loops_m',metrics.get('maximum_loop_gap_m'),.01,'>='))
        else:gates.append(check('expected_fall_without_motors_m',metrics.get('base_fall_m'),.05,'>='))
        ok=all(g['status']=='PASS' for g in gates)
        return {'functional_status':'EXPECTED_FAILURE_OBSERVED' if ok else 'NEGATIVE_CONTROL_FAILED',
                'scope':'negative control; never positive robot/framework validation',
                'gates':gates,'full_source_validation_status':'NOT_APPLICABLE','certified_ready':False}
    for name,threshold in SOURCE['positive_trials'].items():
        if name in UNAVAILABLE:continue
        gates.append(check(name,metrics.get(name),threshold))
    for name,threshold,direction in (
        ('achieved_crouch_m',.1,'>='),('achieved_lateral_excursion_m',.065,'>='),
        ('minimum_slide_margin_m',0.,'>='),('minimum_hinge_margin_rad',0.,'>='),
        ('final_weight_error_N',5.,'<=')):
        gates.append(check(name,metrics.get(name),threshold,direction))
    drift=metrics.get('maximum_foot_drift_after_landing_m')
    gates.append(check('maximum_foot_drift_after_landing_m',max(drift) if drift is not None else None,.015))
    functional=all(g['status']=='PASS' for g in gates)
    unavailable=[{'name':k,'status':'NOT_IMPLEMENTED','reason':v,'source_threshold':SOURCE['positive_trials'][k]} for k,v in UNAVAILABLE.items()]
    return {'functional_status':('FUNCTIONAL_GATES_PASSED' if functional else 'FUNCTIONAL_GATES_FAILED'),
            'scope':'native task/kinematic gates only; complete source validation is separate',
            'is_ablation':case=='no_feedforward','gates':gates,
            'unavailable_source_gates':unavailable,
            'full_source_validation_status':'BLOCKED_INCOMPLETE_REACTION_AUDIT','certified_ready':False}
