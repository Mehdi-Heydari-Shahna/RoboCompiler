"""Reproduce the complete release. Fails with a nonzero exit code on a failed gate."""
import os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
 os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,platform,sys,time
from importlib.metadata import version
import numpy as np
from benchmark_model import ROOT,save_json
from mechanics_validation import validate_mechanics
from digging_simulation import run_case
from contact_validation import audit_contact_run

CASES={
 'nominal':dict(dt=.0005),
 'coarse':dict(dt=.001),
 'fine':dict(dt=.00025),
 'empty':dict(dt=.0005,soil=False),
 'no_bucket_contact':dict(dt=.0005,bucket_contact=False),
 'lower_friction':dict(dt=.0005,friction=.45),
 'denser_material':dict(dt=.0005,density_scale=1.25),
}

def evaluate_suite(reports,mechanics):
 checks=list(mechanics['checks'])
 def gate(name,value,limit):checks.append(dict(name=name,passed=bool(np.isfinite(value) and value<=limit),value=float(value),limit=float(limit)))
 for name,r in reports.items():
  prefix=name+': '
  for key,limit in [('warning_count',0),('peak_loop_gap_m',1e-4),('shadow_position_max_SI',1e-3),('peak_power_identity_error_W',1e-6),('peak_fluid_identity_error_W',1e-6),('peak_native_equilibrium_relative',1e-6),('mechanical_balance_relative',.01),('total_balance_relative',.005)]:gate(prefix+key,r[key],limit)
  gate(prefix+'pressure bounds',max(0.,r['peak_pressure_Pa']-r['supply_pressure_Pa'],-r['min_pressure_Pa']),1e-6)
  gate(prefix+'no post-initialization projection',int(r['native_state_projection']),0)
  gate(prefix+'passive q22 is unactuated',int(r['q22_actuated']),0)
  gate(prefix+'contact tracking remains below 0.2 rad',max(r['peak_tracking_error_rad']),.2)
  if name not in ['empty','no_bucket_contact']:
   mass=r['scene']['soil']['particle_density_kg_m3']*4*np.pi*r['scene']['soil']['radius_m']**3/3*r['scene']['density_scale']
   gate(prefix+'at least three grains carried',3-r['carried_particle_mass_kg']/mass,0)
   gate(prefix+'at least two carried grains deposited',2-r['deposited_carried_mass_kg']/mass,0)
  else:gate(prefix+'no material carried',r['carried_particle_mass_kg'],1e-9)
  a=np.load(ROOT/'results'/f'{name}.npz');steady=(a['time']>=.5)&(a['time']<=.9)
  support=float(np.mean(a['wrenches'][steady,1,2]));weight=r['source_mass_kg']*9.81
  gate(prefix+'initial support balances robot weight',abs(support-weight)/weight,.03)
  if name=='no_bucket_contact':gate(prefix+'disabled bucket has zero contact wrench',np.max(abs(a['wrenches'][:,0,:])),1e-10)
 for name in ['nominal','no_bucket_contact']:
  if name in reports:
   audit=audit_contact_run(name)
   gate(name+': independent floating-body and grain energy',audit['max_independent_energy_error_J'],1e-6)
   gate(name+': independent energy of every individual body',audit['max_individual_body_energy_error_J'],1e-6)
   gate(name+': explicit contact wrenches versus generalized forces',audit['max_contact_wrench_mapping_relative'],1e-8)
 if all(x in reports for x in ['coarse','nominal','fine']):
  errs=[abs(reports[x]['mechanical_balance_defect_J']) for x in ['coarse','nominal','fine']]
  gate('Refined mechanical balance improves over coarse',errs[2]/max(errs[0],1.),1.)
  gate('Refined mechanical balance below 0.5 percent of activity',reports['fine']['mechanical_balance_relative'],.005)
 report=dict(passed=all(x['passed'] for x in checks),passed_count=sum(x['passed'] for x in checks),check_count=len(checks),checks=checks,cases=list(reports),versions={x:version(x) for x in ['numpy','scipy','mujoco','pin','jsonschema']},python=sys.version,platform=platform.platform(),scope='Simulation verification with synthetic hydraulics and coarse noncohesive aggregate; no hardware or measured-soil validation.')
 save_json(ROOT/'results/validation.json',report)
 print(('PASS' if report['passed'] else 'FAIL'),f"{report['passed_count']}/{report['check_count']} gates",flush=True)
 for x in checks:
  if not x['passed']:print('FAILED',x,flush=True)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=list(CASES));p.add_argument('--audit-existing',action='store_true');p.add_argument('--mechanics-only',action='store_true');p.add_argument('--render',action='store_true');a=p.parse_args()
 if a.case:run_case(a.case,**CASES[a.case]);return
 if a.audit_existing:mechanics=json.loads((ROOT/'results/mechanics_validation.json').read_text())
 else:mechanics=validate_mechanics()
 if a.mechanics_only:
  if not mechanics['passed']:sys.exit(1)
  return
 reports={}
 for name,params in CASES.items():
  if a.audit_existing:reports[name]=json.loads((ROOT/'results'/f'{name}.json').read_text())
  else:reports[name]=run_case(name,**params)[0]
 result=evaluate_suite(reports,mechanics)
 from release_report import make_report
 make_report(reports,result)
 if a.render:
  from render_digging import render_video
  render_video('nominal')
 if not result['passed']:sys.exit(1)
if __name__=='__main__':main()
