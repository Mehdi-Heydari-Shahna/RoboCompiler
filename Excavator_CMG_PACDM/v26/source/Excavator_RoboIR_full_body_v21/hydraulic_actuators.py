"""Synthetic double-acting fluid actuators; not identified hardware hydraulics.
Six compressible cylinders plus two torque-limited first-order rotary drives.
Positive source q expands chamber A. Ideal supply/tank pressure reservoirs.
The parameters belong to the excavator model, which is supplied at run time.
"""
import sys
from pathlib import Path
import numpy as np
_ACCESS=next(p/'model_access' for p in Path(__file__).resolve().parents if (p/'model_access'/'excavator_model.py').is_file())
if str(_ACCESS) not in sys.path:sys.path.append(str(_ACCESS))
import excavator_model  # noqa: E402
def parameters():
 """Cylinder and rotary-drive parameters (model parameter file, entry hydraulics)."""
 return excavator_model.parameter('hydraulics')
class HydraulicBank:
 def __init__(self,nominal_ports,initial_effort,capacity_scale=1.):
  self.p=parameters();self.nominal=np.asarray(nominal_ports)[:6].copy();self.A=np.array(self.p['area_A_m2']);self.B=np.array(self.p['area_B_m2'])
  self.ps=self.p['supply_pressure_Pa']*capacity_scale;self.pressure=self.targets(np.asarray(initial_effort)[:6])
  self.rotary_limit=np.array(self.p['rotary_torque_limit_Nm'])*capacity_scale;self.rotary=np.clip(np.asarray(initial_effort)[6:],-self.rotary_limit,self.rotary_limit)
 def targets(self,force,velocity=None):
  low=min(self.p['minimum_command_pressure_Pa'],self.ps*.05);pa=np.where(force>=0,(force+self.B*low)/self.A,low);pb=np.where(force>=0,low,(self.A*low-force)/self.B)
  if velocity is not None:
   va=np.minimum(self.A*velocity,0);vb=np.minimum(-self.B*velocity,0)
   ra=low+1.3*self.ps*(va/self.p['valve_max_flow_m3_s'])**2;rb=low+1.3*self.ps*(vb/self.p['valve_max_flow_m3_s'])**2
   preload=np.maximum.reduce([self.A*(ra-pa),self.B*(rb-pb),np.zeros(6)]);pa+=preload/self.A;pb+=preload/self.B
  return np.clip(np.column_stack((pa,pb)),0,self.ps)
 def volumes(self,q):
  x=np.asarray(q)[:6]-self.nominal;span=self.p['coordinate_half_span_m']
  if np.max(abs(x))>=span:raise ValueError('Cylinder left positive chamber-volume domain')
  return self.p['dead_volume_m3']+np.column_stack((self.A*(span+x),self.B*(span-x)))
 def stored_energy(self,q):return float(np.sum(self.volumes(q)*self.pressure**2)/(2*self.p['bulk_modulus_Pa']))
 def evaluate(self,q,v,command,dt):
  v=np.asarray(v);command=np.asarray(command);p=self.pressure;V=self.volumes(q);beta=self.p['bulk_modulus_Pa']
  dv=np.column_stack((self.A*v[:6],-self.B*v[:6]));target=self.targets(command[:6],v[:6])
  leak=self.p['leakage_m3_s_Pa']*(p[:,0]-p[:,1]);L=np.column_stack((leak,-leak))
  request=dv+L+V/beta*(target-p)/self.p['pressure_response_s']
  capacity=self.p['valve_max_flow_m3_s']*np.sqrt(np.maximum(np.where(request>=0,self.ps-p,p),0)/self.ps);flow=np.clip(request,-capacity,capacity)
  unbounded=p+dt*beta/V*(flow-dv-L);bounded=np.clip(unbounded,0,self.ps);relief=(unbounded-bounded)*V/(dt*beta);dp=beta/V*(flow-dv-L-relief)
  eps=self.p['friction_smoothing_speed_SI'];fric=np.array(self.p['cylinder_coulomb_N'])*np.tanh(v[:6]/eps)+np.array(self.p['cylinder_viscous_N_s_m'])*v[:6]
  rf=np.array(self.p['rotary_coulomb_Nm'])*np.tanh(v[6:]/eps)+np.array(self.p['rotary_viscous_Nm_s_rad'])*v[6:]
  effort=np.r_[self.A*p[:,0]-self.B*p[:,1]-fric,self.rotary-rf];dr=(np.clip(command[6:],-self.rotary_limit,self.rotary_limit)-self.rotary)/self.p['rotary_response_s']
  supply=self.ps*np.maximum(flow,0).sum();fluid_input=np.sum(p*flow)
  power=dict(supply=float(supply),throttle=float(supply-fluid_input),leakage=float(np.sum(leak*(p[:,0]-p[:,1]))),
   relief=float(np.sum(p*relief)),friction=float(fric@v[:6]+rf@v[6:]),compressibility_geometry=float(np.sum(p*p*dv)/(2*beta)),rotary=float(self.rotary@v[6:]),mechanical=float(effort@v))
  rate=float(np.sum(V*p*dp)/beta+np.sum(p*p*dv)/(2*beta))
  identity=rate-(power['supply']-power['throttle']-power['leakage']-power['relief']-float((self.A*p[:,0]-self.B*p[:,1])@v[:6])+power['compressibility_geometry'])
  return dict(effort=effort,pressure_derivative=dp,rotary_derivative=dr,flow=flow,power=power,fluid_identity_W=identity,
   pressure_limit_active=bool(np.any(target<=0)|np.any(target>=self.ps)),valve_saturated=bool(np.any(abs(request)>capacity+1e-15)))
 def advance(self,item,dt):
  self.pressure+=dt*item['pressure_derivative'];self.rotary+=dt*item['rotary_derivative']
  if self.pressure.min()<-1e-6 or self.pressure.max()>self.ps+1e-6:raise ValueError('Pressure bound violated')
  self.pressure=np.clip(self.pressure,0,self.ps)
