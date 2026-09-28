"""Streaming observables from real native states; never advances/project states.

Energy minus known work is deliberately called UNRESOLVED, not ledger error:
external loop reaction work and all contact/friction work are not independently
available in this port. Closing that balance by subtraction is not validation.
"""
from __future__ import annotations
import numpy as np
from .model import finite

class Recorder:
    def __init__(self,model,ref,cfg):
        self.m,self.ref,self.cfg=model,ref,cfg
        self.samples={k:[] for k in ('time','q','qd','body_pose_xyzw','body_velocity_com_world',
            'motor_force','command','motor_velocity','motor_power','motor_work','foot_force',
            'foot_position','com','push','energy','known_work','unresolved_energy','loop_gap',
            'universal_dot','tilt_deg','base','base_velocity_origin','port_power_error')}
        self.metrics={k:0. for k in ('maximum_loop_gap_m','maximum_universal_dot',
            'maximum_motor_force_N','maximum_penetration_m','maximum_tilt_deg',
            'maximum_motor_error_m','maximum_port_power_reconstruction_error_W')}
        self.metrics['minimum_slide_margin_m']=1e10
        self.metrics['minimum_hinge_margin_rad']=1e10
        self.motor_work=np.zeros(12);self.motor_positive_work=np.zeros(12);self.motor_negative_work=np.zeros(12)
        self.passive_work=0.;self.disturbance_work=0.;self.initial_energy=None
        self.foot_anchor=None;self.foot_drift=np.zeros(2);self.touchdown=None
        self.steps=0;self.max_final_speed=0.;self.final_normals=[]
        self.force_limit_violation=False

    def observe(self,k,P,V,q,qd,force,command,feet,push,pose_xyzw,save=False):
        m=self.m;c=self.cfg;t=k*c.dt_s
        q=finite(q,(76,),'measured q');qd=finite(qd,(76,),'measured velocity')
        V=finite(V,(78,6),'measured body velocity');force=finite(force,(12,),'motor force sent to PhysX')
        feet=finite(feet,(2,3),'measured contact force');finite(P,(78,4,4),'measured body poses')
        gap,dots,_=m.closures(P);gapmax=float(np.linalg.norm(gap,axis=1).max())
        base=P[m.root,:3,3];baseR=P[m.root,:3,:3]
        tilt=float(np.degrees(np.arccos(np.clip(baseR[2,2],-1,1))))
        footpoints=m.foot_points(P);pen=float(max(0.,-footpoints[:,:,2].min())) if not c.no_contact else 0.
        margin=np.minimum(q-m.lower,m.upper-q)
        self.metrics['minimum_slide_margin_m']=min(self.metrics['minimum_slide_margin_m'],float(margin[m.slide].min()))
        self.metrics['minimum_hinge_margin_rad']=min(self.metrics['minimum_hinge_margin_rad'],float(margin[~m.slide].min()))
        for key,val in [('maximum_loop_gap_m',gapmax),('maximum_universal_dot',float(abs(dots).max())),
                        ('maximum_motor_force_N',float(abs(force).max())),('maximum_penetration_m',pen),('maximum_tilt_deg',tilt)]:
            self.metrics[key]=max(self.metrics[key],val)
        self.force_limit_violation|=bool(np.any(force<m.bounds[:,0]-1e-5) or np.any(force>m.bounds[:,1]+1e-5))
        if t>=c.duration_s-.5:
            v0=V[m.root,:3]+np.cross(V[m.root,3:],-baseR@m.com_local[m.root])
            self.max_final_speed=max(self.max_final_speed,float(np.linalg.norm(v0)))
            self.final_normals.append(float(feet[:,2].sum()))
        if feet[:,2].sum()>1. and self.touchdown is None:self.touchdown=t
        if self.touchdown is not None and t>=max(1.,self.touchdown+.2):
            fp=P[m.feet,:3,3]
            if self.foot_anchor is None:self.foot_anchor=fp.copy()
            self.foot_drift=np.maximum(self.foot_drift,np.linalg.norm(fp-self.foot_anchor,axis=1))
        if not save:return
        K,U,rotor=m.energy(P,V,qd);E=float(K.sum()+U.sum()+rotor.sum())
        if self.initial_energy is None:self.initial_energy=E
        _,portpower=m.port_wrenches(P,V,force)
        motorpower=force*qd[m.active];port_error=abs(portpower-float(motorpower.sum()))
        self.metrics['maximum_port_power_reconstruction_error_W']=max(self.metrics['maximum_port_power_reconstruction_error_W'],port_error)
        err=float(abs(self.ref.u(t)-q[m.active]).max())
        self.metrics['maximum_motor_error_m']=max(self.metrics['maximum_motor_error_m'],err)
        com=m.com_world(P)
        known=self.motor_work.sum()+self.passive_work+self.disturbance_work
        v0=V[m.root,:3]+np.cross(V[m.root,3:],-baseR@m.com_local[m.root])
        vals=(t,q,qd,pose_xyzw,V,force,command,qd[m.active],motorpower,self.motor_work,
              feet,P[m.feet,:3,3],(m.mass[:,None]*com).sum(axis=0)/m.mass.sum(),push,
              E,known,E-self.initial_energy-known,gap,dots,tilt,base,v0,port_error)
        for key,val in zip(self.samples,vals):self.samples[key].append(np.asarray(val).copy())

    def integrate_step(self,force,pre_qd,post_qd,push,pre_V,post_V):
        """Midpoint quadrature; motor efforts are the submitted filtered forces.

        Passive power is reconstructed from the declared viscous law, not a
        separately measured native drive force. A native force/energy audit
        must not treat this known-work subtotal as a complete force ledger.
        """
        vel=.5*(pre_qd+post_qd)
        increment=force*vel[self.m.active]*self.cfg.dt_s
        self.motor_work+=increment;self.motor_positive_work+=np.maximum(increment,0.)
        self.motor_negative_work+=np.minimum(increment,0.)
        self.passive_work-=float(self.m.damping@(vel**2))*self.cfg.dt_s
        self.disturbance_work+=float(push@(.5*(pre_V[self.m.torso,:3]+post_V[self.m.torso,:3])))*self.cfg.dt_s
        self.steps+=1

    def arrays(self):return {k:np.asarray(v) for k,v in self.samples.items()}

    def summarize(self):
        a=self.arrays();t=a['time'];base=a['base'];c=self.cfg;m=self.m
        if len(t)==0:raise ValueError('No measured native observations')
        settled=(t>=1.)&(t<=1.4)
        crouch=(t>=2.5)&(t<=5.3)
        lat=(t>=2.6)&(t<=5.3)
        achieved_crouch=float(np.mean(base[settled,2])-np.min(base[crouch,2])) if settled.any() and crouch.any() else 0.
        achieved_lateral=float(np.ptp(base[lat,1])) if lat.any() else 0.
        return dict(self.metrics,steps=self.steps,observed_duration_s=float(t[-1]),
            touchdown_s=self.touchdown,final_tilt_deg=float(a['tilt_deg'][-1]),
            final_base_position_m=base[-1].tolist(),base_fall_m=float(base[0,2]-base[-1,2]),
            achieved_crouch_m=achieved_crouch,achieved_lateral_excursion_m=achieved_lateral,
            maximum_final_base_speed_m_s=self.max_final_speed,
            final_position_error_m=float(np.linalg.norm(base[-1]-self.ref.base(t[-1]))),
            maximum_foot_drift_after_landing_m=self.foot_drift.tolist(),
            final_weight_error_N=abs(float(np.mean(self.final_normals))-m.mass.sum()*9.81) if self.final_normals else None,
            final_motor_work_J=self.motor_work.tolist(),positive_motor_work_J=self.motor_positive_work.tolist(),
            negative_motor_work_J=self.motor_negative_work.tolist(),
            work_known_J={'motor':float(self.motor_work.sum()),'passive_reconstructed':self.passive_work,'disturbance':self.disturbance_work},
            energy_change_J=float(a['energy'][-1]-a['energy'][0]),
            final_unresolved_energy_J=float(a['unresolved_energy'][-1]),
            force_limit_violation=self.force_limit_violation,
            extrema_cadence='physics step, except motor error/port power sampled at sample_period_s',
            foot_drift_window='from max(1.0 s, touchdown+0.2 s)',
            complete_energy_ledger_available=False,
            motor_force_measurement='causal force state submitted to native effort API; not an independent force sensor')
