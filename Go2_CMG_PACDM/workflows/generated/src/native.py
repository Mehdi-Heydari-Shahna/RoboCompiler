"""Native Pinocchio CONTACT_3D support oracle, independent of PACDM Jacobians.

Uses the unchanged original CMG-to-Pinocchio backend. Supports are ideal
stationary point centers, zero stabilization and zero regularization. Not the
unilateral compliant contact model of the original locomotion rollout.
"""
from __future__ import annotations
import numpy as np
import pinocchio as pin
from . import bootstrap
from go2.pin_backend import PinBackend,_se3


class NativeSupportOracle:
    def __init__(self,comp,q,sites):
        self.version=pin.__version__;self.backend=PinBackend(comp.cmg);b=self.backend;self.model=b.model
        poses=b.poses(q);self.attachments=[];self.constraints=[]
        for i in sites:
            f=comp.cmg['feet'][i];body=f['body'];p=np.asarray(f['point_m'])
            placement=b._placements[body].copy();placement[:3,3]+=placement[:3,:3]@p
            joint=b._supports[body];target=np.eye(4);target[:3,3]=poses[body][:3,:3]@p+poses[body][:3,3]
            c=pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,self.model,joint,_se3(placement),0,_se3(target),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            c.name='support_'+f['id']
            if np.any(c.corrector.Kp) or np.any(c.corrector.Kd):raise RuntimeError('Expected zero native stabilization')
            self.constraints.append(c);self.attachments.append((joint,_se3(placement)))
        self.data=self.model.createData();self.geomdata=self.model.createData();self.constraint_data=[c.createData()for c in self.constraints]
        pin.initConstraintDynamics(self.model,self.data,self.constraints)

    def acceleration(self,q,v,tau):
        b=self.backend
        settings=pin.ProximalSettings(1e-13,1e-13,0.,50)
        a=pin.constraintDynamics(self.model,self.data,b._native_q(q),b._native_v(v,'v'),b._native_v(tau,'tau'),self.constraints,self.constraint_data,settings)
        a=np.asarray(a)[b._v_indices].copy()
        if not np.all(np.isfinite(a)):raise FloatingPointError('Nonfinite native support acceleration')
        return a

    def geometry(self,q,v):
        b=self.backend;m=self.model;d=self.geomdata
        pin.computeJointJacobians(m,d,b._native_q(q))
        pin.forwardKinematics(m,d,b._native_q(q),b._native_v(v,'v'),np.zeros(m.nv));pin.updateFramePlacements(m,d)
        js=[];bias=[]
        for support,placement in self.attachments:
            js.append(np.asarray(pin.getFrameJacobian(m,d,support,placement,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:3,b._v_indices].copy())
            bias.append(np.asarray(pin.getFrameClassicalAcceleration(m,d,support,placement,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear).copy())
        return dict(J=np.vstack(js),gamma=np.concatenate(bias))
