"""Native Pinocchio oracle (optional).

Native CRBA/RNEA/FK/Jacobian/classical acceleration and ABA are called. The
one-dimensional affine finger equality is applied with a NumPy KKT linear solve
using native mass/bias data. This is NOT Pinocchio constraintDynamics, and the
virtual tool target is never imposed as a fixed physical contact.
"""
from __future__ import annotations
import numpy as np
from . import bootstrap
from panda.pin_backend import PinBackend
import pinocchio as pin

class NativeOracle:
    def __init__(self,comp):
        self.version=pin.__version__;self.backend=PinBackend(comp.cmg);b=self.backend;tool=comp.cmg['tool'];body=tool['body']
        placement=b._placements[body]@np.array(tool['T_body_tool'])
        self.tool_frame=int(b.model.addFrame(pin.Frame('compiled_tool_oracle',b._supports[body],b.body_frame_ids[body],pin.SE3(placement[:3,:3],placement[:3,3]),pin.FrameType.OP_FRAME),False))
        b.data=b.model.createData()
    def tool(self,q,v,a=None):
        b=self.backend;m,d=b.model,b.data;qn=b._native_q(q);vn=b._native_v(v,'v');an=np.zeros(m.nv)if a is None else b._native_v(a,'a')
        J=np.asarray(pin.computeFrameJacobian(m,d,qn,self.tool_frame,pin.LOCAL_WORLD_ALIGNED)).copy()[:,b._v_indices]
        pin.forwardKinematics(m,d,qn,vn,np.zeros(m.nv));pin.updateFramePlacements(m,d)
        gamma=np.asarray(pin.getFrameClassicalAcceleration(m,d,self.tool_frame,pin.LOCAL_WORLD_ALIGNED).vector).copy()
        T=np.asarray(d.oMf[self.tool_frame].homogeneous).copy()
        if a is None:acc=gamma.copy()
        else:
            pin.forwardKinematics(m,d,qn,vn,an);pin.updateFramePlacements(m,d)
            acc=np.asarray(pin.getFrameClassicalAcceleration(m,d,self.tool_frame,pin.LOCAL_WORLD_ALIGNED).vector).copy()
        return dict(T=T,J=J,gamma=gamma,acceleration=acc)
    def aba_roundtrip(self,q,v,a):
        b=self.backend;m,d=b.model,b.data;qn=b._native_q(q);vn=b._native_v(v,'v');an=b._native_v(a,'a')
        tau=np.asarray(pin.rnea(m,d,qn,vn,an)).copy();a2=np.asarray(pin.aba(m,d,qn,vn,tau)).copy()
        return a2[b._v_indices]
