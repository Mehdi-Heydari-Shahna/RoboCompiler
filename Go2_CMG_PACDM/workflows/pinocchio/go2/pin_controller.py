"""Original torque controller extracted unchanged; no MuJoCo dependency."""
import numpy as np
import osqp
from scipy import sparse
from .pin_backend import PinBackend
from .contact import FootKinematics

class WholeBodyController:
    def __init__(self,cmg,friction=.5,feedforward=True):
        self.backend=PinBackend(cmg);self.kin=FootKinematics(cmg,cmg['q_reference'])
        self.armature=np.asarray(cmg['armature']);self.damping=np.asarray(cmg['damping']);self.friction=np.asarray(cmg['frictionloss'])
        self.limits=np.tile([23.7,23.7,45.43],4);self.mu=friction;self.feedforward=feedforward
        self.previous=None

    def command(self,q,v,qref,vref,aref,active,av,aa,stance):
        points,Jfeet=self.kin.points_and_jacobians(q);J=Jfeet.reshape(12,18)
        eps=1e-5
        Jp=self.kin.points_and_jacobians(q+eps*v)[1].reshape(12,18)
        Jm=self.kin.points_and_jacobians(q-eps*v)[1].reshape(12,18)
        jdv=(Jp-Jm)@v/(2*eps)
        M=self.backend.mass(q)+np.diag(self.armature);h=self.backend.bias(q,v)
        passive_comp=self.damping*v+self.friction*np.tanh(v/.02)
        desired_base=aa[:6]+np.array([100,100,160,120,120,120])*(active[:6]-q[:6])+np.array([20,20,25,22,22,22])*(av[:6]-v[:6])
        desired_feet=aa[6:]+160*(active[6:]-points.ravel())+25*(av[6:]-J@v)-jdv
        desired_joint=aref[6:]+60*(qref[6:]-q[6:])+12*(vref[6:]-v[6:])
        # Weighted acceleration tracking; soft foot tasks tolerate native
        # compliant/rolling contact without silently imposing a physical weld.
        L=np.zeros((36,30));target=np.r_[desired_base,desired_feet,desired_joint,np.zeros(6)]
        L[:6,:6]=np.eye(6);L[6:18,:18]=J;L[18:30,6:18]=np.eye(12)
        weights=np.r_[np.full(3,30.),np.full(3,20.),np.full(12,5.),np.full(12,.03),np.zeros(6)]
        P=L.T@(weights[:,None]*L)+np.diag(np.r_[np.full(18,1e-5),np.full(12,2e-5)])
        c=-L.T@(weights*target)
        dynamics=np.c_[M[:6],-J[:,:6].T]
        torquemap=np.c_[M[6:],-J[:,6:].T];offset=h[6:]+passive_comp[6:]
        constraints=[dynamics,torquemap];lower=[-h[:6],-self.limits-offset];upper=[-h[:6],self.limits-offset]
        for leg in range(4):
            rows=np.zeros((5,30));ix=18+3*leg;rows[0,ix+2]=1
            mu=self.mu/np.sqrt(2)
            rows[1,ix]=1;rows[1,ix+2]=-mu;rows[2,ix]=-1;rows[2,ix+2]=-mu
            rows[3,ix+1]=1;rows[3,ix+2]=-mu;rows[4,ix+1]=-1;rows[4,ix+2]=-mu
            constraints.append(rows);lower.append(np.array([0,-np.inf,-np.inf,-np.inf,-np.inf]))
            upper.append(np.array([180 if stance[leg] else 0,0,0,0,0]))
        A=np.vstack(constraints);lo=np.concatenate(lower);hi=np.concatenate(upper)
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(np.triu(P)),q=c,A=sparse.csc_matrix(A),l=lo,u=hi,
                                       verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=4000,polishing=True)
        if self.previous is not None:solver.warm_start(x=self.previous)
        answer=solver.solve(raise_error=False)
        if answer.info.status_val not in (1,2):raise RuntimeError('Whole-body QP: '+answer.info.status)
        self.previous=answer.x.copy();acc=answer.x[:18];forces=answer.x[18:].reshape(4,3)
        tau=torquemap@answer.x+offset
        if not self.feedforward:tau=60*(qref[6:]-q[6:])+3*(vref[6:]-v[6:])
        self.requested_torque=tau.copy()  # motor torque requested before the safety clip
        violation=max(float(np.max(lo-A@answer.x)),float(np.max(A@answer.x-hi)),0.)
        return np.clip(tau,-self.limits,self.limits),forces,violation,answer.info.iter

