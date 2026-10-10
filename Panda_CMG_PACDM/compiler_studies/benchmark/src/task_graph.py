"""Generated task evaluator, affine lowering and dependency-aware continuation.

The SE(3) residual, PACDM corrector/acquisition/mapping are taken from the
unchanged original core. New code generates the evaluated joint path and an
exact affine lift. Tool residual evaluation traverses ancestors only. No IK
closed form, target solution, or hardware measurements are used by a solver.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from . import bootstrap
from .compiler import inverse
from vendor.pacdm_original import PACDM,skew,log,inv_left,adj
from panda.task import TaskGraph

class ToolEvaluator:
    def __init__(self,comp):
        self.comp=comp;self.edges=[];self.tool=np.array(comp.cmg['tool']['T_body_tool']);self.calls=0
        records={j['id']:j for j in comp.cmg['joints']};ids=comp.cmg['coordinate_ids']
        for name in comp.plan['tool_path']:
            j=records[name];axis=np.array(j['axis']);K=skew(axis)
            self.edges.append((j['type'],np.array(j['T_BJ']),inverse(np.array(j['T_FJ'])),axis,K,K@K,
                None if j['type']=='fixed' else ids.index(name)))
    def evaluate(self,q,need_jac=True):
        self.calls+=1;R=np.eye(3);p=np.zeros(3);records=[]
        for kind,E,L,axis,K,K2,k in self.edges:
            p=p+R@E[:3,3];R=R@E[:3,:3]
            if k is not None:
                aw=R@axis
                if need_jac:records.append((kind,k,aw,p.copy()))
                if kind=='revolute':R=R@(np.eye(3)+np.sin(q[k])*K+(1-np.cos(q[k]))*K2)
                else:p=p+aw*q[k]
            p=p+R@L[:3,3];R=R@L[:3,:3]
        p=p+R@self.tool[:3,3];R=R@self.tool[:3,:3];T=np.eye(4);T[:3,:3]=R;T[:3,3]=p
        if not need_jac:return T,None
        J=np.zeros((6,9))
        for kind,k,aw,origin in records:
            if kind=='revolute':J[:3,k]=np.cross(aw,p-origin);J[3:,k]=aw
            else:J[:3,k]=aw
        return T,J

def target_kinematics(coords,reference,velocity=None,acceleration=None):
    """Independent target chart: translation XYZ, local intrinsic XYZ rotation.

    Returns pose, geometric Jacobian [linear;angular], classical acceleration
    for supplied chart velocity/acceleration. No PACDM quantities enter here.
    """
    coords=np.asarray(coords);R=np.asarray(reference).copy();axes=[];omega=np.zeros(3);gamma=np.zeros(3)
    v=np.zeros(6)if velocity is None else np.asarray(velocity)
    a=np.zeros(6)if acceleration is None else np.asarray(acceleration)
    for i,axis in enumerate(np.eye(3)):
        aw=R@axis;axes.append(aw)
        gamma+=np.cross(omega,aw)*v[3+i];omega+=aw*v[3+i]
        K=skew(axis);R=R@(np.eye(3)+np.sin(coords[3+i])*K+(1-np.cos(coords[3+i]))*(K@K))
    A=np.column_stack(axes);T=np.eye(4);T[:3,:3]=R;T[:3,3]=coords[:3]
    J=np.zeros((6,6));J[:3,:3]=np.eye(3);J[3:,3:]=A
    return T,J,np.r_[a[:3],A@a[3:]+gamma]

def target_evaluate(coords,reference,need_jac=True):
    """Residual-only route skips target and tool Jacobian construction alike."""
    R=np.asarray(reference).copy();axes=[]
    for i in range(3):
        if need_jac:axes.append(R[:,i].copy())
        c,s=np.cos(coords[3+i]),np.sin(coords[3+i])
        if i==0:Q=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
        elif i==1:Q=np.array([[c,0,s],[0,1.,0],[-s,0,c]])
        else:Q=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        R=R@Q
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=coords[:3]
    if not need_jac:return T,None
    J=np.zeros((6,6));J[:3,:3]=np.eye(3);J[3:,3:]=np.column_stack(axes)
    return T,J

class GeneratedTaskGraph:
    def __init__(self,comp,eliminate=True,redundancy=None):
        self.comp=comp;self.eliminate=eliminate;self.cmg=comp.cmg
        self.nt=8 if eliminate else 9;self.n=self.nt+6;self.nc=1 if eliminate else 2
        self.ids=list(comp.plan['free_physical_ids'] if eliminate else comp.cmg['coordinate_ids'])+[f'target_{k}'for k in ['x','y','z','rx','ry','rz']]
        self.L=comp.S if eliminate else np.eye(9);self.offset=comp.offset if eliminate else np.zeros(9)
        red=comp.plan['redundancy_id']if redundancy is None else redundancy
        self.active=np.r_[np.arange(self.nt,self.n),self.ids.index(red),self.ids.index(comp.plan['master_id'])]
        self.passive=np.setdiff1d(np.arange(self.n),self.active);self.calls=0
        rec={j['id']:j for j in comp.cmg['joints']}
        lo=comp.lower if eliminate else np.array([rec[k]['limits']['lower']for k in comp.cmg['coordinate_ids']])
        hi=comp.upper if eliminate else np.array([rec[k]['limits']['upper']for k in comp.cmg['coordinate_ids']])
        self.lower=np.r_[lo,[-2,-2,-.5,-1.4,-1.4,-3.1]];self.upper=np.r_[hi,[2,2,2,1.4,1.4,3.1]]
        self.evaluator=ToolEvaluator(comp);self.reference_rotation=np.array(comp.source['target_reference_rotation'])
        self.paths=[dict(tool_path=comp.plan['tool_path'],source='generated ancestors')]
        if not eliminate:self.paths.append(dict(affine_coupling=comp.source['affine_couplings'][0]))
    def physical(self,x):return self.L@np.asarray(x)[:self.nt]+self.offset
    def physical_map(self,N):return self.L@N[:self.nt]
    def target_coordinates(self,T):return np.r_[T[:3,3],Rotation.from_matrix(self.reference_rotation.T@T[:3,:3]).as_euler('XYZ')]
    def lift(self,q):
        q=np.asarray(q);T,_=self.evaluator.evaluate(q,False)
        return np.r_[self.comp.reduced(q) if self.eliminate else q,self.target_coordinates(T)]
    augment=lift
    def target_pose(self,x):return target_kinematics(np.asarray(x)[self.nt:],self.reference_rotation)[0]
    def tool_pose(self,x):return self.evaluator.evaluate(self.physical(x),False)[0]
    def _eval(self,x,defects=None,need_jac=True):
        self.calls+=1;q=self.physical(x);T,J=self.evaluator.evaluate(q,need_jac)
        target,K=target_evaluate(np.asarray(x)[self.nt:],self.reference_rotation,need_jac)
        D0=np.eye(4)if defects is None else defects[0];rev=inverse(target@D0);delta=rev@T;r=log(delta)
        A=None
        if need_jac:
            E=np.vstack([J[3:],J[:3]+skew(T[:3,3])@J[3:]])
            Et=np.vstack([K[3:],K[:3]+skew(target[:3,3])@K[3:]])
            A=inv_left(r)@adj(rev)@np.c_[E@self.L,-Et]
        if self.eliminate:return r,A,np.array([delta])
        coupling=np.eye(4);coupling[0,3]=float((-self.comp.C@q)[0]+self.comp.source['affine_couplings'][0]['offset'])
        Dc=np.eye(4)if defects is None else defects[1];d=inverse(Dc)@coupling;rc=log(d)
        if need_jac:
            Ac=np.zeros((6,self.n));Ac[3,:9]=-self.comp.C[0];A=np.vstack([A,Ac])
        return np.r_[r,rc],A,np.array([delta,d])
    def residual(self,x,defects=None):return self._eval(x,defects,True)
    def residual_only(self,x):return self._eval(x,None,False)[0]

class TRFOracle:
    """Combined analytic evaluations and ALL FD residual perturbations counted."""
    def __init__(self,g,active,analytic=True,sparse=False):
        self.g=g;self.active=np.asarray(active);self.analytic=analytic;self.sparse=sparse;self.last=None;self.cache=None;self.evaluations=0
    def full(self,p):
        x=np.zeros(self.g.n);x[self.g.active]=self.active;x[self.g.passive]=p;return x
    def compute(self,p):
        if self.last is None or not np.array_equal(self.last,p):
            self.evaluations+=1;self.last=np.array(p).copy()
            if self.analytic:
                r,J,_=self.g.residual(self.full(p));self.cache=(r,J[:,self.g.passive])
            else:self.cache=(self.g.residual_only(self.full(p)),None)
        return self.cache
    def fun(self,p):return self.compute(p)[0]
    def jac(self,p):
        J=self.compute(p)[1];return csr_matrix(J)if self.sparse else J

def trf_solve(g,active,pred,analytic=True,sparse=False):
    oracle=TRFOracle(g,active,analytic,sparse)
    kwargs={}
    if sparse and not analytic:kwargs['jac_sparsity']=csr_matrix(np.ones((6,len(g.passive)),dtype=int))
    res=least_squares(oracle.fun,pred,jac=oracle.jac if analytic else '3-point',
        bounds=(g.lower[g.passive],g.upper[g.passive]),method='trf',tr_solver='lsmr'if sparse else 'exact',
        ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=1500,x_scale=1.,**kwargs)
    return oracle.full(res.x),dict(success=bool(res.success),iterations=int(res.nfev),evaluations=oracle.evaluations,status=int(res.status))

class Solver:
    def __init__(self,comp,q,method):
        self.comp=comp;self.method=method;self.original=method=='original_pacdm'
        if self.original:
            self.g=TaskGraph(comp.cmg,q,tool_body=comp.cmg['tool']['body'],tool_transform=comp.cmg['tool']['T_body_tool'],reference_rotation=comp.source['target_reference_rotation'],redundancy=comp.plan['redundancy_id'])
            raw=self.g.residual;self.g.calls=0
            def counted(*args,**kwargs):
                self.g.calls+=1
                return raw(*args,**kwargs)
            self.g.residual=counted
        else:self.g=GeneratedTaskGraph(comp,eliminate=method!='generated_full_pacdm')
        self.pac=PACDM(self.g);self.x=self.g.lift(q);self.N,self.info=self.pac.mapping(self.x)
        if not self.info['success']:raise RuntimeError(str(self.info))
        self.predictor=method!='compiled_no_predictor';self.reuse=method in ['compiled_reuse','trf_reuse'];self.is_trf=method.startswith('trf')
    def output(self):
        if self.original:return self.x[:9].copy(),self.N[:9].copy()
        return self.g.physical(self.x),self.g.physical_map(self.N)
    def step(self,active):
        g=self.g;active=np.asarray(active)
        if active.shape!=(8,) or not np.isfinite(active).all():raise ValueError('Expected eight finite task coordinates')
        if np.any(active<g.lower[g.active]) or np.any(active>g.upper[g.active]):raise ValueError('Independent input outside declared bounds')
        if self.reuse and np.array_equal(active[:7],self.x[g.active][:7]):
            self.x[g.active]=active;q,N=self.output()
            return q,N,dict(fallback=False,reused=True,rcond=self.info['rcond'],evaluations=0,iterations=0)
        pred=self.x[g.passive].copy()
        if self.predictor:pred+=self.N[g.passive]@(active-self.x[g.active])
        startcalls=getattr(g,'calls',0);fallback=False
        if self.is_trf:x,ci=trf_solve(g,active,pred)
        else:
            x,ci=self.pac.correct(active,pred,None,np.array(self.info['rows']),maxiter=8)
            if not ci['success']:x,ci=self.pac.acquire(active,self.x);fallback=True
        N,info=self.pac.mapping(x)
        if not ci['success']or not info['success']:raise RuntimeError(str((ci,info)))
        self.x,self.N,self.info=x,N,info;q,Np=self.output()
        return q,Np,dict(fallback=fallback,reused=False,rcond=info['rcond'],evaluations=ci.get('evaluations',getattr(g,'calls',0)-startcalls),iterations=ci.get('iterations',0))

METHODS=['original_pacdm','generated_full_pacdm','compiled_pacdm','compiled_reuse','compiled_no_predictor','trf_compiled','trf_reuse']
LABELS=dict(original_pacdm='Original global PACDM',generated_full_pacdm='Generated evaluator, explicit coupling',
    compiled_pacdm='Compiled affine-reduced PACDM',compiled_reuse='Compiled PACDM with exact reuse',
    compiled_no_predictor='Compiled PACDM without predictor',trf_compiled='Analytical TRF, same compiled model and predictor',
    trf_reuse='Analytical TRF, compiled model, predictor and reuse')
