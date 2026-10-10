"""Generated pruned point evaluators and conditional PACDM task scheduling.

All nonlinear PACDM correct/acquire/map calls use the original unchanged core.
No inverse-kinematics formula or learned/precomputed solution is used.
Caching is exact: only an exactly unchanged independent block can be reused.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from . import bootstrap
from vendor.pacdm_original import PACDM, skew
from go2.contact import ContactGraph, _translation_residual


def base_pose(q):
    yaw,pitch,roll=np.asarray(q)[3:6]
    cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch);cr,sr=np.cos(roll),np.sin(roll)
    rz=np.array([[cy,-sy,0.],[sy,cy,0.],[0.,0.,1.]])
    ry=np.array([[cp,0.,sp],[0.,1.,0.],[-sp,0.,cp]])
    rx=np.array([[1.,0.,0.],[0.,cr,-sr],[0.,sr,cr]])
    return rz@ry@rx,np.asarray(q)[:3],np.column_stack(([0.,0.,1.],rz[:,1],(rz@ry)[:,0]))


class BranchEvaluator:
    """Root-relative FK and analytic point differential derived from joint path."""
    def __init__(self,comp,index):
        self.comp=comp;self.index=index
        self.path=comp.plan['point_paths'][index];self.dep=np.array(self.path['dependent_indices'],int)
        self.point=np.array(comp.cmg['feet'][index]['point_m']);records={j['id']:j for j in comp.cmg['joints']}
        self.edges=[]
        for name in self.path['joint_path']:
            j=records[name];E=np.array(j['T_BJ']);F=np.array(j['T_FJ']);L=np.linalg.inv(F)
            axis=np.array(j.get('axis',[0,0,1]),float);K=skew(axis)
            k=None if j['type']=='fixed' else list(self.dep).index(comp.cmg['coordinate_ids'].index(name))
            self.edges.append((j['type'],E,L,axis,K,K@K,k))
        self._base=None;self._prefix=None;self.evaluations=0

    def point_jacobian(self,base,joints,need_jac=True):
        self.evaluations+=1
        if self._base is None or not np.array_equal(self._base,base):
            self._base=np.asarray(base).copy();self._prefix=base_pose(base)
        rb,pb,abase=self._prefix;R=rb.copy();p=pb.copy();records=[]
        for kind,E,L,axis,K,K2,k in self.edges:
            p=p+R@E[:3,3];R=R@E[:3,:3]
            if k is not None:
                aw=R@axis
                if need_jac:records.append((kind,k,aw,p.copy()))
                if kind=='revolute':
                    v=joints[k];R=R@(np.eye(3)+np.sin(v)*K+(1-np.cos(v))*K2)
                else:p=p+aw*joints[k]
            p=p+R@L[:3,3];R=R@L[:3,:3]
        point=p+R@self.point
        if not need_jac:return point,None
        J=np.zeros((3,6+len(self.dep)));J[:,:3]=np.eye(3);J[:,3:6]=-skew(point-pb)@abase
        for kind,k,aw,origin in records:
            J[:,6+k]=np.cross(aw,point-origin) if kind=='revolute' else aw
        return point,J


class GeneratedTaskGraph:
    def __init__(self,comp):
        self.comp=comp;self.cmg=comp.cmg;self.nt=18;self.n=30;self.nc=4
        self.ids=list(comp.cmg['coordinate_ids'])+[f"target_{s['id']}_{a}" for s in comp.cmg['feet'] for a in 'xyz']
        self.active=np.r_[np.arange(6),np.arange(18,30)];self.passive=np.arange(6,18)
        records={j['id']:j for j in comp.cmg['joints']}
        self.lower=np.r_[[records[k]['limits']['lower']for k in self.ids[:18]],np.full(12,-np.inf)]
        self.upper=np.r_[[records[k]['limits']['upper']for k in self.ids[:18]],np.full(12,np.inf)]
        self.branches=[BranchEvaluator(comp,i)for i in range(4)];self.calls=0
        self.paths=comp.plan['point_paths']

    def points_jacobian(self,q,need_jac=True):
        points=[];J=np.zeros((4,3,18)) if need_jac else None
        for i,b in enumerate(self.branches):
            p,j=b.point_jacobian(q[:6],q[b.dep],need_jac);points.append(p)
            if need_jac:
                J[i,:,:6]=j[:,:6];J[i][:,b.dep]=j[:,6:]
        return np.array(points),J

    def lift(self,q):return np.r_[q[:18],self.points_jacobian(q,False)[0].ravel()]
    augment=lift

    def residual(self,x,defects=None):
        self.calls+=1;p,J=self.points_jacobian(x[:18]);A=np.pad(J,((0,0),(0,0),(0,12)))
        for i in range(4):A[i,:,18+3*i:21+3*i]=-np.eye(3)
        return _translation_residual(p-x[18:].reshape(4,3),A,self.n,defects)

    def residual_only(self,x):
        self.calls+=1;p,_=self.points_jacobian(x[:18],False);r=np.zeros((4,6));r[:,3:]=p-x[18:].reshape(4,3)
        return r.ravel()


class ModuleGraph:
    def __init__(self,comp,index):
        self.branch=BranchEvaluator(comp,index);self.dep=self.branch.dep;self.index=index
        self.nt=9;self.n=12;self.nc=1;self.active=np.r_[np.arange(6),np.arange(9,12)];self.passive=np.arange(6,9)
        globalgraph=GeneratedTaskGraph(comp)
        self.lower=np.r_[globalgraph.lower[:6],globalgraph.lower[self.dep],np.full(3,-np.inf)]
        self.upper=np.r_[globalgraph.upper[:6],globalgraph.upper[self.dep],np.full(3,np.inf)]
        self.calls=0

    def residual(self,x,defects=None):
        self.calls+=1;p,J=self.branch.point_jacobian(x[:6],x[6:9]);A=np.c_[J,-np.eye(3)]
        return _translation_residual((p-x[9:12])[None,:],A[None,:,:],self.n,defects)


class CompiledStanceGraph:
    def __init__(self,comp,q,sites):
        self.comp=comp;self.task=GeneratedTaskGraph(comp);self.plan=comp.support_plan(sites)
        self.ids=comp.cmg['coordinate_ids'];self.nt=self.n=18;self.nc=len(sites);self.sites=np.array(sites,int)
        self.active=np.array(self.plan['active'],int);self.passive=np.array(self.plan['passive'],int)
        self.lower=self.task.lower[:18];self.upper=self.task.upper[:18]
        self.anchors=self.task.points_jacobian(q,False)[0][self.sites].copy()
        self.paths=[comp.plan['point_paths'][i]for i in sites]
    def lift(self,q):return np.asarray(q).copy()
    augment=lift
    def residual(self,q,defects=None):
        p,J=self.task.points_jacobian(q)
        return _translation_residual(p[self.sites]-self.anchors,J[self.sites],18,defects)


class GlobalSolver:
    """Monolithic assembly. Original and generated evaluator routes are separate."""
    def __init__(self,comp,q,kind='compiled',predictor=True):
        self.g=ContactGraph(comp.cmg,q) if kind=='original' else GeneratedTaskGraph(comp)
        self.solver=PACDM(self.g);self.kind=kind;self.predictor=predictor
        self.x=self.g.lift(q);self.N,self.info=self.solver.mapping(self.x)
        if not self.info['success']:raise RuntimeError(self.info)

    def step(self,active):
        g=self.g;pred=self.x[g.passive].copy()
        if self.predictor:pred+=self.N[g.passive]@(active-self.x[g.active])
        fallback=False
        if self.kind=='trf':
            oracle=TRFOracle(g,active)
            res=least_squares(oracle.fun,pred,jac=oracle.jac,bounds=(g.lower[g.passive],g.upper[g.passive]),
                method='trf',tr_solver='exact',ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=1500,x_scale=1.)
            x=oracle.full(res.x);ci=dict(success=bool(res.success),iterations=res.nfev)
        else:
            x,ci=self.solver.correct(active,pred,None,np.array(self.info['rows']),maxiter=8)
            if not ci['success']:
                x,ci=self.solver.acquire(active,self.x);fallback=True
        N,info=self.solver.mapping(x)
        if not ci['success'] or not info['success']:raise RuntimeError(str((ci,info)))
        self.x,self.N,self.info=x,N,info
        return x[:18].copy(),N[:18].copy(),dict(fallback=fallback,solved_modules=1,skipped_modules=0,
            rcond=info['rcond'],tangent_residual=info['tangent_residual'])


class ModularSolver:
    def __init__(self,comp,q,predictor=True,skip_unchanged=True,corrector='pacdm'):
        self.comp=comp;self.predictor=predictor;self.skip_unchanged=skip_unchanged;self.corrector=corrector
        self.modules=[]
        for i in range(4):
            g=ModuleGraph(comp,i);p,_=g.branch.point_jacobian(q[:6],q[g.dep],False)
            x=np.r_[q[:6],q[g.dep],p];solver=PACDM(g);N,info=solver.mapping(x)
            if not info['success']:raise RuntimeError(info)
            self.modules.append(dict(g=g,solver=solver,x=x,N=N,info=info))

    def step(self,active):
        q=np.zeros(18);q[:6]=active[:6];N=np.zeros((18,18));N[:6,:6]=np.eye(6)
        solved=skipped=0;fallback=False;rc=1.;tr=0.
        for i,m in enumerate(self.modules):
            g=m['g'];cols=np.r_[np.arange(6),np.arange(6+3*i,9+3*i)];qa=active[cols]
            if self.skip_unchanged and np.array_equal(qa,m['x'][g.active]):
                skipped+=1
            else:
                solved+=1;pred=m['x'][g.passive].copy()
                if self.predictor:pred+=m['N'][g.passive]@(qa-m['x'][g.active])
                if self.corrector=='trf':
                    oracle=TRFOracle(g,qa)
                    res=least_squares(oracle.fun,pred,jac=oracle.jac,method='trf',tr_solver='exact',
                        bounds=(g.lower[g.passive],g.upper[g.passive]),ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=1500,x_scale=1.)
                    x=oracle.full(res.x);ci=dict(success=bool(res.success))
                else:
                    x,ci=m['solver'].correct(qa,pred,None,np.array(m['info']['rows']),maxiter=8)
                    if not ci['success']:x,ci=m['solver'].acquire(qa,m['x']);fallback=True
                nm,info=m['solver'].mapping(x)
                if not ci['success'] or not info['success']:raise RuntimeError(str((ci,info)))
                m.update(x=x,N=nm,info=info)
            q[g.dep]=m['x'][g.passive];N[np.ix_(g.dep,cols)]=m['N'][g.passive]
            rc=min(rc,m['info']['rcond']);tr=max(tr,m['info']['tangent_residual'])
        return q,N,dict(fallback=fallback,solved_modules=solved,skipped_modules=skipped,rcond=rc,tangent_residual=tr)


class TRFOracle:
    """Count EVERY fresh evaluation, including numerical perturbations.

    An analytic (residual, Jacobian) pair is computed once at an identical x;
    calls to fun and jac share it. FD fun never calculates an analytic Jacobian.
    """
    def __init__(self,graph,active,analytic=True,sparse=False):
        self.g=graph;self.active=np.asarray(active).copy();self.analytic=analytic;self.sparse=sparse
        self.last=None;self.r=None;self.J=None;self.evaluations=0
    def full(self,p):
        x=np.zeros(self.g.n);x[self.g.active]=self.active;x[self.g.passive]=p;return x
    def _evaluate(self,p):
        if self.last is not None and np.array_equal(p,self.last):return
        if self.evaluations>=1500:raise RuntimeError('Common fresh-evaluation budget exceeded')
        self.last=np.asarray(p).copy();self.evaluations+=1;x=self.full(p)
        if self.analytic:
            self.r,j,_=self.g.residual(x);self.J=j[:,self.g.passive]
        else:self.r=self.g.residual_only(x);self.J=None
    def fun(self,p):self._evaluate(p);return self.r.copy()
    def jac(self,p):
        self._evaluate(p)
        return csr_matrix(self.J) if self.sparse else self.J.copy()


def create_solver(name,comp,q):
    if name=='original_monolithic':return GlobalSolver(comp,q,'original')
    if name=='compiled_monolithic':return GlobalSolver(comp,q,'compiled')
    if name=='compiled_modular':return ModularSolver(comp,q)
    if name=='modular_no_predictor':return ModularSolver(comp,q,predictor=False)
    if name=='modular_no_reuse':return ModularSolver(comp,q,skip_unchanged=False)
    if name=='trf_compiled_predictor':return GlobalSolver(comp,q,'trf')
    if name=='trf_modular_predictor':return ModularSolver(comp,q,corrector='trf')
    raise ValueError(name)
