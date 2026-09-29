"""Slow dig/lift/slew/dump reference. Only first six angles are controlled."""
import numpy as np
from scipy.optimize import least_squares
from benchmark_model import GRAVITY
LIP=np.array([-.108,-1.96733,1.44613]);HEEL=np.array([-.108,-2.13268,.45688])
STAGES=['settle','lower','draw through material','curl','lift','slew','dump','settle material']
def tip_pose(engine,q):
 T=engine.source.evaluate(q,np.zeros(23),GRAVITY)['poses']['body_56'];p=T[:3,:3]@LIP+T[:3,3];h=T[:3,:3]@HEEL+T[:3,3]
 return np.array([p[0],p[2],np.arctan2(p[2]-h[2],p[0]-h[0])])
def make_path(engine,reference):
 base=reference.nominal.copy();active=np.array(engine.active);nom=base[active]
 targets=[[7.4,.8,-1.5],[7.3,.025,-1.7],[6.5,.025,-1.85],[5.8,.2,-2.25],[5.2,1.3,-2.3],[5.2,1.3,-2.3],[7.6,1.5,-.90]]
 knots=[];seed=nom[1:4].copy()
 for target in targets:
  def fun(x):
   q=base.copy();q[active[1:4]]=x;return tip_pose(engine,q)-target
  fit=least_squares(fun,seed,bounds=(nom[1:4]-1.49,nom[1:4]+1.49),xtol=1e-13,ftol=1e-13,gtol=1e-13)
  if np.max(abs(fun(fit.x)))>1e-7:raise ValueError('Unreachable waypoint')
  u=nom.copy();u[1:4]=fit.x;knots.append(u);seed=fit.x
 knots[5][0]+=.36;knots[5][4]+=.025;knots[5][5]+=.05;knots[6][0]+=.36
 times=np.array([0.,1.,3.,4.5,6.,8.,10.,12.,14.]);u=np.array([knots[0],knots[0],knots[1],knots[2],knots[3],knots[4],knots[5],knots[6],knots[6]])
 return dict(times=times,independent=u,closed_tree=np.array([reference.reconstruct(x) for x in u]),tip_targets=targets)
def reference_at(t,path):
 ts=path['times'];us=path['independent'];k=min(max(int(np.searchsorted(ts,t,side='right')-1),0),len(ts)-2)
 dur=ts[k+1]-ts[k];s=np.clip((t-ts[k])/dur,0,1);delta=us[k+1]-us[k]
 f=10*s**3-15*s**4+6*s**5;fd=(30*s**2-60*s**3+30*s**4)/dur;fdd=(60*s-180*s**2+120*s**3)/dur**2
 q=us[k]+f*delta;v=fd*delta;acc=fdd*delta
 if k in (2,3):
  height=-.020 if k==2 else -.008
  q[1]+=height*64*(s**3-3*s**4+3*s**5-s**6)
  v[1]+=height*64*(3*s**2-12*s**3+15*s**4-6*s**5)/dur
  acc[1]+=height*64*(6*s-36*s**2+60*s**3-30*s**4)/dur**2
 return q,v,acc
def stage_at(t,path):return STAGES[min(max(int(np.searchsorted(path['times'],t,side='right')-1),0),len(STAGES)-1)]
