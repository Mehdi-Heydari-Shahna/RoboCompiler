"""Revolute-cut source adapter for the unchanged Kangaroo/excavator PACDM core."""
import numpy as np
from pacdm import inv,adj,exp,log,inv_left,PACDM
class RevoluteGraph:
 def __init__(self,cmg,mapping,tree_ids):
  self.cmg=cmg;self.nt=len(tree_ids);self.ids=list(tree_ids)+list(mapping['cut_joint_ids']);self.n=len(self.ids);self.records={j['id']:j for j in cmg['joints']}
  ref=next(r for r in cmg['reference_poses'] if r['id']=='working_reconstruction');val=dict(zip(ref['coordinate_ids'],ref['q_SI']));self.seed=np.array([val[x] for x in self.ids])
  self.active=np.array([self.ids.index(x) for x in ['q23','q7','q4','q0','q1','q21','q22']]);self.passive=np.setdiff1d(np.arange(self.n),self.active)
  spans=np.array([1.8 if self.records[x]['type']=='prismatic' else 3. for x in self.ids]);self.lower=self.seed-spans;self.upper=self.seed+spans;self.root=cmg['root_body'];self.edges=[]
  for edge in mapping['tree']:
   j=self.records[edge['id']];rev=edge['direction']==-1;enter=np.array(j['T_FJ'] if rev else j['T_BJ']);leave=inv(np.array(j['T_BJ'] if rev else j['T_FJ']));axis=np.asarray(j['axis'])*edge['direction']
   twist=np.r_[axis,[0,0,0]] if j['type']=='revolute' else np.r_[[0,0,0],axis];k=None if j['type']=='fixed' else self.ids.index(j['id']);self.edges.append((edge['parent_body'],edge['child_body'],enter,leave,twist,k))
  self.cuts=[self.records[x] for x in mapping['cut_joint_ids']]
 def poses(self,q):
  P={self.root:np.eye(4)};E={self.root:np.zeros((6,self.n))}
  for parent,child,enter,leave,twist,k in self.edges:
   prefix=P[parent]@enter;eta=E[parent].copy();motion=np.eye(4)
   if k is not None:motion=exp(twist*q[k]);eta[:,k]+=adj(prefix)@twist
   P[child]=prefix@motion@leave;E[child]=eta
  return P,E
 def lift(self,q):
  out=self.seed.copy();out[:self.nt]=q;P,_=self.poses(out)
  for i,j in enumerate(self.cuts):
   A=P[j['base_body']]@np.asarray(j['T_BJ']);B=P[j['follower_body']]@np.asarray(j['T_FJ']);angle=float(np.asarray(j['axis'])@log(inv(A)@B)[:3]);nom=self.seed[self.nt+i];out[self.nt+i]=angle+2*np.pi*np.round((nom-angle)/(2*np.pi))
  return out
 def residual(self,q,defects=None):
  P,E=self.poses(q);rs=[];js=[];deltas=[]
  for i,j in enumerate(self.cuts):
   base,child=j['base_body'],j['follower_body'];k=self.nt+i;twist=np.r_[j['axis'],[0,0,0]];prefix=P[base]@np.asarray(j['T_BJ']);minus=prefix@exp(twist*q[k]);em=E[base].copy();em[:,k]+=adj(prefix)@twist
   plus=P[child]@np.asarray(j['T_FJ']);D=np.eye(4) if defects is None else defects[i];reverse=inv(minus@D);delta=reverse@plus;r=log(delta);rs.append(r);js.append(inv_left(r)@adj(reverse)@(E[child]-em));deltas.append(delta)
  return np.concatenate(rs),np.vstack(js),np.asarray(deltas)
def compile_pacdm(cmg,mapping,tree_ids):
 g=RevoluteGraph(cmg,mapping,tree_ids);return g,PACDM(g)
