"""Exact v22 CutGraph class, isolated from MuJoCo/Pinocchio imports."""
import numpy as np
from scipy.spatial.transform import Rotation
from .pacdm import PointGraph, exp, log, inv, adj, inv_left

class CutGraph(PointGraph):
    def __init__(self,c,seed):
        super().__init__(c,seed)
        self.n=self.nt+sum(2 if x['type']=='universal' else 3 for x in self.cuts)
        self.passive=np.setdiff1d(np.arange(self.n),self.active)
        self.lower=self.lower[:self.n];self.upper=self.upper[:self.n]
        self.frames=[];self.chart_columns=[];at=self.nt
        for k,cut in enumerate(self.cuts):
            A=np.eye(4);B=np.eye(4);A[:3,3]=cut['point1_m'];B[:3,3]=cut['point2_m']
            if cut['type']=='universal':
                A[:3,:3]=cut['frame1_R'];B[:3,:3]=cut['frame2_R'];n=2
            else:A[:3,:3]=self.references[k];n=3
            self.frames.append((A,B));self.chart_columns.append(list(range(at,at+n)));at+=n
        for cut,path,cols in zip(self.cuts,self.paths,self.chart_columns):
            path['cut_type']=cut['type'];path['chart_columns']=cols
    def augment(self,q):return np.r_[q,np.zeros(self.n-self.nt)]
    def lift(self,q):
        out=self.augment(q);P,_=self.poses(out)
        for cut,(A,B),cols in zip(self.cuts,self.frames,self.chart_columns):
            relative=(P[cut['body1']]@A)[:3,:3].T@(P[cut['body2']]@B)[:3,:3]
            out[cols]=Rotation.from_matrix(relative).as_euler('XYZ')[:len(cols)]
        return out
    def residual(self,q,defects=None):
        P,E=self.poses(q);rs=[];js=[];deltas=[]
        for k,(cut,(A,B),cols) in enumerate(zip(self.cuts,self.frames,self.chart_columns)):
            minus=P[cut['body1']]@A;eta=E[cut['body1']].copy()
            for i,axis in zip(cols,np.eye(3)):
                twist=np.r_[axis,[0,0,0]];eta[:,i]+=adj(minus)@twist;minus=minus@exp(twist*q[i])
            plus=P[cut['body2']]@B;D=np.eye(4) if defects is None else defects[k]
            reverse=inv(minus@D);delta=reverse@plus;r=log(delta)
            rs.append(r);js.append(inv_left(r)@adj(reverse)@(E[cut['body2']]-eta));deltas.append(delta)
        return np.concatenate(rs),np.vstack(js),np.asarray(deltas)
