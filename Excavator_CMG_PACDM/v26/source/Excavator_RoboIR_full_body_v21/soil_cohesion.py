"""Breakable tensile bonds between initially neighbouring coarse soil grains.
A declared weak-clod surrogate: no adhesion to the bucket and no prescribed
particle motion. Equal/opposite COM forces; fracture energy is accounted for.
"""
import numpy as np
PARAMETERS={'stiffness_N_m':8000.,'break_extension_m':.03,'neighbour_distance_radii':2.025,
 'provenance':'Synthetic weak-clod cohesion; no measured soil calibration'}
class CohesiveBed:
 def __init__(self,positions,radius,enabled=True):
  p=np.asarray(positions);self.pairs=np.array([(i,j) for i in range(len(p)) for j in range(i) if np.linalg.norm(p[i]-p[j])<=PARAMETERS['neighbour_distance_radii']*radius],int).reshape(-1,2)
  self.rest=np.linalg.norm(p[self.pairs[:,0]]-p[self.pairs[:,1]],axis=1);self.active=np.full(len(self.pairs),enabled,bool);self.fracture_J=0.
 def evaluate(self,p,v):
  p=np.asarray(p);force=np.zeros_like(p);pairs=self.pairs
  if not len(pairs):return force,0.,0.
  dr=p[pairs[:,1]]-p[pairs[:,0]];length=np.linalg.norm(dr,axis=1);extension=np.maximum(length-self.rest,0);k=PARAMETERS['stiffness_N_m']
  broken=self.active&(extension>PARAMETERS['break_extension_m']);self.fracture_J+=float(.5*k*np.sum(extension[broken]**2));self.active[broken]=False
  f=k*(extension*self.active)[:,None]*dr/np.maximum(length[:,None],1e-12);np.add.at(force,pairs[:,0],f);np.add.at(force,pairs[:,1],-f)
  energy=float(.5*k*np.sum((extension*self.active)**2));power=float(np.sum(force*v))
  return force,energy,power
