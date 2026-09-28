"""Independent classical rigid-body checks and ideal-support dynamics.

Rigid body inertias only. Motor rotor/armature, friction, damping, contact
compliance, impacts and unilateral inequalities are intentionally excluded on
both compared routes. Twelve motor efforts, no fictitious base actuators.
"""
from __future__ import annotations
import numpy as np
from . import bootstrap
from .numpy_reference_base import NumpyTree,cross_matrix
from .task_graph import CompiledStanceGraph
from vendor.pacdm_original import PACDM,rank


def normmax(x):return float(np.max(np.abs(x))) if np.size(x) else 0.
def relative(x,y):return normmax(np.asarray(x)-np.asarray(y))/max(1.,normmax(y))


class PhysicalReference(NumpyTree):
    def feet_geometry(self,q,v=None):
        states=self.forward(q,v,acceleration=v is not None);p=[];J=[];gamma=[]
        for f in self.cmg['feet']:
            b=states[f['body']];r=b['R']@np.asarray(f['point_m'])
            p.append(b['p']+r);J.append(b['Jv']-cross_matrix(r)@b['Jw'])
            gamma.append(b['a']+np.cross(b['alpha'],r)+np.cross(b['w'],np.cross(b['w'],r)))
        return dict(points=np.array(p),jacobian=np.array(J),gamma=np.array(gamma),states=states)

    def effort(self,q,motors,wrench):
        B=np.asarray(self.cmg['actuation']['moment_matrix']);tau=B@np.asarray(motors)
        states=self.forward(q);b=states[self.cmg['physical_root']]
        J=np.vstack((b['Jv'],b['Jw']))
        return tau+J.T@np.asarray(wrench),tau


def mapping_curvature(graph,q,active_velocity):
    N,info=PACDM(graph).mapping(q)
    if not info['success']:raise RuntimeError(info)
    v=N@np.asarray(active_velocity);eps=1e-5/max(1.,normmax(v))
    J=graph.residual(q)[1];Jp=graph.residual(q+eps*v)[1];Jm=graph.residual(q-eps*v)[1]
    gamma=((Jp-Jm)/(2*eps))@v
    c=np.zeros(graph.n);rows=np.array(info['rows'],int)
    c[graph.passive]=-np.linalg.solve(J[np.ix_(rows,graph.passive)],gamma[rows])
    return N,v,c,info


def dynamics_witness(comp,q,sites,va,motors,wrench,native=False):
    ref=PhysicalReference(comp.cmg);g=CompiledStanceGraph(comp,q,sites)
    N,v,c,info=mapping_curvature(g,q,va);geom=ref.feet_geometry(q,v)
    J=geom['jacobian'][sites].reshape(-1,18);gamma=geom['gamma'][sites].ravel()
    M,b,U=ref.mass_bias(q,v);tau,motor_tau=ref.effort(q,motors,wrench)
    Mr=N.T@M@N;aa=np.linalg.solve(Mr,N.T@(tau-b-M@c));a=N@aa+c
    A=np.block([[M,-J.T],[J,np.zeros((len(J),len(J)))]])
    sol=np.linalg.solve(A,np.r_[tau-b,-gamma]);ak=sol[:18];lam=sol[18:]
    # Independent support tangent using the physical point Jacobian, not PACDM rows.
    Ni=np.zeros_like(N);Ni[g.active]=np.eye(len(g.active));Ni[g.passive]=-np.linalg.solve(J[:,g.passive],J[:,g.active])
    data=dict(relative_acceleration_difference=relative(a,ak),absolute_acceleration_difference=normmax(a-ak),
        point_acceleration_residual_m_s2=normmax(J@a+gamma),tangent_residual=normmax(J@N),
        mapping_discrepancy=normmax(N-Ni),virtual_power_defect_W=abs(float(tau@v-(N.T@tau)@va)),
        reduced_inertia_relative=relative(Mr,Ni.T@M@Ni),min_reduced_inertia_eigenvalue=float(np.linalg.eigvalsh(Mr).min()),
        kkt_force_balance=normmax(M@ak+b-tau-J.T@lam),kinematic_curvature_constraint=normmax(J@c+gamma),
        base_motor_effort_max=normmax(motor_tau[:6]),rank=int(rank(J)),mobility=len(g.active),
        rcond=info['rcond'],native_acceleration_relative=None,native_mass_inf=None,native_bias_inf=None,
        native_point_jacobian_inf=None,native_point_bias_inf=None)
    native_version=None
    if native:
        from .native import NativeSupportOracle
        oracle=NativeSupportOracle(comp,q,sites);an=oracle.acceleration(q,v,tau)
        native_geom=oracle.geometry(q,v)
        data.update(native_acceleration_relative=relative(a,an),native_mass_inf=normmax(oracle.backend.mass(q)-M),
            native_bias_inf=normmax(oracle.backend.bias(q,v)-b),native_point_jacobian_inf=normmax(native_geom['J']-J),
            native_point_bias_inf=normmax(native_geom['gamma']-gamma))
        native_version=oracle.version
    good=(data['relative_acceleration_difference']<1e-8 and data['point_acceleration_residual_m_s2']<2e-6
          and data['mapping_discrepancy']<1e-9 and data['virtual_power_defect_W']<1e-9
          and data['reduced_inertia_relative']<1e-9 and data['min_reduced_inertia_eigenvalue']>0
          and data['base_motor_effort_max']==0 and data['rank']==3*len(sites))
    if native:
        good=good and data['native_acceleration_relative']<1e-7 and data['native_mass_inf']<1e-8 and data['native_bias_inf']<1e-7 and data['native_point_jacobian_inf']<1e-9 and data['native_point_bias_inf']<1e-8
    data['success']=bool(good)
    return data,dict(q=q,sites=sites,active_velocity=va,motors=motors,wrench=wrench,v=v,acceleration=a,
        independent_ids=[comp.cmg['coordinate_ids'][i]for i in g.active],native_version=native_version)
