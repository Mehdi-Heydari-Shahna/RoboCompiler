"""Independent classical physical reference and task-coordinate dynamics.

Franka arm is a tree. The only imposed physical constraint is the declared
finger equality. A virtual tool task is NOT welded to ground in these dynamics:
all eight task-coordinate accelerations are solved from applied physical effort.
Source diagonal armature and viscous damping are included on all routes.
"""
from __future__ import annotations
import numpy as np
from . import bootstrap
from .numpy_reference import NumpyTree,cross_matrix
from .task_graph import GeneratedTaskGraph,target_kinematics
from vendor.pacdm_original import PACDM


def mx(x):return float(np.max(np.abs(x))) if np.size(x)else 0.
def relative(x,y):return mx(np.asarray(x)-np.asarray(y))/max(1.,mx(y))

class PhysicalReference(NumpyTree):
    def tool(self,q,v=None,a=None):
        states=self.forward(q,v,acceleration=v is not None);b=states[self.cmg['tool']['body']];H=np.array(self.cmg['tool']['T_body_tool']);r=b['R']@H[:3,3]
        T=np.eye(4);T[:3,:3]=b['R']@H[:3,:3];T[:3,3]=b['p']+r
        J=np.vstack([b['Jv']-cross_matrix(r)@b['Jw'],b['Jw']]);gamma=np.r_[b['a']+np.cross(b['alpha'],r)+np.cross(b['w'],np.cross(b['w'],r)),b['alpha']]
        acc=gamma if a is None else J@a+gamma
        return dict(T=T,J=J,gamma=gamma,acceleration=acc,states=states)

def independent_map(comp,q,active,red=None):
    ref=PhysicalReference(comp.cmg);t=ref.tool(q);ids=comp.cmg['coordinate_ids'];rid=comp.plan['redundancy_id']if red is None else red
    K=target_kinematics(active[:6],np.array(comp.source['target_reference_rotation']))[1]
    C=np.zeros((1,9));rec=comp.source['affine_couplings'][0];C[0,ids.index(rec['slave'])]=1;C[0,ids.index(rec['master'])]=-rec['multiplier']
    E=np.zeros((2,9));E[0,ids.index(rid)]=1;E[1,ids.index(rec['master'])]=1
    A=np.vstack([t['J'],E,C]);R=np.zeros((9,8));R[:6,:6]=K;R[6:8,6:8]=np.eye(2)
    return np.linalg.solve(A,R),A

def validate(comp,q,N,active):
    ref=PhysicalReference(comp.cmg);tool=ref.tool(q);T,_,_=target_kinematics(active[:6],np.array(comp.source['target_reference_rotation']))
    gap=float(np.linalg.norm(tool['T'][:3,3]-T[:3,3]));from scipy.spatial.transform import Rotation
    ori=float(np.linalg.norm(Rotation.from_matrix(T[:3,:3].T@tool['T'][:3,:3]).as_rotvec()))
    coupling=float(abs((comp.C@q)[0]-comp.source['affine_couplings'][0]['offset']))
    ids=comp.cmg['coordinate_ids'];rec={j['id']:j for j in comp.cmg['joints']};lo=np.array([rec[k]['limits']['lower']for k in ids]);hi=np.array([rec[k]['limits']['upper']for k in ids])
    bounds=float(min(np.min(q-lo),np.min(hi-q)));ad=max(abs(q[ids.index(comp.plan['redundancy_id'])]-active[6]),abs(q[comp.plan['master_index']]-active[7]))
    Ni,A=independent_map(comp,q,active);mapping=mx(N-Ni);rc=1/np.linalg.cond(A,p=1)
    success=(np.isfinite(q).all()and np.isfinite(N).all()and gap<=1e-8 and ori<=1e-8 and coupling<=1e-8 and ad<=1e-12 and bounds>=-1e-12 and rc>=1e-10 and mapping<2e-7)
    return dict(success=bool(success),max_gap_m=gap,orientation_error_rad=ori,coupling_error_m=coupling,
        active_error=ad,bound_margin=bounds,independent_rcond=float(rc),mapping_discrepancy=mapping)

def curvature(g,x,va):
    N,info=PACDM(g).mapping(x)
    if not info['success']:raise ValueError(str(info))
    v=N@np.asarray(va);eps=1e-5/max(1.,mx(v));Jdot=(g.residual(x+eps*v)[1]-g.residual(x-eps*v)[1])/(2*eps)
    J=g.residual(x)[1];rows=np.array(info['rows'],int);c=np.zeros(g.n)
    c[g.passive]=-np.linalg.solve(J[np.ix_(rows,g.passive)],(Jdot@v)[rows])
    return N,v,c,info

def dynamics_witness(comp,q,va,u,wrench,native=False):
    g=GeneratedTaskGraph(comp);x=g.lift(q);N,v,c,info=curvature(g,x,va)
    Np=g.physical_map(N);vp=g.L@v[:g.nt];cp=g.L@c[:g.nt];q=g.physical(x)
    ref=PhysicalReference(comp.cmg);G=ref.tool(q,vp);Mr,h,U=ref.mass_bias(q,vp)
    arm=np.array(comp.cmg['armature']);damp=np.array(comp.cmg['damping']);M=Mr+np.diag(arm);bias=h+damp*vp
    B=np.array(comp.cmg['actuation']['moment_matrix']);tau=B@u+G['J'].T@wrench
    mass=Np.T@M@Np;aa=np.linalg.solve(mass,Np.T@(tau-bias-M@cp));a=Np@aa+cp
    C=comp.C;KKT=np.block([[M,-C.T],[C,np.zeros((1,1))]]);sol=np.linalg.solve(KKT,np.r_[tau-bias,0.]);ak=sol[:9];lam=sol[9:]
    S=comp.S;as_=S@np.linalg.solve(S.T@M@S,S.T@(tau-bias))
    Ni,_=independent_map(comp,q,x[g.active]);target=target_kinematics(x[g.nt:],g.reference_rotation,va[:6],aa[:6]);toolacc=ref.tool(q,vp,a)['acceleration']
    rec=dict(success=False,relative_acceleration_difference=relative(a,ak),absolute_acceleration_difference=mx(a-ak),
        affine_vs_kkt_relative=relative(as_,ak),finger_acceleration_residual_m_s2=mx(C@a),
        tool_acceleration_consistency_mixed=mx(toolacc-target[2]),physical_mapping_discrepancy=mx(Np-Ni),
        virtual_power_defect_W=abs(float(tau@vp-(Np.T@tau)@va)),affine_power_defect_W=abs(float(tau@vp-(S.T@tau)@(v[:g.nt]))),
        reduced_inertia_relative=relative(mass,Ni.T@M@Ni),physical_reduced_mass_min_eig=float(np.linalg.eigvalsh(S.T@M@S).min()),
        task_reduced_mass_min_eig=float(np.linalg.eigvalsh(mass).min()),kkt_force_balance=mx(M@ak+bias-tau-C.T@lam),
        physical_coupling_rank=int(np.linalg.matrix_rank(C)),physical_mobility=8,rcond=info['rcond'],
        native_kkt_acceleration_relative=None,native_mass_inf=None,native_bias_inf=None,native_tool_jacobian_inf=None,
        native_tool_gamma_inf=None,native_task_acceleration_mixed=None,native_rnea_inf=None,native_aba_roundtrip=None)
    version=None
    if native:
        from .native import NativeOracle
        oracle=NativeOracle(comp);Mn=oracle.backend.mass(q);hn=oracle.backend.bias(q,vp);Gn=oracle.tool(q,vp)
        An=np.block([[Mn+np.diag(arm),-C.T],[C,np.zeros((1,1))]])
        # Scalar coupling is imposed by an explicitly assembled KKT solve on
        # native matrices. This does NOT call Pinocchio constraintDynamics.
        taun=B@u+Gn['J'].T@wrench;an=np.linalg.solve(An,np.r_[taun-hn-damp*vp,0.])[:9]
        rec.update(native_kkt_acceleration_relative=relative(a,an),native_mass_inf=mx(Mn-Mr),native_bias_inf=mx(hn-h),
            native_tool_jacobian_inf=mx(Gn['J']-G['J']),native_tool_gamma_inf=mx(Gn['gamma']-G['gamma']),
            native_task_acceleration_mixed=mx(oracle.tool(q,vp,a)['acceleration']-target[2]),
            native_rnea_inf=mx(oracle.backend.inverse(q,vp,ak)-(Mr@ak+h)),native_aba_roundtrip=mx(oracle.aba_roundtrip(q,vp,ak)-ak))
        version=oracle.version
    limits={'relative_acceleration_difference':1e-8,'affine_vs_kkt_relative':1e-8,'finger_acceleration_residual_m_s2':1e-8,
        'tool_acceleration_consistency_mixed':2e-6,'physical_mapping_discrepancy':1e-9,'virtual_power_defect_W':1e-9,
        'affine_power_defect_W':1e-9,'reduced_inertia_relative':1e-9,'kkt_force_balance':1e-8}
    if native:limits.update(native_kkt_acceleration_relative=1e-7,native_mass_inf=1e-8,native_bias_inf=1e-7,native_tool_jacobian_inf=1e-9,native_tool_gamma_inf=1e-8,native_task_acceleration_mixed=2e-6,native_rnea_inf=1e-7,native_aba_roundtrip=1e-7)
    rec['success']=bool(all(np.isfinite(rec[k])and rec[k]<lim for k,lim in limits.items())and rec['physical_reduced_mass_min_eig']>0 and rec['task_reduced_mass_min_eig']>0)
    state=dict(q=q,active_velocity=va,motor_efforts=u,tool_wrench=wrench,physical_velocity=vp,physical_acceleration=a,
        task_acceleration=aa,native_version=version)
    return rec,state

def curvature_witness(comp,q,va,scale):
    g=GeneratedTaskGraph(comp);x=g.lift(q);sv=np.asarray(va)*scale;N,v,c,info=curvature(g,x,sv);Np=g.physical_map(N);vp=g.L@v[:g.nt];cp=g.L@c[:g.nt]
    ref=PhysicalReference(comp.cmg);tar=target_kinematics(x[g.nt:],g.reference_rotation,sv[:6],np.zeros(6))[2]
    full=ref.tool(q,vp,cp)['acceleration']-tar;omitted=ref.tool(q,vp,np.zeros(9))['acceleration']-tar
    return dict(speed_scale=scale,full_linear_m_s2=mx(full[:3]),full_angular_rad_s2=mx(full[3:]),
        omitted_linear_m_s2=mx(omitted[:3]),omitted_angular_rad_s2=mx(omitted[3:]),
        success=bool(mx(full[:3])<2e-6 and mx(full[3:])<2e-6))
