"""Local instantaneous rigid-body dynamics with exact acceleration constraints.

No integration or actuator allocation. Native backends supply tree components;
the two constraint solvers here belong to RoboCompiler, not MuJoCo's compliant
equality solver. Every synthetic applied tree load must be supplied explicitly.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr

from native_bias import NativeBiasDynamics
from source_bias import SourceBiasDynamics


def finite_vector(value, length, label):
    if np.iscomplexobj(value):
        raise ValueError(f'{label} must be real')
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{label} must contain {length} finite numbers')
    return result.copy()


def reduce_kinematics(residual, jacobian, jacobian_dot, velocity, active,
                      closure_tolerance=1e-8, velocity_tolerance=1e-8):
    """Recover N and Ndot from all physical closure rows on a local branch."""
    r, j, jd, v = map(lambda x:np.asarray(x,dtype=float),
                      (residual,jacobian,jacobian_dot,velocity))
    if j.ndim != 2 or jd.shape != j.shape or r.shape != (j.shape[0],) or v.shape != (j.shape[1],):
        raise ValueError('Invalid closure matrix/vector dimensions')
    if not all(np.all(np.isfinite(x)) for x in (r,j,jd,v)):
        raise ValueError('Closure inputs must be finite')
    if len(set(active)) != len(active) or any(i < 0 or i >= j.shape[1] for i in active):
        raise ValueError('Invalid independent coordinate indices')
    if len(active) != 7:
        raise ValueError('This source requires seven independent coordinates, including its pin')
    passive = [i for i in range(j.shape[1]) if i not in active]
    if np.max(abs(r))>closure_tolerance:
        raise ValueError('Configuration does not satisfy physical loop closure')
    if np.max(abs(j@v))>velocity_tolerance:
        raise ValueError('Velocity is not tangent to the physical loops')
    rank = int(np.linalg.matrix_rank(j,tol=1e-9))
    if rank != len(passive) or np.linalg.matrix_rank(j[:,passive],tol=1e-9)!=len(passive):
        raise ValueError('Closure/coordinate partition is rank deficient')
    n = np.zeros((j.shape[1],len(active)))
    n[active] = np.eye(len(active))
    n[passive] = np.linalg.lstsq(j[:,passive],-j[:,active],rcond=1e-12)[0]
    nd = np.zeros_like(n)
    nd[passive] = np.linalg.lstsq(j[:,passive],-jd@n,rcond=1e-12)[0]
    b = nd@v[active]
    if (np.max(abs(j@n))>1e-8 or np.max(abs(j@nd+jd@n))>1e-8 or
            np.max(abs(v-n@v[active]))>velocity_tolerance or
            np.max(abs(j@b+jd@v))>1e-8):
        raise ValueError('Reduction fails all-row velocity/acceleration compatibility')
    return {'tangent_map':n,'tangent_map_dot':nd,'curvature':b,
            'constraint_rank':rank,'residual':r,'jacobian':j,'jacobian_dot':jd,
            'velocity':v,'jdot_velocity':jd@v}


def solve_reduced(component, applied_tree_load):
    """Return instantaneous acceleration and unique generalized reaction."""
    m, h = component['mass_matrix'],component['bias_forces']
    n, b = component['tangent_map'],component['curvature']
    tau = finite_vector(applied_tree_load,m.shape[0],'Applied synthetic tree load')
    mr, hr = n.T@m@n,n.T@(h+m@b)
    # Diagonal congruence avoids treating a small pin inertia as numerical zero.
    scale = 1./np.sqrt(np.diag(mr))
    normalized = (scale[:,None]*mr)*scale[None,:]
    np.linalg.cholesky((normalized+normalized.T)/2.)
    udd = scale*np.linalg.solve(normalized,scale*(n.T@tau-hr))
    acceleration = n@udd+b
    reaction = m@acceleration+h-tau
    if not all(np.all(np.isfinite(x)) for x in (udd,acceleration,reaction)):
        raise ValueError('Nonfinite reduced acceleration result')
    return {'acceleration':acceleration,'independent_acceleration':udd,
            'generalized_reaction':reaction,'reduced_mass':mr,'reduced_bias':hr}


def solve_kkt(component, applied_tree_load):
    """Separate full-tree constrained solve using rank-revealing row selection.

    M a + h = tau + J_selected.T lambda; J a + Jdot v = 0.
    Reported lambda depends on the selected row basis and is not a unique
    set of physical joint/cylinder forces. Generalized reaction is comparable.
    """
    m, h = component['mass_matrix'],component['bias_forces']
    j, gamma = component['jacobian'],component['jdot_velocity']
    tau = finite_vector(applied_tree_load,m.shape[0],'Applied synthetic tree load')
    scale = 1./np.sqrt(np.diag(m))
    weighted = j*scale[None,:]
    _,_,piv = qr(weighted.T,mode='economic',pivoting=True)
    rank = int(component['constraint_rank'])
    rows = np.asarray(piv[:rank],dtype=int)
    selected = j[rows]
    row_scale = np.linalg.norm(weighted[rows],axis=1)
    if np.any(row_scale<=0) or np.linalg.matrix_rank(selected,tol=1e-9)!=rank:
        raise ValueError('Selected constraints do not retain physical rank')
    normalized_mass = scale[:,None]*m*scale[None,:]
    normalized_j = weighted[rows]/row_scale[:,None]
    matrix = np.block([[normalized_mass,-normalized_j.T],
                       [normalized_j,np.zeros((rank,rank))]])
    rhs = np.r_[scale*(tau-h),-gamma[rows]/row_scale]
    solution = np.linalg.solve(matrix,rhs)
    acceleration = scale*solution[:m.shape[0]]
    multipliers = solution[m.shape[0]:]/row_scale
    reaction = selected.T@multipliers
    if not np.all(np.isfinite(solution)):
        raise ValueError('Nonfinite KKT acceleration result')
    return {'acceleration':acceleration,'generalized_reaction':reaction,
            'selected_rows':rows,'selected_row_multipliers':multipliers}


class ClosedDynamics:
    """Source-specific local components for three implementations in source order."""

    def __init__(self, cmg, mjcf_path, mj_mapping):
        self.native = NativeBiasDynamics(cmg,mjcf_path,mj_mapping)
        self.tree_ids = self.native.tree_ids
        self.independent_ids = self.native.pin_backend.independent_ids
        self.active = [self.tree_ids.index(x) for x in self.independent_ids]
        self.passive = [i for i in range(len(self.tree_ids)) if i not in self.active]
        self.cut_ids = self.native.pin_backend.cut_ids
        self.source = SourceBiasDynamics(cmg,self.tree_ids)

    def source_kinematics(self, q, velocity, gravity):
        e = self.source.evaluate(q,velocity,gravity)
        r,j,jd = self.source.closure(e,self.cut_ids)
        return e,reduce_kinematics(r,j,jd,velocity,self.active)

    def state(self, q, independent_velocity, gravity):
        """Build components at a closed q and explicit seven-angle velocity."""
        q = finite_vector(q,len(self.tree_ids),'Tree configuration')
        u = finite_vector(independent_velocity,len(self.independent_ids),'Independent velocity')
        gravity = finite_vector(gravity,3,'Gravity')
        _,zero = self.source_kinematics(q,np.zeros(len(q)),gravity)
        v = zero['tangent_map']@u
        source,kin = self.source_kinematics(q,v,gravity)
        values = self.native.evaluate(q,v,gravity)
        closures = self.native.constraint_kinematics(q,v)
        components = {'source':{**source,**kin}}
        for name in ['pin','mujoco']:
            c = closures[name]
            k = reduce_kinematics(c['residual'],c['jacobian'],c['jacobian_dot'],v,self.active)
            components[name] = {**values[name],**k}
        for value in components.values():
            n,m,b,h = [value[x] for x in ['tangent_map','mass_matrix','curvature','bias_forces']]
            value['reduced_mass'] = n.T@m@n
            value['reduced_bias'] = n.T@(h+m@b)
        return {'q':q,'velocity':v,'independent_velocity':u,'gravity':gravity,'components':components}
