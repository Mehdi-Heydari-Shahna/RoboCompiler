"""Independent reduced and KKT solvers for constrained rigid-body dynamics.

The solvers evaluate instantaneous accelerations and generalized reactions
from physical tree mass, bias, and closure terms. Acceleration constraints
are enforced exactly in the local solve. Time integration and actuator
allocation are handled by the benchmark. All applied tree loads are explicit."""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr



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
