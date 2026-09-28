"""Strict native mass-property audit, independent of Isaac imports.

RigidBodyView.get_inertias() returns a full inertia tensor about the COM,
expressed along BODY axes. It is not the USD diagonalInertia attribute.
Rotating it by get_coms()'s principal-axis quaternion a second time is wrong.

Reference: Omni Physics 108.0, RigidBodyView.get_inertias() Returns contract.
The alternate rotation below is diagnostic only, never an acceptance option.
"""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
from scipy.spatial.transform import Rotation

# Unchanged from the v2 runtime. No tolerance was enlarged to pass the audit.
MASS_ATOL_KG = 1e-5
COM_ATOL_M = 1e-6
INERTIA_ATOL_KG_M2 = 1e-6


def principal_properties(inertia):
    """Equivalent USD diagonal inertia and proper principal-axis rotation.

    Keep diagonal tensors in their original axes instead of sorting/permuting
    them with eigh. This avoids gratuitous principal-frame changes on the
    supplied benchmark. Off-diagonal tensors still use exact eigendecomposition.
    """
    matrix = np.asarray(inertia, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError('Inertia must be a finite 3x3 matrix')
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError('Inertia must be symmetric')
    if np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ValueError('Inertia must be positive definite')
    if np.array_equal(matrix, np.diag(np.diag(matrix))):
        return np.diag(matrix).copy(), np.eye(3)
    diagonal, axes = np.linalg.eigh(matrix)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    return diagonal, axes


def _json_array(value):
    array = np.asarray(value, dtype=float)
    # Keep evidence writable on NaN/Inf failures without JSON nonstandard NaN.
    out = array.astype(object)
    out[~np.isfinite(array)] = None
    return out.tolist()


def audit_native_properties(names: Sequence[str], masses, com_frames, inertias,
                            expected: Mapping[str, Mapping], prim_paths=None):
    """Return a JSON-safe audit; callers must save it BEFORE raising on failure.

    No expected-value fitting, frame auto-selection, eigenvalue-only comparison,
    or native property setters are used. All nine tensor components are checked.
    """
    names = list(names)
    n = len(names)
    result = dict(
        schema='stewart.native_inertial_audit/1', passed=False,
        inertia_frame='body_axes_about_center_of_mass',
        inertia_conversion='none: native full tensor is already body-frame',
        tolerances=dict(mass_kg=MASS_ATOL_KG, com_m=COM_ATOL_M,
                        inertia_kg_m2=INERTIA_ATOL_KG_M2),
        bodies={}, errors=[],
        mass_max_error_kg=None, com_max_error_m=None, inertia_max_error_kg_m2=None)
    try:
        m = np.asarray(masses, dtype=float)
        c = np.asarray(com_frames, dtype=float)
        I = np.asarray(inertias, dtype=float)
        result['raw'] = dict(body_names=names, prim_paths=list(prim_paths or []),
                             masses=_json_array(m), com_frames_xyzw=_json_array(c),
                             inertias=_json_array(I))
        if not n or len(set(names)) != n or set(names) != set(expected):
            raise ValueError('Native and expected body names must match uniquely')
        if m.shape not in ((n,), (n, 1)) or c.shape != (n, 7) or I.shape not in ((n, 9), (n, 3, 3)):
            raise ValueError(f'Invalid native mass-property shapes: mass={m.shape}, COM={c.shape}, inertia={I.shape}')
        m = m.reshape(n); I = I.reshape(n, 3, 3)
        if not all(np.all(np.isfinite(a)) for a in (m, c, I)):
            raise ValueError('Nonfinite native mass properties (null in raw evidence)')
        if np.max(abs(np.linalg.norm(c[:, 3:], axis=1) - 1)) > 1e-5:
            raise ValueError('Invalid native COM quaternion norm')
        em = np.array([expected[key]['mass_kg'] for key in names], dtype=float)
        ec = np.array([expected[key]['com_m'] for key in names], dtype=float)
        ei = np.array([expected[key]['inertia_kg_m2'] for key in names], dtype=float)
        if em.shape != (n,) or ec.shape != (n, 3) or ei.shape != (n, 3, 3):
            raise ValueError('Invalid expected mass-property shapes')
        if not all(np.all(np.isfinite(a)) for a in (em, ec, ei)):
            raise ValueError('Nonfinite expected mass properties')
        for matrix in ei:
            principal_properties(matrix)
        if np.any(em <= 0):
            raise ValueError('Nonpositive expected body mass')
        me = np.abs(m - em)
        ce = np.max(np.abs(c[:, :3] - ec), axis=1)
        ie = np.max(np.abs(I - ei), axis=(1, 2))
        axes = Rotation.from_quat(c[:, 3:]).as_matrix()
        # Reproduces the v2 error for diagnosis, NEVER used to obtain a pass.
        twice_rotated = axes @ I @ np.transpose(axes, (0, 2, 1))
        alternate = np.max(abs(twice_rotated - ei), axis=(1, 2))
        for j, key in enumerate(names):
            symmetry = float(np.max(abs(I[j] - I[j].T)))
            physical = bool(m[j] > 0 and symmetry <= 1e-7 and
                            np.min(np.linalg.eigvalsh((I[j] + I[j].T) / 2)) > 0)
            passed = bool(physical and me[j] <= MASS_ATOL_KG and
                          ce[j] <= COM_ATOL_M and ie[j] <= INERTIA_ATOL_KG_M2)
            result['bodies'][key] = dict(
                passed=passed, native_mass_kg=float(m[j]),
                native_com_m=c[j, :3].tolist(), native_com_quat_xyzw=c[j, 3:].tolist(),
                native_body_inertia_kg_m2=I[j].tolist(),
                expected_mass_kg=float(em[j]), expected_com_m=ec[j].tolist(),
                expected_body_inertia_kg_m2=ei[j].tolist(),
                mass_error_kg=float(me[j]), com_error_m=float(ce[j]),
                inertia_error_kg_m2=float(ie[j]), physical_tensor=physical,
                diagnostic_only_v2_double_rotation_error_kg_m2=float(alternate[j]))
            if not passed:
                result['errors'].append(f'{key}: mass={me[j]:.9g} kg, COM={ce[j]:.9g} m, '
                                        f'inertia={ie[j]:.9g} kg m2, physical={physical}')
        result.update(mass_max_error_kg=float(np.max(me)), com_max_error_m=float(np.max(ce)),
                      inertia_max_error_kg_m2=float(np.max(ie)))
        result['passed'] = not result['errors']
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result
