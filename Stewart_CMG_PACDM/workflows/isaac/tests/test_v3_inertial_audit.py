"""Regression for the earlier 0.009 kg m2 failure. No engine is emulated here."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from isaac_validation.inertial_audit import (
    audit_native_properties, principal_properties,
    MASS_ATOL_KG, COM_ATOL_M, INERTIA_ATOL_KG_M2,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(inertia=None):
    I = np.diag([.010, .010, .001]) if inertia is None else np.array(inertia, float)
    diagonal, axes = np.linalg.eigh(I)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    com = np.r_[0., 0., .19, Rotation.from_matrix(axes).as_quat()][None, :]
    expected = {'barrel': dict(mass_kg=1.2, com_m=[0., 0., .19], inertia_kg_m2=I.tolist())}
    return ['barrel'], np.array([1.2], np.float32), com.astype(np.float32), I.astype(np.float32).reshape(1, 9), expected


def test_reproduces_uploaded_v2_failure_and_v3_passes_same_native_tensor():
    args = fixture()
    audit = audit_native_properties(*args)
    assert audit['passed'], audit
    body = audit['bodies']['barrel']
    assert body['inertia_error_kg_m2'] < 1e-8
    assert body['diagnostic_only_v2_double_rotation_error_kg_m2'] == pytest.approx(.009, abs=2e-8)
    json.dumps(audit, allow_nan=False)


def test_default_cmg_diagonal_inertias_keep_identity_axes():
    cmg = json.loads((ROOT/'data/stewart.cmg.json').read_text())
    count = 0
    for body in cmg['bodies']:
        if body['mass_kg'] <= 0:
            continue
        I = np.array(body['inertia_kg_m2'])
        diagonal, axes = principal_properties(I)
        np.testing.assert_allclose(axes @ np.diag(diagonal) @ axes.T, I, atol=1e-14)
        if np.array_equal(I, np.diag(np.diag(I))):
            np.testing.assert_array_equal(axes, np.eye(3)); count += 1
    assert count >= 19


@pytest.mark.parametrize('seed', range(12))
def test_offdiagonal_tensors_reconstruct_and_audit_full_components(seed):
    rng = np.random.default_rng(seed)
    axes = Rotation.random(random_state=rng).as_matrix()
    I = axes @ np.diag([.07, .09, .11]) @ axes.T
    diagonal, authored_axes = principal_properties(I)
    np.testing.assert_allclose(authored_axes @ np.diag(diagonal) @ authored_axes.T, I, atol=1e-13)
    assert np.linalg.det(authored_axes) == pytest.approx(1.)
    args = fixture(I)
    assert audit_native_properties(*args)['passed']
    # Same eigenvalues but wrong axes MUST FAIL. Eigenvalues alone do not suffice.
    args[3][0] = np.diag([.07, .09, .11]).reshape(9)
    assert not audit_native_properties(*args)['passed']


@pytest.mark.parametrize('field,delta', [(1, 1.1e-5), (2, 1.1e-6), (3, 1.1e-6)])
def test_original_tolerances_are_still_enforced(field, delta):
    args = list(fixture())
    if field == 1:
        args[field] = args[field].astype(float) + delta
    elif field == 2:
        args[field][0, 0] += delta
    else:
        args[field][0, 0] += delta
    result = audit_native_properties(*args)
    assert not result['passed'], result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field', [1, 2, 3])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_native_evidence_fails_and_remains_serializable(field, bad):
    args = list(fixture()); args[field].flat[0] = bad
    result = audit_native_properties(*args)
    assert not result['passed']
    assert 'Nonfinite' in str(result['errors'])
    json.dumps(result, allow_nan=False)


def test_wrong_quaternion_norm_fails():
    args = list(fixture()); args[2][0, 3:] = 0
    assert not audit_native_properties(*args)['passed']


def test_no_choose_whichever_frame_passes_escape():
    args = list(fixture())
    args[3][0] = np.diag([.001, .010, .010]).reshape(9)
    audit = audit_native_properties(*args)
    assert not audit['passed']  # This is in principal axes, not the API's body axes.
    assert audit['bodies']['barrel']['diagnostic_only_v2_double_rotation_error_kg_m2'] < 1e-7


@pytest.mark.parametrize('kind', ['shape', 'duplicate_names', 'unknown_body', 'negative_mass', 'nonsymmetric', 'negative_inertia'])
def test_malformed_native_input_fails_closed(kind):
    args = list(fixture())
    if kind == 'shape': args[3] = args[3][:, :3]
    elif kind == 'duplicate_names': args[0] = ['barrel', 'barrel']
    elif kind == 'unknown_body': args[0] = ['rod']
    elif kind == 'negative_mass': args[1][0] = -1.
    elif kind == 'nonsymmetric': args[3][0, 1] = 1e-3
    elif kind == 'negative_inertia': args[3][0, 0] = -1.
    result = audit_native_properties(*args)
    assert not result['passed']; json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('I', [np.eye(2), np.zeros((3, 3)), np.diag([-1, 2, 3]),
                              np.full((3, 3), np.nan), [[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
def test_invalid_usd_inertia_is_not_authored(I):
    with pytest.raises(ValueError): principal_properties(I)


def test_unchanged_native_tolerance_values():
    assert (MASS_ATOL_KG, COM_ATOL_M, INERTIA_ATOL_KG_M2) == (1e-5, 1e-6, 1e-6)
