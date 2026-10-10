"""23.0.5 source-contact compliance: derivation, configuration and USD authoring.

These are software tests. They do NOT show PhysX behaviour; the development
cross-check with NVIDIA ovphysx is documented separately and is not Isaac Sim.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import importlib
import math
import re
import subprocess
import sys
import types

import pytest

from kangaroo_isaac.contact_model import (CONTACT_MODELS, SOURCE_CONTACT, SOURCE_DAMPING_PER_S,
                                          SOURCE_STIFFNESS_PER_S2, describe, source_compliance)
from kangaroo_isaac.control import Config, case_config
from kangaroo_isaac.model import ROOT


def test_derivation_matches_closed_form_from_source_solref_solimp():
    T, dmin, dmax, zeta = .003, .95, .99, 1.
    k = dmin**2 / ((1 - dmin) * dmax**2 * T**2 * zeta**2)
    assert math.isclose(SOURCE_STIFFNESS_PER_S2, k, rel_tol=1e-15)
    assert math.isclose(SOURCE_DAMPING_PER_S, 2 * zeta * math.sqrt(k), rel_tol=1e-15)
    # Numerical record for reviewers.
    assert math.isclose(SOURCE_STIFFNESS_PER_S2, 2046276.457050866, rel_tol=1e-12)
    assert math.isclose(SOURCE_DAMPING_PER_S, 2860.962395454275, rel_tol=1e-12)


def test_source_parameters_are_the_ones_in_the_preserved_mujoco_task():
    text = (ROOT / 'vendor_v22/contact_task.py').read_text(encoding='utf-8')
    assert re.search(r"contact_time_constant_s=\.003\b", text)
    assert "geom.set('solref',f\"{config['contact_time_constant_s']} 1\")" in text
    assert "geom.set('solimp','.95 .99 .001')" in text
    assert SOURCE_CONTACT['time_constant_s'] == .003 and SOURCE_CONTACT['damping_ratio'] == 1.
    assert (SOURCE_CONTACT['solimp_dmin'], SOURCE_CONTACT['solimp_dmax'], SOURCE_CONTACT['solimp_width_m']) == (.95, .99, .001)


@pytest.mark.parametrize('bad', [dict(time_constant_s=0.), dict(damping_ratio=-1.), dict(solimp_dmin=1.),
                                 dict(solimp_dmin=.995), dict(time_constant_s=float('nan'))])
def test_invalid_source_parameters_rejected(bad):
    with pytest.raises(ValueError):
        source_compliance(dict(SOURCE_CONTACT, **bad))


def test_default_configuration_uses_source_compliance():
    c = Config().validate()
    assert c.contact_model == 'source_compliance' and set(CONTACT_MODELS) == {'source_compliance', 'rigid'}
    assert c.contact_stiffness_per_s2 == SOURCE_STIFFNESS_PER_S2 and c.contact_damping_per_s == SOURCE_DAMPING_PER_S
    d = describe(c)
    assert d['physx_material']['compliantContactAccelerationSpring'] is True
    assert d['equivalence_claim'] is False


@pytest.mark.parametrize('field,value', [('contact_model', 'automatic'), ('contact_stiffness_per_s2', 0.),
                                         ('contact_damping_per_s', -1.), ('contact_stiffness_per_s2', float('inf'))])
def test_invalid_contact_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        replace(Config(), **{field: value}).validate()


def test_legacy_rigid_contact_remains_selectable_for_comparison():
    c = case_config('nominal', contact_model='rigid')
    assert c.contact_model == 'rigid' and describe(c)['model'] == 'rigid'


@pytest.mark.parametrize('case', ['nominal', 'refined', 'solver_check', 'low_friction', 'higher_drop_push',
                                  'slow_actuators', 'no_feedforward', 'no_contact', 'no_loops', 'passive'])
def test_contact_model_identical_in_every_case(case):
    c = asdict(case_config(case)); n = asdict(case_config('nominal'))
    for key in ('contact_model', 'contact_stiffness_per_s2', 'contact_damping_per_s'):
        assert c[key] == n[key]


def test_verification_cases_differ_only_in_the_varied_setting():
    n = asdict(case_config('nominal')); r = asdict(case_config('refined')); s = asdict(case_config('solver_check'))
    assert {k for k in n if n[k] != r[k]} == {'dt_s'}
    assert {k for k in n if n[k] != s[k]} == {'solver_position_iterations'}


def test_cli_exposes_contact_model_and_propagates_it_to_workers():
    p = subprocess.run([sys.executable, str(ROOT / 'run_isaac.py'), '--help'], capture_output=True, text=True)
    assert p.returncode == 0 and '--contact-model {source_compliance,rigid}' in p.stdout
    source = (ROOT / 'run_isaac.py').read_text(encoding='utf-8')
    assert "'--contact-model',args.contact_model" in source
    assert source.count('args.contact_mode,args.contact_model') == 2


class _Attr:
    def __init__(self, value=None): self.value = value
    def Get(self): return self.value
    def IsValid(self): return True
    def HasAuthoredValue(self): return self.value is not None


def _fake_pxr(monkeypatch, with_compliance=True):
    calls = {}

    class MaterialAPI:
        @classmethod
        def Apply(cls, prim): return cls()
        def CreateStaticFrictionAttr(self, v): calls['static'] = v
        def CreateDynamicFrictionAttr(self, v): calls['dynamic'] = v
        def CreateRestitutionAttr(self, v): calls['restitution'] = v

    class PhysxMaterialAPI:
        @classmethod
        def Apply(cls, prim): return cls()
        def CreateFrictionCombineModeAttr(self, v): calls['friction_combine'] = v
        def CreateRestitutionCombineModeAttr(self, v): calls['restitution_combine'] = v
    if with_compliance:
        for name in ('CompliantContactAccelerationSpring', 'CompliantContactStiffness', 'CompliantContactDamping'):
            setattr(PhysxMaterialAPI, 'Create' + name + 'Attr',
                    lambda self, v, _n=name: (calls.__setitem__(_n, v), calls.setdefault('order', []).append(_n)))

    class Material:
        @staticmethod
        def Define(stage, path): return types.SimpleNamespace(GetPrim=lambda: path)

    pxr = types.ModuleType('pxr')
    names = dict(Gf=types.SimpleNamespace(), Sdf=types.SimpleNamespace(), Usd=types.SimpleNamespace(),
                 UsdGeom=types.SimpleNamespace(), UsdPhysics=types.SimpleNamespace(MaterialAPI=MaterialAPI),
                 UsdShade=types.SimpleNamespace(Material=Material), UsdLux=types.SimpleNamespace(),
                 PhysxSchema=types.SimpleNamespace(PhysxMaterialAPI=PhysxMaterialAPI), Vt=types.SimpleNamespace())
    for key, value in names.items():
        setattr(pxr, key, value)
    monkeypatch.setitem(sys.modules, 'pxr', pxr)
    monkeypatch.delitem(sys.modules, 'kangaroo_isaac.usd_builder', raising=False)
    module = importlib.import_module('kangaroo_isaac.usd_builder')
    monkeypatch.delitem(sys.modules, 'kangaroo_isaac.usd_builder', raising=False)
    return module, calls


def test_usd_material_authors_source_compliance_and_keeps_friction(monkeypatch):
    builder, calls = _fake_pxr(monkeypatch)
    builder.physical_material(object(), Config())
    assert calls['static'] == calls['dynamic'] == .8 and calls['restitution'] == 0.
    assert calls['CompliantContactAccelerationSpring'] is True
    assert calls['CompliantContactStiffness'] == SOURCE_STIFFNESS_PER_S2
    assert calls['CompliantContactDamping'] == SOURCE_DAMPING_PER_S
    # omni.physx warns when damping/acceleration-spring change while stiffness is 0.
    assert calls['order'] == ['CompliantContactStiffness', 'CompliantContactDamping', 'CompliantContactAccelerationSpring']


def test_rigid_model_authors_no_compliance(monkeypatch):
    builder, calls = _fake_pxr(monkeypatch)
    builder.physical_material(object(), case_config('nominal', contact_model='rigid'))
    assert not any(k.startswith('Compliant') for k in calls)


def test_missing_compliant_schema_fails_closed(monkeypatch):
    builder, _ = _fake_pxr(monkeypatch, with_compliance=False)
    with pytest.raises(RuntimeError, match='compliant source contact cannot be authored'):
        builder.physical_material(object(), Config())


# --- 23.0.5 solver profile -------------------------------------------------

def test_default_profile_is_pgs_64_8_and_doubles_within_the_physx_limit():
    from kangaroo_isaac.control import SOLVER_PROFILES
    c = Config().validate()
    assert c.solver_profile == 'pgs_64_8'
    assert (c.solver_type, c.solver_position_iterations, c.solver_velocity_iterations) == ('PGS', 64, 8)
    assert {k: getattr(c, k) for k in SOLVER_PROFILES['pgs_64_8']} == SOLVER_PROFILES['pgs_64_8']
    assert case_config('solver_check').solver_position_iterations == 128 <= 255
    assert case_config('refined').dt_s == c.dt_s / 2


def test_pgs_profiles_never_request_the_tgs_only_external_force_flag():
    # omni.physx warns "enable external forces every iteration is not supported
    # by the PGS solver type"; the log classifier would fail the case.
    from kangaroo_isaac.control import SOLVER_PROFILES
    for name, profile in SOLVER_PROFILES.items():
        if profile['solver_type'] == 'PGS':
            assert profile['external_forces_every_iteration'] is False, name


def test_23_0_4_profile_remains_selectable_for_comparison():
    c = case_config('nominal', solver_profile='tgs_32_1')
    assert (c.solver_type, c.solver_position_iterations, c.solver_velocity_iterations, c.external_forces_every_iteration) == ('TGS', 32, 1, True)
    assert c.contact_model == 'source_compliance'  # profile and contact law are independent choices


def test_cli_default_profile_and_verify_native_acceptance():
    p = subprocess.run([sys.executable, str(ROOT / 'run_isaac.py'), '--help'], capture_output=True, text=True)
    assert p.returncode == 0
    source = (ROOT / 'run_isaac.py').read_text(encoding='utf-8')
    assert "default='pgs_64_8'" in source
    # The verification guard doubles the position count; 64 -> 128 is allowed, 128 -> 256 is not.
    from kangaroo_isaac.control import SOLVER_PROFILES
    assert SOLVER_PROFILES['pgs_64_8']['solver_position_iterations'] <= 127 < SOLVER_PROFILES['pgs_128_8']['solver_position_iterations']


def test_solver_settings_authored_for_pgs_profile_read_back_exactly():
    from kangaroo_isaac.solver_settings import author_solver_settings, expected_settings

    class Attr:
        def __init__(self, v): self.v = v
        def Get(self): return self.v

    class Scene:
        def __init__(self): self.a = {}
        def __getattr__(self, key):
            if key.startswith('Create') and key.endswith('Attr'):
                return lambda v: self.a.__setitem__(key[6:-4], Attr(v))
            if key.startswith('Get') and key.endswith('Attr'):
                return lambda: self.a[key[3:-4]]
            raise AttributeError(key)
    scene = Scene(); r = author_solver_settings(scene, Config())
    assert r['actual'] == expected_settings(Config()) == {
        'SolverType': 'PGS', 'MinPositionIterationCount': 64, 'MaxPositionIterationCount': 64,
        'MinVelocityIterationCount': 8, 'MaxVelocityIterationCount': 8, 'EnableExternalForcesEveryIteration': False}
