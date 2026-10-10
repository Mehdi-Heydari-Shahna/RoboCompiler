"""Startup/shutdown regressions with mocked APIs, NOT native PhysX evidence."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from isaac_validation.runtime import (
    close_simulation_app, create_physx_numpy_view, current_stage_id, select_physx_engine,
)


def install_module(monkeypatch, name, **attributes):
    """Install a complete fake package chain without importing Isaac."""
    parent = None
    for index, part in enumerate(name.split('.')):
        qualified = '.'.join(name.split('.')[:index + 1])
        module = ModuleType(qualified)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, qualified, module)
        if parent is not None:
            setattr(parent, part, module)
        parent = module
    for key, value in attributes.items():
        setattr(parent, key, value)
    return parent


def test_110x_default_stage_failure_is_avoided_with_explicit_id(monkeypatch):
    calls = []
    expected_view = object()

    def create(frontend_name, stage_id=-1, backend='physx'):
        calls.append((frontend_name, stage_id, backend))
        if stage_id == -1:
            raise RuntimeError('Failed to get a valid attached USD stage id from PhysX simulation')
        return expected_view

    install_module(monkeypatch, 'omni.physics.tensors', create_simulation_view=create)
    # This is the original failing call, reproduced against an API-shaped fake.
    with pytest.raises(RuntimeError, match='attached USD stage id'):
        create('numpy')
    assert create_physx_numpy_view(42) is expected_view
    assert calls == [('numpy', -1, 'physx'), ('numpy', 42, 'physx')]


def test_legacy_tensor_api_without_backend_keyword(monkeypatch):
    calls = []
    view = object()

    def create(frontend_name, stage_id=-1):
        calls.append((frontend_name, stage_id))
        return view

    install_module(monkeypatch, 'omni.physics.tensors', create_simulation_view=create)
    assert create_physx_numpy_view(12) is view
    assert calls == [('numpy', 12)]


def test_initialization_failure_is_preserved_without_unsafe_fallback(monkeypatch):
    calls = []
    original = TypeError('native physics initialization failed')

    def create(frontend_name, stage_id=-1, backend='physx'):
        calls.append((frontend_name, stage_id, backend))
        raise original

    install_module(monkeypatch, 'omni.physics.tensors', create_simulation_view=create)
    with pytest.raises(RuntimeError, match='USD stage 91') as exc:
        create_physx_numpy_view(91)
    assert exc.value.__cause__ is original
    assert calls == [('numpy', 91, 'physx')]


@pytest.mark.parametrize('stage_id', [-1, 0, None, True, '17', 1.5])
def test_tensor_view_rejects_invalid_stage_ids(monkeypatch, stage_id):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid ID must never reach native PhysX')

    install_module(monkeypatch, 'omni.physics.tensors', create_simulation_view=forbidden)
    with pytest.raises(ValueError, match='positive, explicit'):
        create_physx_numpy_view(stage_id)


def test_null_tensor_view_is_an_error(monkeypatch):
    install_module(monkeypatch, 'omni.physics.tensors',
                   create_simulation_view=lambda frontend_name, stage_id=-1: None)
    with pytest.raises(RuntimeError, match='no tensor view'):
        create_physx_numpy_view(17)


def test_current_stage_uses_usd_context_id(monkeypatch):
    stage = object()
    context = SimpleNamespace(get_stage=lambda: stage, get_stage_id=lambda: 123)
    install_module(monkeypatch, 'omni.usd', get_context=lambda: context)
    assert current_stage_id(expected_stage=stage) == 123


@pytest.mark.parametrize('stage_id', [-1, 0, None, True, '17'])
def test_current_stage_rejects_bad_ids(monkeypatch, stage_id):
    context = SimpleNamespace(get_stage=object, get_stage_id=lambda: stage_id)
    install_module(monkeypatch, 'omni.usd', get_context=lambda: context)
    with pytest.raises(RuntimeError, match='Invalid USD stage ID'):
        current_stage_id()


def test_missing_or_changed_stage_fails_before_tensor_creation(monkeypatch):
    context = SimpleNamespace(get_stage=lambda: None, get_stage_id=lambda: 42)
    install_module(monkeypatch, 'omni.usd', get_context=lambda: context)
    with pytest.raises(RuntimeError, match='No USD stage'):
        current_stage_id()
    context.get_stage = object
    with pytest.raises(RuntimeError, match='stage changed'):
        current_stage_id(expected_stage=object())


def test_engine_already_physx_is_not_reinitialized(monkeypatch):
    manager = SimpleNamespace(get_active_physics_engine=lambda: 'physx')
    install_module(monkeypatch, 'isaacsim.core.simulation_manager', SimulationManager=manager)
    assert select_physx_engine()['engine'] == 'physx'


def test_physx_selected_before_world_creation(monkeypatch):
    engines = ['newton']
    calls = []

    def switch(engine):
        calls.append(engine)
        engines[0] = engine
        return True

    manager = SimpleNamespace(get_active_physics_engine=lambda: engines[0], switch_physics_engine=switch)
    install_module(monkeypatch, 'isaacsim.core.simulation_manager', SimulationManager=manager)
    result = select_physx_engine()
    assert calls == ['physx']
    assert result['engine'] == 'physx' and result['previous_engine'] == 'newton'


def test_failed_engine_switch_cannot_be_labeled_physx(monkeypatch):
    manager = SimpleNamespace(get_active_physics_engine=lambda: 'newton',
                              switch_physics_engine=lambda name: False)
    install_module(monkeypatch, 'isaacsim.core.simulation_manager', SimulationManager=manager)
    with pytest.raises(RuntimeError, match='Could not select PhysX'):
        select_physx_engine()


def test_single_engine_legacy_runtime(monkeypatch):
    install_module(monkeypatch, 'isaacsim.core.simulation_manager', SimulationManager=SimpleNamespace())
    assert select_physx_engine()['engine_selection'] == 'legacy PhysX runtime'


@pytest.mark.parametrize('code', [0, 1, 130])
def test_new_close_preserves_requested_exit_code(code):
    calls = []

    class App:
        def close(self, wait_for_replicator=True, skip_cleanup=False, exit_code=0):
            calls.append((skip_cleanup, exit_code))

    close_simulation_app(App(), code)
    assert calls == [(False, code)]


def test_legacy_close_without_exit_code():
    calls = []

    class App:
        def close(self, wait_for_replicator=True, skip_cleanup=False):
            calls.append(skip_cleanup)

    close_simulation_app(App(), 1)
    assert calls == [False]
