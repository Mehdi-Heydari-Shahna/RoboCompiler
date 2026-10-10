"""USD/API-shaped camera authoring fakes, not native rendering tests."""
from types import SimpleNamespace
import pytest
from isaac_validation.capture import configure_capture_camera
from test_runtime import install_module


def fixture(monkeypatch, attribute='omni:sensor:Core:tickRate', rate=0.0, apply_schema=True):
    calls = []
    schemas, attrs = [], {}
    # Sentinel optics and transform must not be modified by authoring.
    optics = {'focalLength': 35., 'view_matrix': 'original-view'}
    prim = SimpleNamespace(GetTypeName=lambda: 'Camera', GetAppliedSchemas=lambda: list(schemas))
    prim.GetAttributes = lambda: [SimpleNamespace(GetName=lambda key=key: key,
                                                 Get=lambda value=value: value) for key, value in attrs.items()]
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prim)
    settings = {'/rtx/hydra/supportMultiTickRate': True}
    install_module(monkeypatch, 'carb.settings', get_settings=lambda: SimpleNamespace(get=settings.get))
    def author(path, **kwargs):
        calls.append((path, kwargs))
        if apply_schema:
            schemas.append('OmniSensorAPI')
        if attribute is not None:
            attrs[attribute] = rate
    install_module(monkeypatch, 'isaacsim.sensors.experimental.rtx', RtxCamera=author)
    return stage, calls, optics, settings


@pytest.mark.parametrize('attribute', ['omni:sensor:Core:tickRate', 'omni:sensor:tickRate'])
def test_installed_schema_is_used_with_autotrigger_without_transform_reset(monkeypatch, attribute):
    stage, calls, optics, _ = fixture(monkeypatch, attribute=attribute)
    result = configure_capture_camera(stage, '/Camera')
    assert calls == [('/Camera', {'tick_rate': 0.0, 'reset_xform_op_properties': False})]
    assert result['sensor_schema_applied'] and result['tick_rate_hz'] == 0
    assert result['tick_rate_attributes'] == {attribute: 0.0}
    assert optics == {'focalLength': 35., 'view_matrix': 'original-view'}


@pytest.mark.parametrize('attribute,rate,apply_schema', [
    (None, 0., True), ('omni:sensor:Core:tickRate', 25., True),
    ('omni:sensor:Core:tickRate', float('nan'), True),
    ('omni:sensor:Core:tickRate', 0., False)])
def test_missing_schema_or_unverified_autotrigger_fails(monkeypatch, attribute, rate, apply_schema):
    stage, *_ = fixture(monkeypatch, attribute, rate, apply_schema)
    with pytest.raises(RuntimeError, match='autotrigger|OmniSensorAPI'):
        configure_capture_camera(stage, '/Camera')


def test_bad_camera_prim_fails_before_native_authoring(monkeypatch):
    stage, calls, *_ = fixture(monkeypatch)
    stage.GetPrimAtPath = lambda path: None
    with pytest.raises(RuntimeError, match='not a USD Camera'):
        configure_capture_camera(stage, '/Missing')
    assert calls == []


def test_missing_api_is_not_silently_ignored_with_multitick(monkeypatch):
    import sys
    stage, *_ = fixture(monkeypatch)
    monkeypatch.delitem(sys.modules, 'isaacsim.sensors.experimental.rtx')
    with pytest.raises(RuntimeError, match='authoring API is unavailable'):
        configure_capture_camera(stage, '/Camera')


def test_legacy_non_multitick_can_use_plain_usd_camera(monkeypatch):
    import sys
    stage, _, _, settings = fixture(monkeypatch)
    settings['/rtx/hydra/supportMultiTickRate'] = False
    monkeypatch.delitem(sys.modules, 'isaacsim.sensors.experimental.rtx')
    result = configure_capture_camera(stage, '/Camera')
    assert result['authoring'] == 'legacy USD camera'
    assert result['sensor_schema_applied'] is False
