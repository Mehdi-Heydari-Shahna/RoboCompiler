"""Minimal pxr.PhysxSchema stand-in built from NVIDIA's codeless PhysX schema.

Development only. Isaac Sim ships a compiled PhysxSchema module; stock usd-core
does not. This shim reads ovphysx's codeless generatedSchema.usda and exposes
Apply / Create<Name>Attr / Get<Name>Attr for the single-apply API schemas the
package authors, with the schema's exact attribute names and value types, so
kangaroo_isaac/usd_builder.py can run unchanged against usd-core.
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import ovphysx
from pxr import Plug
Plug.Registry().RegisterPlugins([str(p) for p in ovphysx.codeless_schema_paths()])
from pxr import Sdf  # noqa: E402

_SCHEMA = Path(ovphysx.codeless_schema_root()) / 'PhysxSchema' / 'resources' / 'generatedSchema.usda'


def _load():
    layer = Sdf.Layer.FindOrOpen(str(_SCHEMA))
    apis = {}
    for spec in layer.rootPrims:
        props = {}
        for prop in spec.properties:
            if isinstance(prop, Sdf.AttributeSpec):
                props[prop.name] = prop.typeName
        apis[spec.name] = props
    return apis


def _make_api(name, props):
    def __init__(self, prim=None):
        self._prim = prim
    def GetPrim(self):
        return self._prim
    def Apply(cls, prim):
        if not prim.ApplyAPI(name):
            raise RuntimeError(f'ApplyAPI({name}) failed; codeless schemas not registered?')
        return cls(prim)
    ns = {'__init__': __init__, 'GetPrim': GetPrim, 'Apply': classmethod(Apply), '_schema_name': name}
    for attr_name, type_name in props.items():
        base = attr_name.split(':')[-1]
        base = base[0].upper() + base[1:]
        def create(self, value=None, writeSparsely=False, _n=attr_name, _t=type_name):
            a = self._prim.CreateAttribute(_n, _t)
            if value is not None:
                a.Set(value)
            return a
        def get(self, _n=attr_name):
            return self._prim.GetAttribute(_n)
        ns['Create' + base + 'Attr'] = create
        ns['Get' + base + 'Attr'] = get
    return type(name, (), ns)


def install():
    import pxr
    mod = types.ModuleType('pxr.PhysxSchema')
    for name, props in _load().items():
        if name.endswith('API'):
            setattr(mod, name, _make_api(name, props))
    sys.modules['pxr.PhysxSchema'] = mod
    pxr.PhysxSchema = mod
    return mod
