import fake_world
from pxr import Stage
class _Ctx:
    stage = None
    def open_stage(self, path):
        self.stage = Stage(path); fake_world.open_world(path); return True
    def get_stage(self): return self.stage
    def reset_renderer_accumulation(self): pass
_ctx = _Ctx()
def get_context(): return _ctx
