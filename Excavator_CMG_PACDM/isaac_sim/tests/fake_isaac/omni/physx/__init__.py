import fake_world
class _Stream:
    def __init__(self): self.subs = []
    def create_subscription_to_pop(self, cb): self.subs.append(cb); return object()
    def pump(self): pass
_stream = _Stream()
class _Physx:
    def update_transformations(self, *a): pass
    def get_error_event_stream(self): return _stream
    def start_simulation(self): fake_world.world().started = True; return None
class _Sim:
    attached = 0
    def attach_stage(self, sid): self.attached = int(sid); return True
    def get_attached_stage(self): return self.attached
    def flush_changes(self): return None
    def simulate(self, dt, t): fake_world.world().step(dt)
    def fetch_results(self): return True
_physx, _sim = _Physx(), _Sim()
def get_physx_interface(): return _physx
def get_physx_simulation_interface(): return _sim
