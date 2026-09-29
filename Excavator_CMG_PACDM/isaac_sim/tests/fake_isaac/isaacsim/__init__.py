class SimulationApp:
    def __init__(self, config):
        self.config = dict(config); self._running = True
        print('[fake] Simulation App Startup Complete', flush=True)
    def is_running(self): return self._running
    def update(self): pass
    def reset_render_settings(self): pass
    def close(self): self._running = False
