class _S:
    def __init__(self): self.d = {}
    def set(self, k, v): self.d[k] = v
    def get(self, k): return self.d.get(k)
_s = _S()
def get_settings(): return _s
