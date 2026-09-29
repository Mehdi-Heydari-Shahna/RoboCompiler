class _TL:
    def stop(self): pass
    def is_playing(self): return False
    def get_current_time(self): return 0.
_tl = _TL()
def get_timeline_interface(): return _tl
