class _EM:
    def set_extension_enabled_immediate(self, name, flag): return True
    def get_enabled_extension_id(self, name): return name + '-fake'
class _App:
    def get_extension_manager(self): return _EM()
    def get_build_version(self): return 'fake-kit-mujoco-plant'
def get_app(): return _App()
