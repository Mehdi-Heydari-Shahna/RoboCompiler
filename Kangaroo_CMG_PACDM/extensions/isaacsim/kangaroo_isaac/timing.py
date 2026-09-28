"""Low-overhead elapsed-time accounting; no extrapolated speedup or ETA."""
from __future__ import annotations
import time


class StepTiming:
    def __init__(self,clock=time.perf_counter):
        self.clock=clock;self.previous=None;self.seconds={};self.calls={}

    def begin(self):self.previous=self.clock()

    def mark(self,name: str):
        now=self.clock()
        if self.previous is None:raise RuntimeError('Timing began before begin()')
        elapsed=now-self.previous
        if elapsed<0:raise RuntimeError('Monotonic clock moved backwards')
        self.seconds[name]=self.seconds.get(name,0.)+elapsed
        self.calls[name]=self.calls.get(name,0)+1
        self.previous=now

    def report(self) -> dict:
        return {'scope':'Measured step components; excludes startup, progress output, result compression and shutdown',
                'seconds':dict(self.seconds),'calls':dict(self.calls),
                'microseconds_per_call':{key:1e6*value/self.calls[key] for key,value in self.seconds.items()},
                'native_speedup_claim':False}
