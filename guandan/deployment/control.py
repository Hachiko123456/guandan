"""Cooperative checkpoint-stop policy. No claims about hard-killed session recovery."""
from __future__ import annotations

from contextlib import contextmanager
import math
import signal
import time

from ..training.runtime import BudgetExceeded


class SessionStop(BudgetExceeded):
    pass


class CheckpointController:
    def __init__(self, session_hours=10.0, save_margin_seconds=300.0, checkpoint_seconds=600.0, *, clock=None):
        for value in (session_hours,save_margin_seconds,checkpoint_seconds):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('session/save/checkpoint budgets must be positive finite numbers')
        if save_margin_seconds>=session_hours*3600:
            raise ValueError('save margin must be smaller than session budget')
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.hard_seconds = session_hours*3600
        self.soft_seconds = self.hard_seconds-save_margin_seconds
        self.interval = checkpoint_seconds
        self.last_checkpoint = self.started
        self.reason = None
        self.completed_update_seconds = []

    def request_stop(self, reason='requested_stop'):
        self.reason = str(reason)

    def check(self, **_):
        if self.reason is not None:
            raise SessionStop(self.reason)
        if self.clock()-self.started>=self.soft_seconds:
            self.reason = 'session_save_margin'
            raise SessionStop(self.reason)

    def before_update(self):
        self.check()
        estimate = max(self.completed_update_seconds,default=0.0)*1.5
        if self.clock()-self.started+estimate>=self.soft_seconds:
            self.reason='insufficient_time_for_next_update'
            raise SessionStop(self.reason)

    def after_update(self, elapsed):
        self.completed_update_seconds.append(float(elapsed))

    def checkpoint_due(self):
        return self.clock()-self.last_checkpoint>=self.interval

    def checkpoint_saved(self):
        self.last_checkpoint=self.clock()

    @contextmanager
    def signal_handlers(self):
        originals = {}
        def handler(number, frame):
            self.request_stop(f'signal_{number}')
        for name in ('SIGINT','SIGTERM'):
            number = getattr(signal,name,None)
            if number is not None:
                originals[number] = signal.signal(number,handler)
        try:
            yield self
        finally:
            for number, callback in originals.items():
                signal.signal(number,callback)


class CombinedDeadline:
    def __init__(self, profile_deadline, controller):
        self.profile_deadline,self.controller=profile_deadline,controller
    def check(self,**details):
        self.profile_deadline.check(**details)
        self.controller.check(**details)
