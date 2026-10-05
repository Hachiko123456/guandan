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
    def __init__(self, session_hours=10.0, save_margin_seconds=300.0, checkpoint_seconds=600.0, *, clock=None, no_time_limit=False):
        if isinstance(save_margin_seconds, bool) or not isinstance(save_margin_seconds, (int, float))                 or not math.isfinite(save_margin_seconds) or save_margin_seconds <= 0:
            raise ValueError('save margin must be a positive finite number')
        if isinstance(checkpoint_seconds, bool) or not isinstance(checkpoint_seconds, (int, float))                 or not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
            raise ValueError('checkpoint budget must be a positive finite number')
        if no_time_limit:
            if session_hours not in (None, 0, float('inf')):
                raise ValueError('no_time_limit requires session_hours=None/0/inf')
            self.unlimited = True
            self.hard_seconds = math.inf
            self.soft_seconds = math.inf
        else:
            for value in (session_hours,):
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                    raise ValueError('session budget must be a positive finite number')
            if save_margin_seconds >= session_hours*3600:
                raise ValueError('save margin must be smaller than session budget')
            self.unlimited = False
            self.hard_seconds = session_hours*3600
            self.soft_seconds = self.hard_seconds-save_margin_seconds
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.hard_deadline = self.started + self.hard_seconds
        self.soft_deadline = self.started + self.soft_seconds
        self.interval = checkpoint_seconds
        # Before the first completed update there is no observation-based
        # duration. Reserve at least one checkpoint interval (and one minute)
        # so a fresh session cannot start work with no save margin at all.
        self.initial_update_seconds = max(60.0, self.interval)
        self.last_checkpoint = self.started
        self.reason = None
        self.completed_update_seconds = []

    def request_stop(self, reason='requested_stop'):
        self.reason = str(reason)

    def check(self, **_):
        if self.reason is not None:
            raise SessionStop(self.reason)
        elapsed = self.clock() - self.started
        if self.unlimited:
            return
        if elapsed >= self.hard_seconds:
            self.reason = 'session_hard_deadline'
            raise SessionStop(self.reason)
        if elapsed >= self.soft_seconds:
            self.reason = 'session_save_margin'
            raise SessionStop(self.reason)

    def before_update(self):
        self.check()
        if self.unlimited:
            return
        estimate = max(self.completed_update_seconds, default=self.initial_update_seconds) * 1.5
        if self.clock()-self.started+estimate>=self.soft_seconds:
            self.reason='insufficient_time_for_next_update'
            raise SessionStop(self.reason)

    def after_update(self, elapsed):
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise ValueError('update elapsed time must be a finite non-negative number')
        elapsed = float(elapsed)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('update elapsed time must be a finite non-negative number')
        self.completed_update_seconds.append(elapsed)

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
