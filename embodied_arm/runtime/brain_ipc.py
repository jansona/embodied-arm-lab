"""Typed, owner-only IPC for the simulation's high-level brain.

This is a fixed protocol, not a network proxy. Importing or constructing the
bridge neither loads credentials nor starts a worker. The explicit worker owns
the provider configuration, and this queue contains no configuration fields.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from embodied_arm.paths import RUNTIME_ROOT
import re
import stat
import threading
import time
import uuid

from embodied_arm.providers.brain import (BrainResult, ENDPOINT, MODEL, PROVIDER_ERROR_REASONS,
                   REPLAN_ERROR_REASONS, scene_snapshot, validate_feedback, validate_plan)
from embodied_arm.runtime.ipc import MAX_BYTES, Store, number


BRAIN_IPC_ROOT = RUNTIME_ROOT / 'brain-ipc'
MAX_CALLS = 10
MAX_REPLANS = 8
MAX_SECONDS = 600
REQUEST_SECONDS = 45
POLL_SECONDS = .02
REQUEST_SCHEMA = 'simulation-brain-request-v1'
REPLAN_REQUEST_SCHEMA = 'simulation-brain-replan-v1'
RESULT_SCHEMA = 'simulation-brain-result-v1'
_TEXT_FORBIDDEN = re.compile(
    r'(?:[a-z][a-z0-9+.-]*://|\b(?:https?|file|data|javascript):|'
    r'\bauthorization\s*:|\b(?:api[_-]?key|access[_-]?token|password)\s*[:=])',
    re.IGNORECASE,
)
_REASONS = frozenset((
    'accepted', 'unavailable', 'busy', 'invalid_input', 'invalid_response',
    'call_budget_exhausted', 'timeout', 'network_error', 'response_too_large',
    'unexpected_model', 'incomplete_response', 'adapter_error', 'cancelled',
    'expired', 'worker_error',
)) | PROVIDER_ERROR_REASONS | REPLAN_ERROR_REASONS


class BrainStore(Store):
    """An independent queue using the existing no-symlink owner-only store."""

    def __init__(self):
        super().__init__(BRAIN_IPC_ROOT)


def _text(value, limit, *, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError('Invalid brain text')
    if _TEXT_FORBIDDEN.search(value) or any(ord(c) < 32 and c not in '\n\t' for c in value):
        raise ValueError('Unsupported brain text')
    return value.strip()


def normalize_input(text, scene, history=None, *, previous_plan=None, feedback=None):
    """Select only user text, known numeric simulation fields and bounded history.

    Old history is omitted when needed to keep even multibyte text within the
    underlying store's fixed byte limit. No user-selected endpoint can enter IPC.
    """
    text = _text(text, 2000)
    safe_scene = scene_snapshot(scene)
    if history is not None and not isinstance(history, list):
        raise ValueError('Invalid brain history')
    safe_history = []
    for item in (history or [])[-6:]:
        if (not isinstance(item, dict) or item.get('role') not in ('user', 'assistant')
                or not isinstance(item.get('content'), str)):
            raise ValueError('Invalid brain history')
        safe_history.append({'role': item['role'],
                             'content': _text(item['content'][:1000], 1000, empty=True)})
    safe = {'text': text, 'scene': safe_scene, 'history': safe_history}
    if previous_plan is not None or feedback is not None:
        previous, outcome = validate_feedback(previous_plan, feedback, safe_scene)
        _text(previous['message'], 300)
        safe.update(previous_plan=previous, feedback=outcome)
    while len(json.dumps(safe, allow_nan=False, separators=(',', ':')).encode()) > MAX_BYTES - 512:
        if not safe_history:
            raise ValueError('Brain request too large')
        del safe_history[0]
    return json.loads(json.dumps(safe, allow_nan=False))


def validate_job(data, job):
    keys = {'schema', 'simulation_only', 'id', 'expires_at', 'text', 'scene', 'history'}
    if isinstance(data, dict) and data.get('schema') == REPLAN_REQUEST_SCHEMA:
        keys |= {'previous_plan', 'feedback'}
    if (not isinstance(data, dict) or set(data) != keys
            or data['schema'] not in (REQUEST_SCHEMA, REPLAN_REQUEST_SCHEMA)
            or data['simulation_only'] is not True or data['id'] != job
            or not number(data['expires_at'])):
        raise ValueError('Invalid brain job')
    safe = normalize_input(data['text'], data['scene'], data['history'],
                           previous_plan=data.get('previous_plan'), feedback=data.get('feedback'))
    if (data['schema'] == REPLAN_REQUEST_SCHEMA and 'feedback' not in safe):
        raise ValueError('Missing repair feedback')
    if any(data[key] != safe[key] for key in safe):
        raise ValueError('Brain job contains unfiltered fields')
    return data


def validate_result(data, scene):
    """Revalidate every model plan after transport, before the UI can use it."""
    fields = {'plan', 'accepted', 'reason', 'model', 'calls', 'provenance'}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError('Invalid brain result fields')
    reason = data['reason']
    if (type(data['accepted']) is not bool or data['model'] != MODEL
            or type(data['calls']) is not int or not 0 <= data['calls'] <= MAX_CALLS
            or data['provenance'] not in ('live', 'mock')
            or not isinstance(reason, str)
            or (reason not in _REASONS and not re.fullmatch(r'http_error_[1-5][0-9]{2}', reason))):
        raise ValueError('Invalid brain result values')
    if data['accepted']:
        if reason != 'accepted':
            raise ValueError('Inconsistent accepted brain result')
        plan = validate_plan(data['plan'], scene)
        _text(plan.message, 300)
    else:
        if data['plan'] is not None or reason == 'accepted':
            raise ValueError('Rejected brain result must have no plan')
        plan = None
    return BrainResult(plan=plan, accepted=data['accepted'], reason=reason,
                       model=MODEL, calls=data['calls'], provenance=data['provenance'])


def worker_status():
    """A ready file alone is insufficient: the bounded worker must own its lock."""
    safe = {'configured': False, 'available': False, 'status': 'offline', 'busy': False,
            'model': MODEL, 'endpoint': ENDPOINT, 'calls': 0, 'max_calls': 0,
            'remaining_calls': 0, 'remaining_seconds': 0.0, 'fallback': False,
            'replans': 0, 'max_replans': 1, 'remaining_replans': 0}
    store = None
    fd = None
    try:
        store = BrainStore()
        fd = os.open('worker.lock', os.O_RDWR | os.O_NOFOLLOW, dir_fd=store.fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return safe
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return safe
        except BlockingIOError:
            pass
        data = store.read('ready.json')
        if not isinstance(data, dict) or data.get('status') not in ('ready', 'calling'):
            return safe
        calls, limit, deadline = data.get('calls'), data.get('max_calls'), data.get('deadline_unix')
        replans = data.get('replans', 0)
        max_replans=data.get('max_replans',1)
        if (type(calls) is not int or type(limit) is not int
                or not 0 <= calls <= limit <= MAX_CALLS or not number(deadline)
                or type(max_replans) is not int or not 0 <= max_replans <= MAX_REPLANS
                or type(replans) is not int or not 0 <= replans <= max_replans):
            return safe
        remaining = deadline - time.time()
        if not 0 < remaining <= MAX_SECONDS + .5 or (calls >= limit and data['status'] != 'calling'):
            return safe
        return {**safe, 'configured': True, 'available': True, 'status': data['status'],
                'busy': data['status'] == 'calling', 'calls': calls, 'max_calls': limit,
                'remaining_calls': limit - calls, 'remaining_seconds': round(remaining, 2),
                'replans': replans, 'max_replans':max_replans, 'remaining_replans': max_replans - replans}
    except (OSError, ValueError, TypeError, RecursionError):
        return safe
    finally:
        if fd is not None:
            os.close(fd)
        if store is not None:
            store.close()


class FileBrainBridge:
    """Synchronous brain API with cancellation and per-instance stale guards."""

    def __init__(self, max_replans=1):
        if type(max_replans) is not int or not 0<=max_replans<=MAX_REPLANS:raise ValueError("Invalid replan budget")
        self.max_replans=max_replans
        self._lock = threading.RLock()
        self._call_lock = threading.Lock()
        self._pending = set()
        self._epoch = 0
        self._replans = 0

    def status(self):
        return worker_status()

    def cancel(self):
        # Invalidate locally even if storage is unavailable. Never let a late
        # model answer survive a stop/reset merely because a file write failed.
        with self._lock:
            self._epoch += 1
            if not self._pending:
                return
            store = None
            try:
                store = BrainStore()
                for job in self._pending:
                    store.write(job + '.cancel', {'cancelled': True})
            except (OSError, ValueError, TypeError):
                pass
            finally:
                if store is not None:
                    store.close()

    def interpret(self, text, scene, history=None):
        return self._call(text, scene, history)

    def replan(self, text, scene, previous_plan, feedback, history=None):
        """A bounded explicit review for this bridge, sharing the worker call budget."""
        return self._call(text, scene, history, replan=True,
                          previous_plan=previous_plan, feedback=feedback)

    def _call(self, text, scene, history, *, replan=False, previous_plan=None, feedback=None):
        if not self._call_lock.acquire(blocking=False):
            return BrainResult(reason='busy')
        try:
            with self._lock:
                epoch = self._epoch
            return self._interpret(text, scene, history, epoch, replan=replan,
                                   previous_plan=previous_plan, feedback=feedback)
        finally:
            self._call_lock.release()

    def _interpret(self, text, scene, history, epoch, *, replan=False, previous_plan=None, feedback=None):
        if replan:
            try:
                validate_feedback(previous_plan, feedback, scene)
            except (ValueError, TypeError, OverflowError, RecursionError):
                return BrainResult(reason='invalid_feedback')
        try:
            safe = (normalize_input(text, scene, history, previous_plan=previous_plan, feedback=feedback)
                    if replan else normalize_input(text, scene, history))
        except (ValueError, TypeError, OverflowError, RecursionError):
            return BrainResult(reason='invalid_input')
        with self._lock:
            if epoch != self._epoch:
                return BrainResult(reason='cancelled')
            if replan and self._replans >= self.max_replans:
                return BrainResult(reason='replan_budget_exhausted')
        status = worker_status()
        if not status['available']:
            return BrainResult(reason='unavailable')
        if status['remaining_calls'] <= 0:
            return BrainResult(reason='busy')
        if replan and status['remaining_replans'] <= 0:
            return BrainResult(reason='replan_budget_exhausted')
        store = None
        job = uuid.uuid4().hex
        deadline = time.monotonic() + REQUEST_SECONDS
        expires = time.time() + REQUEST_SECONDS
        try:
            store = BrainStore()
            with self._lock:
                if epoch != self._epoch:
                    return BrainResult(reason='cancelled')
                if replan:
                    self._replans += 1
                self._pending.add(job)
                store.write(job + '.job', {'schema': REPLAN_REQUEST_SCHEMA if replan else REQUEST_SCHEMA,
                            'simulation_only': True,
                            'id': job, 'expires_at': expires, **safe})
            while time.monotonic() < deadline:
                with self._lock:
                    if epoch != self._epoch:
                        return BrainResult(reason='cancelled')
                if store.exists(job + '.result'):
                    raw = store.read(job + '.result')
                    if (not isinstance(raw, dict) or set(raw) != {'schema', 'id', 'expires_at', 'result'}
                            or raw['schema'] != RESULT_SCHEMA or raw['id'] != job
                            or raw['expires_at'] != expires):
                        raise ValueError('Invalid brain response envelope')
                    result = validate_result(raw['result'], safe['scene'])
                    with self._lock:
                        if epoch != self._epoch:
                            return BrainResult(reason='cancelled')
                        if time.monotonic() >= deadline or time.time() >= expires:
                            return BrainResult(reason='expired')
                        return result
                # A worker may finish its window before this request's TTL.
                if not worker_status()['available'] and not store.exists(job + '.result'):
                    store.write(job + '.cancel', {'cancelled': True})
                    return BrainResult(reason='unavailable')
                time.sleep(POLL_SECONDS)
            store.write(job + '.cancel', {'cancelled': True})
            return BrainResult(reason='timeout')
        except (OSError, ValueError, TypeError, RecursionError):
            return BrainResult(reason='invalid_response')
        finally:
            with self._lock:
                self._pending.discard(job)
            if store is not None:
                for suffix in ('.job', '.result'):
                    try:
                        store.delete(job + suffix)
                    except OSError:
                        pass
                store.close()
