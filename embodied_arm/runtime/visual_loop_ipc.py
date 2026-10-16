"""Private typed full-image Qwen / textual Jev IPC, simulation only.

No configuration, credentials, worker spawning or network access in this module.
Robot whitelisting is hardware metadata only; JPEG decoding is format validation,
not image inference. The caller retains capture/robot freshness checks before motion.
"""
from __future__ import annotations
import base64
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid

from embodied_arm.paths import RUNTIME_ROOT
from embodied_arm.runtime.ipc import Store
from embodied_arm.providers.visual_brain import (frame_snapshot, sanitize_robot,
    normalize_history, text_value, number, validate_visual_observation,
    validate_visual_result, visual_failure)

VISUAL_IPC_ROOT = RUNTIME_ROOT / 'visual-loop-ipc'
COMPONENTS = ('qwen', 'jev')
MODES = ('qwen_direct', 'two_model')
MAX_CALLS = 200
MAX_QWEN_CALLS = 201
MAX_SECONDS = 600
MAX_BYTES = 9 * 1024 * 1024  # Three bounded full JPEGs, base64 and fixed metadata.
MAX_RESULT_BYTES = 32768
MAX_FRAME_AGE = 60
REQUEST_SECONDS = 65
POLL_SECONDS = .02
REQUEST_SCHEMA = 'simulation-visual-loop-request-v1'
RESULT_SCHEMA = 'simulation-visual-loop-result-v1'
_ID = re.compile(r'[0-9a-f]{32}\Z')
_NAME = re.compile(r'(?:[0-9a-f]{32}\.(?:job|result|cancel|tmp)|ready\.json|last-run-audit\.json|worker\.lock)\Z')


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def _loads(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _component(component):
    if not isinstance(component, str) or component not in COMPONENTS:
        raise ValueError('Invalid visual component')
    return component


class VisualStore(Store):
    """Separate bounded store; legacy file size and schemas remain unchanged."""
    def __init__(self):
        super().__init__(VISUAL_IPC_ROOT)

    def _name(self, name):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError('Invalid visual IPC name')

    @staticmethod
    def _safe(info):
        return (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and info.st_nlink == 1 and not info.st_mode & 0o077)

    def exists(self, name):
        self._name(name)
        try: info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError: return False
        if not self._safe(info): raise ValueError('Unsafe visual IPC file')
        return True

    def write(self, name, data):
        self._name(name)
        self.exists(name)
        raw = _json(data)
        if len(raw) > MAX_BYTES: raise ValueError('Visual IPC payload too large')
        temporary = uuid.uuid4().hex + '.tmp'
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self.fd)
        try:
            with os.fdopen(fd, 'wb') as file:
                file.write(raw)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        finally:
            try: os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError: pass

    def read(self, name):
        self._name(name)
        # Atomic ready snapshots can be replaced between open and fstat.
        # Reopen only this known snapshot once; never accept an unlinked inode.
        for attempt in range(2):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
            with os.fdopen(fd, 'rb') as file:
                info = os.fstat(file.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mode & 0o077 or info.st_size > MAX_BYTES):
                    raise ValueError('Unsafe visual IPC file')
                if info.st_nlink == 0 and name == 'ready.json' and attempt == 0:
                    continue
                if info.st_nlink != 1: raise ValueError('Unsafe visual IPC file')
                raw = file.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES: raise ValueError('Visual IPC payload too large')
            return _loads(raw)


def normalize_payload(component, payload):
    _component(component)
    fields = {'task', 'packet', 'robot', 'history', 'review'} if component == 'qwen' else {'observation', 'robot'}
    if not isinstance(payload, dict) or set(payload) != fields: raise ValueError('Invalid request fields')
    robot = sanitize_robot(payload['robot'])
    if robot['busy'] or not robot['paused']: raise ValueError('robot_not_idle')
    if component == 'jev':
        observation = validate_visual_observation(payload['observation'])
        if observation['decision'] != 'continue' or observation['confidence'] < .75:
            raise ValueError('vision_not_continue')
        return {'observation': observation, 'robot': robot}
    if not text_value(payload['task'], 2000) or type(payload['review']) is not bool:
        raise ValueError('Invalid task')
    packet = payload['packet']
    if not isinstance(packet, dict) or set(packet) != {'frame_id', 'captured_at', 'images'}:
        raise ValueError('Invalid image-only packet')
    if len(_json(payload)) > MAX_BYTES - 2048: raise ValueError('Image packet too large')
    packet = frame_snapshot(packet, max_frame_age=MAX_FRAME_AGE)
    for image in packet['images']:
        raw = base64.b64decode(image['data_url'].split(',', 1)[1], validate=True)
        if not raw.startswith(b'\xff\xd8') or not raw.endswith(b'\xff\xd9'):
            raise ValueError('Invalid complete JPEG')
    return {'task': payload['task'], 'packet': packet, 'robot': robot,
            'history': normalize_history(payload['history']), 'review': payload['review']}


def payload_hash(payload):
    return hashlib.sha256(_json(payload)).hexdigest()


def fresh(component, payload):
    return component != 'qwen' or 0 <= time.monotonic() - payload['packet']['captured_at'] <= MAX_FRAME_AGE


def failure(component, reason, calls=0):
    if component == 'qwen': return visual_failure(reason, calls)
    from embodied_arm.providers.visual_cerebellum import failure as jev_failure
    return jev_failure(reason)


def validate_result(component, result, payload):
    if component == 'qwen':
        return validate_visual_result(result, payload['packet']['frame_id'])
    from embodied_arm.providers.visual_cerebellum import validate_jev_result
    return validate_jev_result(result)


def validate_job(data, job, session_id):
    fields = {'schema', 'simulation_only', 'id', 'session_id', 'client_id', 'generation',
              'component', 'expires_at', 'payload_sha256', 'payload'}
    if (not isinstance(data, dict) or set(data) != fields or data['schema'] != REQUEST_SCHEMA
            or data['simulation_only'] is not True or data['id'] != job
            or not isinstance(job, str) or not _ID.fullmatch(job)
            or data['session_id'] != session_id or not isinstance(data['client_id'], str)
            or not _ID.fullmatch(data['client_id']) or type(data['generation']) is not int
            or not 0 <= data['generation'] < 1_000_000_000 or not number(data['expires_at'])):
        raise ValueError('Invalid visual request')
    clean = normalize_payload(data['component'], data['payload'])
    if clean != data['payload'] or data['payload_sha256'] != payload_hash(clean):
        raise ValueError('Unfiltered or mismatched visual request')
    return data


def worker_status(component='qwen'):
    _component(component)
    safe = {'configured': False, 'available': False, 'status': 'offline', 'busy': False,
            'component': component, 'calls': 0, 'max_calls': 0, 'remaining_calls': 0,
            'remaining_seconds': 0., 'session_id': None, 'fallback': False, 'mode': None, 'enabled': False}
    store = None
    fd = None
    try:
        store = VisualStore()
        if not store.exists('worker.lock'): return safe
        fd = os.open('worker.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=store.fd)
        if not store._safe(os.fstat(fd)): return safe
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return safe
        except BlockingIOError: pass
        data = store.read('ready.json')
        if not isinstance(data, dict) or data.get('status') not in ('ready', 'calling'): return safe
        session = data.get('session_id')
        mode = data.get('mode')
        calls, limit, deadline = data.get('calls'), data.get('max_calls'), data.get('deadline_unix')
        if (not isinstance(session, str) or not _ID.fullmatch(session)
                or not isinstance(mode, str) or mode not in MODES
                or not isinstance(limit, dict) or set(limit) != set(COMPONENTS)
                or type(limit['qwen']) is not int or not 2 <= limit['qwen'] <= MAX_QWEN_CALLS
                or type(limit['jev']) is not int
                or limit['jev'] != (limit['qwen'] - 1 if mode == 'two_model' else 0)
                or not isinstance(calls, dict) or set(calls) != set(COMPONENTS)
                or any(type(v) is not int or not 0 <= v <= limit[k] for k, v in calls.items())
                or not number(deadline) or data.get('active_component') not in (None, *COMPONENTS)):
            return safe
        remaining = deadline - time.time()
        if not 0 < remaining <= MAX_SECONDS + .5: return safe
        enabled = component == 'qwen' or mode == 'two_model'
        if not enabled:
            if data['active_component'] == component: return safe
            return {**safe, 'status': 'disabled', 'mode': mode, 'session_id': session,
                    'remaining_seconds': round(remaining, 3)}
        available = calls[component] < limit[component] or data['active_component'] == component
        return {**safe, 'configured': True, 'available': available, 'mode': mode, 'enabled': True,
                'status': data['status'] if available else 'call_budget', 'busy': data['status'] == 'calling',
                'calls': calls[component], 'max_calls': limit[component], 'remaining_calls': limit[component] - calls[component],
                'remaining_seconds': round(remaining, 3), 'session_id': session}
    except (OSError, ValueError, TypeError, RecursionError): return safe
    finally:
        if fd is not None: os.close(fd)
        if store is not None: store.close()


class FileVisualBridge:
    def __init__(self, component='qwen'):
        self.component = _component(component)
        self._lock = threading.RLock()
        self._call_lock = threading.Lock()
        self._generation = 0
        self._client_id = uuid.uuid4().hex
        self._pending = set()
        self._closed = False

    def status(self):
        status = worker_status(self.component)
        with self._lock:
            if self._closed: status.update(available=False, status='closed')
        return status

    def cancel(self):
        with self._lock:
            self._generation += 1
            if not self._pending: return
            store = None
            try:
                store = VisualStore()
                for job in self._pending: store.write(job + '.cancel', {'cancelled': True})
            except (OSError, ValueError, TypeError): pass
            finally:
                if store is not None: store.close()

    def close(self):
        with self._lock:
            self._closed = True
            self.cancel()

    def observe(self, task, packet, robot, history=None, review=False):
        if self.component != 'qwen': return failure(self.component, 'invalid_input')
        return self._request({'task': task, 'packet': packet, 'robot': robot, 'history': history, 'review': review})

    def decide(self, observation, robot):
        if self.component != 'jev': return failure(self.component, 'invalid_input')
        return self._request({'observation': observation, 'robot': robot})

    def _request(self, payload):
        if not self._call_lock.acquire(blocking=False): return failure(self.component, 'busy')
        try:
            with self._lock:
                if self._closed: return failure(self.component, 'closed')
                generation = self._generation
            try: payload = normalize_payload(self.component, payload)
            except (OSError, ValueError, TypeError, OverflowError, RecursionError):
                return failure(self.component, 'invalid_input')
            return self._exchange(payload, generation)
        finally: self._call_lock.release()

    def _exchange(self, payload, generation):
        with self._lock:
            if generation != self._generation: return failure(self.component, 'cancelled')
        status = self.status()
        if not status['available']: return failure(self.component, 'unavailable')
        if not status['remaining_calls']: return failure(self.component, 'call_budget')
        duration = min(REQUEST_SECONDS, status['remaining_seconds'])
        expires = time.time() + duration
        deadline = time.monotonic() + duration
        job = uuid.uuid4().hex
        binding = {'id': job, 'session_id': status['session_id'], 'client_id': self._client_id,
                   'generation': generation, 'component': self.component, 'expires_at': expires,
                   'payload_sha256': payload_hash(payload)}
        store = None
        try:
            store = VisualStore()
            with self._lock:
                if generation != self._generation: return failure(self.component, 'cancelled')
                self._pending.add(job)
                store.write(job + '.job', {**binding, 'schema': REQUEST_SCHEMA,
                            'simulation_only': True, 'payload': payload})
            while time.monotonic() < deadline:
                with self._lock:
                    if generation != self._generation: return failure(self.component, 'cancelled')
                if not fresh(self.component, payload):
                    store.write(job + '.cancel', {'cancelled': True})
                    return failure(self.component, 'stale_frame')
                if store.exists(job + '.result'):
                    raw = store.read(job + '.result')
                    if (not isinstance(raw, dict) or set(raw) != set(binding) | {'schema', 'result'}
                            or raw['schema'] != RESULT_SCHEMA
                            or _json({k: raw[k] for k in binding}) != _json(binding)):
                        raise ValueError('Visual response binding mismatch')
                    result = validate_result(self.component, raw['result'], payload)
                    with self._lock:
                        if generation != self._generation: return failure(self.component, 'cancelled')
                        if time.monotonic() >= deadline or time.time() >= expires: return failure(self.component, 'expired')
                        if not fresh(self.component, payload): return failure(self.component, 'stale_frame')
                        return result
                current = worker_status(self.component)
                if (not current['available'] or current['session_id'] != binding['session_id']) and not store.exists(job + '.result'):
                    store.write(job + '.cancel', {'cancelled': True})
                    return failure(self.component, 'unavailable')
                time.sleep(POLL_SECONDS)
            store.write(job + '.cancel', {'cancelled': True})
            return failure(self.component, 'timeout')
        except (OSError, ValueError, TypeError, RecursionError): return failure(self.component, 'invalid_response')
        finally:
            with self._lock: self._pending.discard(job)
            if store is not None:
                for suffix in ('.job', '.result'):
                    try: store.delete(job + suffix)
                    except OSError: pass
                store.close()
