"""Strict private SO101 RGB-target/encoder IPC; no Panda schema or network proxy."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid
from adapters import ALIYUN_ENDPOINT, ALIYUN_MODEL
from ipc import Store, number
from so101_jev import SO101Decision, validate_observation, validate_decision

SO101_IPC_ROOT = Path(__file__).resolve().parent / '.runtime' / 'so101-jev-ipc'
MAX_BYTES = 32767
MAX_CALLS = 200
DEFAULT_CALLS = 12
MAX_SECONDS = 420
DEFAULT_SECONDS = 180
REQUEST_SECONDS = 30
POLL_SECONDS = .02
REQUEST_SCHEMA = 'so101-jev-ipc-request-v1'
RESULT_SCHEMA = 'so101-jev-ipc-result-v1'
_NAME = re.compile(r'(?:[0-9a-f]{32}\.(?:job|result|cancel|tmp)|ready\.json|last-run-audit\.json|worker\.lock)\Z')


def _loads(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _json(data):
    return json.dumps(data, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


class SO101Store(Store):
    """Owner-only 32 KiB SO101 store, independent of the Panda queue."""

    def __init__(self):
        super().__init__(SO101_IPC_ROOT)

    def _name(self, name):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError('Invalid SO101 IPC name')

    def exists(self, name):
        self._name(name)
        try:
            info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            raise ValueError('Unsafe SO101 IPC file')
        return True

    def write(self, name, data):
        self._name(name)
        self.exists(name)
        raw = _json(data)
        if len(raw) > MAX_BYTES:
            raise ValueError('SO101 IPC payload too large')
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
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def read(self, name):
        self._name(name)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        with os.fdopen(fd, 'rb') as file:
            info = os.fstat(file.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > MAX_BYTES):
                raise ValueError('Unsafe SO101 IPC file')
            raw = file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('SO101 IPC payload too large')
        return _loads(raw)


def observation_hash(observation):
    return hashlib.sha256(_json(observation)).hexdigest()


def failure(reason, observation=None):
    return SO101Decision(reason=reason, axis_index=observation['axis_index'] if observation else None)


def validate_job(data, job):
    fields = {'schema', 'simulation_only', 'id', 'expires_at', 'observation_sha256', 'observation'}
    if (not isinstance(data, dict) or set(data) != fields or data['schema'] != REQUEST_SCHEMA
            or data['simulation_only'] is not True or data['id'] != job or not number(data['expires_at'])):
        raise ValueError('Invalid SO101 IPC job')
    observation = validate_observation(data['observation'])
    if (observation != data['observation'] or data['observation_sha256'] != observation_hash(observation)
            or observation['robot']['busy'] or not observation['robot']['paused']):
        raise ValueError('SO101 IPC requires a bound, idle, paused observation')
    return data


def worker_status():
    safe = {'configured': False, 'available': False, 'status': 'offline', 'busy': False,
            'model': ALIYUN_MODEL, 'endpoint': ALIYUN_ENDPOINT, 'robot': 'SO-ARM101',
            'source': 'rgb_only', 'calls': 0, 'max_calls': 0,
            'remaining_calls': 0, 'remaining_seconds': 0.0, 'fallback': False}
    store = None
    fd = None
    try:
        store = SO101Store()
        if not store.exists('worker.lock'):
            return safe
        fd = os.open('worker.lock', os.O_RDWR | os.O_NOFOLLOW, dir_fd=store.fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
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
        if (type(calls) is not int or type(limit) is not int or not 1 <= limit <= MAX_CALLS
                or not 0 <= calls <= limit or not number(deadline)):
            return safe
        remaining = deadline - time.time()
        if not 0 < remaining <= MAX_SECONDS + .5 or (calls >= limit and data['status'] != 'calling'):
            return safe
        return {**safe, 'configured': True, 'available': True, 'status': data['status'],
                'busy': data['status'] == 'calling', 'calls': calls, 'max_calls': limit,
                'remaining_calls': limit - calls, 'remaining_seconds': round(remaining, 2)}
    except (OSError, ValueError, TypeError, RecursionError):
        return safe
    finally:
        if fd is not None: os.close(fd)
        if store is not None: store.close()


class FileSO101Bridge:
    """Synchronous controller contract; no configuration or worker autostart."""
    def __init__(self):
        self._lock = threading.RLock()
        self._call_lock = threading.Lock()
        self._epoch = 0
        self._pending = set()

    def status(self):
        return worker_status()

    def cancel(self):
        with self._lock:
            self._epoch += 1
            if not self._pending: return
            store = None
            try:
                store = SO101Store()
                for job in self._pending:
                    store.write(job + '.cancel', {'cancelled': True})
            except (OSError, ValueError, TypeError):
                pass
            finally:
                if store is not None: store.close()

    def decide(self, observation):
        if not self._call_lock.acquire(blocking=False):
            return failure('busy')
        try:
            with self._lock: epoch = self._epoch
            return self._decide(observation, epoch)
        finally:
            self._call_lock.release()

    def _decide(self, observation, epoch):
        try:
            safe = validate_observation(observation)
        except (ValueError, TypeError, OverflowError, RecursionError):
            return failure('invalid_input')
        with self._lock:
            if epoch != self._epoch: return failure('cancelled', safe)
        if safe['robot']['busy'] or not safe['robot']['paused']:
            return failure('busy', safe)
        status = worker_status()
        if not status['available']: return failure('unavailable', safe)
        if status['remaining_calls'] <= 0: return failure('busy', safe)
        store = None
        job = uuid.uuid4().hex
        digest = observation_hash(safe)
        deadline = time.monotonic() + REQUEST_SECONDS
        expires = time.time() + REQUEST_SECONDS
        try:
            store = SO101Store()
            with self._lock:
                if epoch != self._epoch: return failure('cancelled', safe)
                self._pending.add(job)
                store.write(job + '.job', {'schema': REQUEST_SCHEMA, 'simulation_only': True, 'id': job,
                    'expires_at': expires, 'observation_sha256': digest, 'observation': safe})
            while time.monotonic() < deadline:
                with self._lock:
                    if epoch != self._epoch: return failure('cancelled', safe)
                if store.exists(job + '.result'):
                    raw = store.read(job + '.result')
                    if (not isinstance(raw, dict)
                            or set(raw) != {'schema', 'id', 'expires_at', 'observation_sha256', 'result'}
                            or raw['schema'] != RESULT_SCHEMA or raw['id'] != job
                            or raw['expires_at'] != expires or raw['observation_sha256'] != digest):
                        raise ValueError('SO101 IPC response binding mismatch')
                    result = validate_decision(raw['result'], safe)
                    with self._lock:
                        if epoch != self._epoch: return failure('cancelled', safe)
                        if time.monotonic() >= deadline or time.time() >= expires:
                            return failure('expired', safe)
                        return result
                if not worker_status()['available'] and not store.exists(job + '.result'):
                    store.write(job + '.cancel', {'cancelled': True})
                    return failure('unavailable', safe)
                time.sleep(POLL_SECONDS)
            store.write(job + '.cancel', {'cancelled': True})
            return failure('timeout', safe)
        except (OSError, ValueError, TypeError, RecursionError):
            return failure('invalid_response', safe)
        finally:
            with self._lock: self._pending.discard(job)
            if store is not None:
                for suffix in ('.job', '.result'):
                    try: store.delete(job + suffix)
                    except OSError: pass
                store.close()

