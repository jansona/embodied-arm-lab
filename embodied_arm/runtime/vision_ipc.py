"""Private image-only IPC; no credentials, destination override, or robot state.

RGB frames and all replies stay bound to the original image packet. This module
does not load configuration, launch workers, or make network requests.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
from embodied_arm.paths import RUNTIME_ROOT
import re
import stat
import threading
import time
import uuid

from embodied_arm.providers.brain import ENDPOINT, MODEL, PROVIDER_ERROR_REASONS
from embodied_arm.runtime.ipc import Store, number
from embodied_arm.providers.vision_adapter import VisionResult, frame_snapshot, validate_detections

VISION_IPC_ROOT = RUNTIME_ROOT / 'vision-ipc'
MAX_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 16384
MAX_CALLS = 6
MAX_SECONDS = 600
MAX_FRAME_AGE = 60
REQUEST_SECONDS = 45
POLL_SECONDS = .02
REQUEST_SCHEMA = 'simulation-rgb-request-v1'
RESULT_SCHEMA = 'simulation-rgb-result-v1'
_NAME = re.compile(r'(?:[0-9a-f]{32}\.(?:job|result|cancel|tmp)|ready\.json|last-run-audit\.json|last-diagnostic\.json|worker\.lock)\Z')
_REASONS = frozenset(('accepted', 'unavailable', 'busy', 'invalid_or_stale_frame',
    'stale_frame', 'call_budget_exhausted', 'timeout', 'network_error', 'response_too_large',
    'unexpected_model', 'invalid_response', 'incomplete_response', 'occluded_or_missing_feature',
    'low_confidence', 'adapter_error', 'worker_error', 'cancelled', 'expired')) | PROVIDER_ERROR_REASONS


def _loads(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _json(data):
    return json.dumps(data, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


class VisionStore(Store):
    """Independent 1 MiB store; the legacy 32 KiB store remains unchanged."""

    def __init__(self):
        super().__init__(VISION_IPC_ROOT)

    def _name(self, name):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError('Invalid vision IPC name')

    def exists(self, name):
        self._name(name)
        try:
            info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            raise ValueError('Unsafe vision IPC file')
        return True

    def write(self, name, data):
        self._name(name)
        self.exists(name)
        raw = _json(data)
        if len(raw) > MAX_BYTES:
            raise ValueError('Vision IPC payload too large')
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
                raise ValueError('Unsafe vision IPC file')
            raw = file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('Vision IPC payload too large')
        return _loads(raw)


def normalize_packet(packet):
    """Exactly two bounded RGB JPEGs; reject state, URLs and extra metadata."""
    if not isinstance(packet, dict) or set(packet) != {'frame_id', 'captured_at', 'images'}:
        raise ValueError('Invalid image-only packet fields')
    images = packet['images']
    if (not isinstance(images, list) or len(images) != 2
            or any(not isinstance(i, dict) or set(i) != {'camera_id', 'width', 'height', 'data_url'}
                   for i in images)):
        raise ValueError('Invalid image-only camera fields')
    if len(_json(packet)) > MAX_BYTES - 1024:
        raise ValueError('Image packet too large')
    safe = frame_snapshot(packet, max_frame_age=MAX_FRAME_AGE)
    for image in safe['images']:
        raw = base64.b64decode(image['data_url'][len('data:image/jpeg;base64,'):], validate=True)
        if not raw.startswith(b'\xff\xd8') or not raw.endswith(b'\xff\xd9'):
            raise ValueError('JPEG packet must contain only complete JPEG data')
    return safe


def packet_hash(packet):
    return hashlib.sha256(_json(packet)).hexdigest()


def frame_is_fresh(packet):
    age = time.monotonic() - packet['captured_at']
    return 0 <= age <= MAX_FRAME_AGE


def failure(reason, packet=None, calls=0):
    return VisionResult(reason=reason, calls=calls,
                        frame_id=packet['frame_id'] if packet else None,
                        captured_at=packet['captured_at'] if packet else None)


def validate_job(data, job):
    fields = {'schema', 'simulation_only', 'id', 'expires_at', 'frame_sha256', 'packet'}
    if (not isinstance(data, dict) or set(data) != fields or data['schema'] != REQUEST_SCHEMA
            or data['simulation_only'] is not True or data['id'] != job
            or not number(data['expires_at'])):
        raise ValueError('Invalid vision request')
    safe = normalize_packet(data['packet'])
    if data['packet'] != safe or data['frame_sha256'] != packet_hash(safe):
        raise ValueError('Unfiltered or mismatched RGB packet')
    return data


def validate_result(data, packet):
    fields = {'accepted', 'reason', 'frame_id', 'captured_at', 'detections', 'model', 'calls', 'provenance'}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError('Invalid vision response fields')
    reason = data['reason']
    if (type(data['accepted']) is not bool or data['model'] != MODEL or data['provenance'] != 'qwen_rgb'
            or type(data['calls']) is not int or not 0 <= data['calls'] <= MAX_CALLS
            or not isinstance(reason, str)
            or (reason not in _REASONS and not re.fullmatch(r'http_error_[1-5][0-9]{2}', reason))):
        raise ValueError('Invalid vision response values')
    same_frame = (data['frame_id'] == packet['frame_id'] and number(data['captured_at'])
                  and data['captured_at'] == packet['captured_at'])
    if not same_frame and not (not data['accepted'] and data['frame_id'] is None and data['captured_at'] is None):
        raise ValueError('Vision response is for a different frame')
    if data['accepted']:
        if reason != 'accepted' or not same_frame:
            raise ValueError('Inconsistent accepted vision response')
        detections = validate_detections({'frame_id': packet['frame_id'], 'detections': data['detections']}, packet,
                                         min_confidence=.75, max_uncertainty_px=25)
        if any(d.visibility != 'visible' or d.confidence < .75 for d in detections):
            raise ValueError('Accepted response must have six visible confident correspondences')
    else:
        if reason == 'accepted' or data['detections'] != []:
            raise ValueError('Rejected vision response cannot expose detections')
        detections = ()
    return VisionResult(data['accepted'], reason, packet['frame_id'], packet['captured_at'],
                        detections, MODEL, data['calls'], 'qwen_rgb')


def worker_status():
    safe = {'configured': False, 'available': False, 'status': 'offline', 'busy': False,
            'model': MODEL, 'endpoint': ENDPOINT, 'calls': 0, 'max_calls': 0,
            'remaining_calls': 0, 'remaining_seconds': 0.0, 'provenance': 'qwen_rgb',
            'ground_truth_input': False, 'fallback': False}
    store = None
    fd = None
    try:
        store = VisionStore()
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
                or not 0 <= calls <= limit
                or not number(deadline)):
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
        if fd is not None:
            os.close(fd)
        if store is not None:
            store.close()


class FileVisionBridge:
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
            if not self._pending:
                return
            store = None
            try:
                store = VisionStore()
                for job in self._pending:
                    store.write(job + '.cancel', {'cancelled': True})
            except (OSError, ValueError, TypeError):
                pass
            finally:
                if store is not None:
                    store.close()

    def perceive(self, packet):
        if not self._call_lock.acquire(blocking=False):
            return failure('busy')
        try:
            with self._lock:
                epoch = self._epoch
            return self._perceive(packet, epoch)
        finally:
            self._call_lock.release()

    def _perceive(self, packet, epoch):
        try:
            safe = normalize_packet(packet)
        except (OSError, ValueError, TypeError, OverflowError, RecursionError):
            return failure('invalid_or_stale_frame')
        with self._lock:
            if epoch != self._epoch:
                return failure('cancelled', safe)
        status = worker_status()
        if not status['available']:
            return failure('unavailable', safe)
        if status['remaining_calls'] <= 0:
            return failure('busy', safe)
        store = None
        job = uuid.uuid4().hex
        digest = packet_hash(safe)
        deadline = time.monotonic() + REQUEST_SECONDS
        expires = time.time() + REQUEST_SECONDS
        try:
            store = VisionStore()
            with self._lock:
                if epoch != self._epoch:
                    return failure('cancelled', safe)
                self._pending.add(job)
                store.write(job + '.job', {'schema': REQUEST_SCHEMA, 'simulation_only': True, 'id': job,
                    'expires_at': expires, 'frame_sha256': digest, 'packet': safe})
            while time.monotonic() < deadline:
                with self._lock:
                    if epoch != self._epoch:
                        return failure('cancelled', safe)
                if not frame_is_fresh(safe):
                    store.write(job + '.cancel', {'cancelled': True})
                    return failure('stale_frame', safe)
                if store.exists(job + '.result'):
                    raw = store.read(job + '.result')
                    if (not isinstance(raw, dict)
                            or set(raw) != {'schema', 'id', 'expires_at', 'frame_sha256', 'result'}
                            or raw['schema'] != RESULT_SCHEMA or raw['id'] != job
                            or raw['expires_at'] != expires or raw['frame_sha256'] != digest):
                        raise ValueError('Vision response envelope mismatch')
                    result = validate_result(raw['result'], safe)
                    with self._lock:
                        if epoch != self._epoch:
                            return failure('cancelled', safe)
                        if time.monotonic() >= deadline or time.time() >= expires:
                            return failure('expired', safe)
                        if not frame_is_fresh(safe):
                            return failure('stale_frame', safe)
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
            with self._lock:
                self._pending.discard(job)
            if store is not None:
                for suffix in ('.job', '.result'):
                    try:
                        store.delete(job + suffix)
                    except OSError:
                        pass
                store.close()
