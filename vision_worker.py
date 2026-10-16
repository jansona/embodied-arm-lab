"""Explicit bounded vision window: image-only protocol, no daemon or restart."""
from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import stat
import time

from ipc import number
from vision_adapter import VisionResult, sanitize_diagnostic
from vision_ipc import (VisionStore, MAX_CALLS, MAX_SECONDS, MAX_RESULT_BYTES,
    REQUEST_SECONDS, POLL_SECONDS, RESULT_SCHEMA, _loads, failure, frame_is_fresh,
    validate_job, validate_result)


def _invoke(adapter, packet, connection, inherited_fds):
    for fd in inherited_fds:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        try:
            result = adapter.perceive(packet)
        except BaseException:
            result = failure('adapter_error', packet)
        diagnostic_store = None
        try:
            diagnostic_store = VisionStore()
            diagnostic_store.write('last-diagnostic.json', sanitize_diagnostic(
                getattr(adapter, 'last_diagnostic', {'stage': 'worker', 'error':
                    result.reason if isinstance(result, VisionResult) else 'invalid_response'})))
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        finally:
            if diagnostic_store is not None: diagnostic_store.close()
        try:
            if not isinstance(result, VisionResult):
                raise ValueError('Invalid vision adapter result')
            safe = validate_result(result.to_dict(), packet).to_dict()
        except (ValueError, TypeError, RecursionError):
            safe = failure('invalid_response', packet).to_dict()
        raw = json.dumps(safe, allow_nan=False, separators=(',', ':')).encode()
        if len(raw) > MAX_RESULT_BYTES:
            raw = json.dumps(failure('invalid_response', packet).to_dict()).encode()
        connection.send_bytes(raw)
    except (OSError, ValueError, TypeError):
        pass
    finally:
        connection.close()


def _reap(process):
    process.join(timeout=.02)
    if process.is_alive():
        process.terminate()
        process.join(timeout=.1)
    if process.is_alive():
        process.kill()
        process.join(timeout=.1)
    process.close()


def worker_run(adapter, max_calls=MAX_CALLS, max_seconds=MAX_SECONDS):
    if type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
        raise ValueError('Vision call budget must be 1 to 6')
    if not number(max_seconds) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError('Vision time budget must be positive and at most 600 seconds')
    context = multiprocessing.get_context('fork')
    store = VisionStore()
    lock_fd = None
    locked = False
    started = time.monotonic()
    deadline = started + max_seconds
    counters = {'status': 'ready', 'calls': 0, 'processed': 0, 'rejected': 0, 'cancelled': 0,
                'elapsed_seconds': 0.0, 'max_calls': max_calls, 'max_seconds': max_seconds,
                'deadline_unix': time.time() + max_seconds}
    audit = []

    def publish(status):
        counters['status'] = status
        counters['elapsed_seconds'] = round(time.monotonic() - started, 3)
        store.write('ready.json', counters)

    try:
        store.exists('worker.lock')
        lock_fd = os.open('worker.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=store.fd)
        info = os.fstat(lock_fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            raise ValueError('Unsafe vision worker lock')
        # Status probes briefly take the unowned lock to verify availability.
        # Do not mistake that tiny startup race for an existing worker.
        lock_wait_deadline = min(deadline, time.monotonic() + .05)
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= lock_wait_deadline:
                    raise ValueError('A vision worker is already active') from None
                time.sleep(.001)
        locked = True
        store.write('last-run-audit.json', [])
        store.write('last-diagnostic.json', sanitize_diagnostic({'stage': 'idle', 'error': 'none'}))
        publish('ready')
        while counters['calls'] < max_calls and time.monotonic() < deadline:
            jobs = store.jobs()
            if not jobs:
                time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
                continue
            for name in jobs:
                if counters['calls'] >= max_calls or time.monotonic() >= deadline:
                    break
                job = name[:-4]
                try:
                    data = validate_job(store.read(name), job)
                    if not 0 < data['expires_at'] - time.time() <= REQUEST_SECONDS + .5:
                        counters['rejected'] += 1
                        store.delete(name)
                        continue
                    if store.exists(job + '.cancel'):
                        counters['cancelled'] += 1
                        store.delete(name)
                        continue
                except (OSError, ValueError, TypeError, RecursionError):
                    counters['rejected'] += 1
                    store.delete(name)
                    continue
                packet = data['packet']
                receive, send = context.Pipe(duplex=False)
                process = context.Process(target=_invoke,
                    args=(adapter, packet, send, (store.fd, lock_fd, receive.fileno())),
                    daemon=False, name='bounded-rgb-vision-call')
                result = None
                cancelled = False
                call_started = time.monotonic()
                try:
                    if store.exists(job + '.cancel') or not store.exists(name):
                        counters['cancelled'] += 1
                        store.delete(name)
                        continue
                    counters['calls'] += 1
                    store.write('last-diagnostic.json', sanitize_diagnostic({'stage': 'request', 'error': 'none'}))
                    publish('calling')
                    process.start()
                    send.close()
                    while (time.monotonic() < deadline and time.time() < data['expires_at']
                           and frame_is_fresh(packet)):
                        if store.exists(job + '.cancel') or not store.exists(name):
                            cancelled = True
                            break
                        if receive.poll(min(POLL_SECONDS, max(0, deadline - time.monotonic()))):
                            try:
                                result = validate_result(_loads(receive.recv_bytes(MAX_RESULT_BYTES)), packet)
                            except (EOFError, OSError, ValueError, TypeError, RecursionError):
                                result = failure('invalid_response', packet)
                            break
                        if not process.is_alive():
                            result = failure('adapter_error', packet)
                            break
                finally:
                    send.close()
                    receive.close()
                    if process.pid is not None:
                        _reap(process)
                    else:
                        process.close()
                cancelled = cancelled or store.exists(job + '.cancel') or not store.exists(name)
                if cancelled:
                    result = failure('cancelled', packet)
                    counters['cancelled'] += 1
                elif not frame_is_fresh(packet):
                    result = failure('stale_frame', packet)
                elif time.time() >= data['expires_at']:
                    result = failure('expired', packet)
                elif time.monotonic() >= deadline:
                    result = failure('timeout', packet)
                elif result is None:
                    result = failure('adapter_error', packet)
                safe = result.to_dict()
                safe['calls'] = counters['calls']
                safe = validate_result(safe, packet).to_dict()
                counters['processed'] += 1
                delivered = not cancelled and time.time() < data['expires_at']
                # No images, pixels, object geometry, raw errors or credentials.
                audit.append({'id': job, 'frame_sha256': data['frame_sha256'], 'accepted': safe['accepted'],
                              'reason': safe['reason'], 'calls': counters['calls'],
                              'detection_count': len(safe['detections']),
                              'latency_seconds': round(time.monotonic() - call_started, 4), 'delivered': delivered})
                store.write('last-run-audit.json', audit)
                if delivered:
                    store.write(job + '.result', {'schema': RESULT_SCHEMA, 'id': job,
                        'expires_at': data['expires_at'], 'frame_sha256': data['frame_sha256'], 'result': safe})
                store.delete(name)
                publish('ready')
        publish('call_budget' if counters['calls'] >= max_calls else 'time_budget')
        return dict(counters)
    finally:
        if lock_fd is not None:
            if locked:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run one bounded image-only Qwen vision window')
    parser.add_argument('--config', required=True, help='Explicit approved external configuration path')
    parser.add_argument('--max-calls', type=int, default=MAX_CALLS)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_calls <= MAX_CALLS or not number(args.max_seconds) or not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error('Limits must be 1 to 6 calls and greater than 0 to 600 seconds')
    try:
        from vision_adapter import QwenVision
        adapter = QwenVision.from_config(args.config, max_calls=args.max_calls,
            timeout=min(30, args.max_seconds), use_environment_network=True)
        print(json.dumps(worker_run(adapter, args.max_calls, args.max_seconds)))
    except Exception:
        print(json.dumps({'status': 'worker_error'}))
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
