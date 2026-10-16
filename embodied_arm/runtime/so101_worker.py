"""One explicit SO101 window, with a pooled inference child and no restart.

The child owns one adapter/client for the whole bounded window. Cancellation,
expiry, or a stuck request terminates it and closes the window, never retries.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import multiprocessing
import os
import stat
import time
from embodied_arm.providers.request_telemetry import worker_telemetry, request_audit_fields
from embodied_arm.runtime.ipc import number
from embodied_arm.providers.so101_jev import SO101Decision, validate_observation, validate_decision
from embodied_arm.runtime.so101_ipc import (SO101Store, MAX_BYTES, MAX_CALLS, MAX_SECONDS, DEFAULT_CALLS, DEFAULT_SECONDS, POLL_SECONDS,
    REQUEST_SECONDS, RESULT_SCHEMA, _loads, _json, failure, validate_job)


def _inference_loop(adapter, connection, inherited_fds):
    for fd in inherited_fds:
        try: os.close(fd)
        except OSError: pass
    try:
        while True:
            command = _loads(connection.recv_bytes(MAX_BYTES))
            if command == {'stop': True}: break
            if not isinstance(command, dict) or set(command) != {'id', 'observation'}:
                break
            observation = validate_observation(command['observation'])
            try:
                if hasattr(adapter, 'last_request'): adapter.last_request = None
                result = adapter.decide(observation)
            except BaseException:
                result = failure('adapter_error', observation)
            try:
                if not isinstance(result, SO101Decision):
                    raise ValueError('Invalid SO101 adapter result')
                result = validate_decision(result.to_dict(), observation)
            except (ValueError, TypeError, RecursionError):
                result = failure('invalid_response', observation)
            connection.send_bytes(_json({'id': command['id'], 'result': result.to_dict(),
                'telemetry': worker_telemetry(getattr(adapter, 'last_request', None), 'cerebellum')}))
    except (EOFError, OSError, ValueError, TypeError, RecursionError):
        pass
    finally:
        connection.close()
        if hasattr(adapter, 'close'):
            try: adapter.close()
            except Exception: pass


def _reap(process):
    process.join(timeout=.02)
    if process.is_alive():
        process.terminate()
        process.join(timeout=.1)
    if process.is_alive():
        process.kill()
        process.join(timeout=.1)
    process.close()


def worker_run(adapter, max_calls=DEFAULT_CALLS, max_seconds=DEFAULT_SECONDS):
    if type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
        raise ValueError('SO101 call budget must be 1 to 200')
    if not number(max_seconds) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError('SO101 time budget must be positive and at most 420 seconds')
    context = multiprocessing.get_context('fork')
    store = SO101Store()
    lock_fd = None
    locked = False
    process = None
    connection = None
    started = time.monotonic()
    deadline = started + max_seconds
    counters = {'status': 'ready', 'calls': 0, 'processed': 0, 'rejected': 0, 'cancelled': 0,
                'max_calls': max_calls, 'max_seconds': max_seconds, 'elapsed_seconds': 0.,
                'deadline_unix': time.time() + max_seconds}
    audit = []
    terminal = None

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
            raise ValueError('Unsafe SO101 worker lock')
        lock_wait_deadline = min(deadline, time.monotonic() + .05)
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= lock_wait_deadline:
                    raise ValueError('An SO101 worker is already active') from None
                time.sleep(.001)
        locked = True
        store.write('last-run-audit.json', [])
        publish('ready')
        while not terminal and counters['calls'] < max_calls and time.monotonic() < deadline:
            jobs = store.jobs()
            if not jobs:
                time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
                continue
            for name in jobs:
                if terminal or counters['calls'] >= max_calls or time.monotonic() >= deadline: break
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
                observation = data['observation']
                if store.exists(job + '.cancel') or not store.exists(name):
                    counters['cancelled'] += 1
                    store.delete(name)
                    continue
                if process is None:
                    connection, child_connection = context.Pipe(duplex=True)
                    process = context.Process(target=_inference_loop,
                        args=(adapter, child_connection, (store.fd, lock_fd, connection.fileno())),
                        daemon=False, name='bounded-pooled-so101-jev')
                    process.start()
                    child_connection.close()
                result = None
                telemetry = None
                cancelled = False
                call_started = time.monotonic()
                counters['calls'] += 1
                publish('calling')
                try:
                    connection.send_bytes(_json({'id': job, 'observation': observation}))
                    while time.monotonic() < deadline and time.time() < data['expires_at']:
                        if store.exists(job + '.cancel') or not store.exists(name):
                            cancelled = True
                            break
                        if connection.poll(min(POLL_SECONDS, max(0, deadline - time.monotonic()))):
                            raw = _loads(connection.recv_bytes(MAX_BYTES))
                            if not isinstance(raw, dict) or set(raw) != {'id', 'result', 'telemetry'} or raw['id'] != job:
                                raise ValueError('Invalid SO101 helper response')
                            result = validate_decision(raw['result'], observation)
                            telemetry = worker_telemetry(raw['telemetry'], 'cerebellum')
                            break
                        if not process.is_alive():
                            terminal = 'worker_error'
                            result = failure('worker_error', observation)
                            break
                except (EOFError, OSError, ValueError, TypeError, RecursionError):
                    terminal = 'worker_error'
                    result = failure('worker_error', observation)
                cancelled = cancelled or store.exists(job + '.cancel') or not store.exists(name)
                if cancelled:
                    terminal = 'cancelled'
                    counters['cancelled'] += 1
                    result = failure('cancelled', observation)
                elif time.time() >= data['expires_at']:
                    terminal = 'expired'
                    result = failure('expired', observation)
                elif time.monotonic() >= deadline:
                    terminal = 'time_budget'
                    result = failure('timeout', observation)
                elif result is None:
                    terminal = 'worker_error'
                    result = failure('worker_error', observation)
                safe = validate_decision(result.to_dict(), observation).to_dict()
                counters['processed'] += 1
                delivered = not cancelled and time.time() < data['expires_at']
                audit.append({'id': job, 'observation_sha256': data['observation_sha256'],
                    **request_audit_fields(telemetry, 'cerebellum'),
                    'accepted': safe['accepted'], 'reason': safe['reason'], 'action': safe['action'],
                    'proposed_choice': safe['proposed_choice'], 'confidence': safe['confidence'],
                    'axis_index': safe['axis_index'], 'calls': counters['calls'],
                    'latency_seconds': round(time.monotonic() - call_started, 4), 'delivered': delivered})
                store.write('last-run-audit.json', audit)
                if delivered:
                    store.write(job + '.result', {'schema': RESULT_SCHEMA, 'id': job,
                        'expires_at': data['expires_at'], 'observation_sha256': data['observation_sha256'], 'result': safe})
                store.delete(name)
                if not terminal: publish('ready')
        publish(terminal or ('call_budget' if counters['calls'] >= max_calls else 'time_budget'))
        return dict(counters)
    finally:
        if connection is not None:
            if process is not None and process.is_alive():
                try: connection.send_bytes(_json({'stop': True}))
                except (OSError, ValueError): pass
            connection.close()
        if process is not None:
            if process.pid is not None: _reap(process)
            else: process.close()
        if lock_fd is not None:
            if locked: fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run one bounded SO101-specific Jev window')
    parser.add_argument('--config', required=True, help='Explicit approved external configuration path')
    parser.add_argument('--max-calls', type=int, default=DEFAULT_CALLS)
    parser.add_argument('--max-seconds', type=float, default=DEFAULT_SECONDS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_calls <= MAX_CALLS or not number(args.max_seconds) or not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error('Limits must be 1 to 200 calls and greater than 0 to 420 seconds')
    try:
        from embodied_arm.providers.so101_jev import SO101JevAdapter
        with SO101JevAdapter.from_config(args.config, max_calls=args.max_calls,
                timeout=min(15, args.max_seconds), use_environment_network=True) as adapter:
            print(json.dumps(worker_run(adapter, args.max_calls, args.max_seconds)))
    except Exception:
        print(json.dumps({'status': 'worker_error'}))
        raise SystemExit(1) from None


if __name__ == '__main__': main()
