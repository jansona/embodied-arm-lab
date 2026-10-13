"""Explicit, bounded Qwen brain worker for the simulation-only typed queue.

No imports or library calls load credentials. Only the CLI's explicit --config
option loads the saved external configuration. There is no daemon, automatic
restart, arbitrary destination, or fallback provider.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import stat
import time

from brain import BrainResult, MODEL
from brain_ipc import (BrainStore, MAX_CALLS, MAX_SECONDS, POLL_SECONDS,
                       REQUEST_SECONDS, REPLAN_REQUEST_SCHEMA, RESULT_SCHEMA,
                       validate_job, validate_result)
from ipc import MAX_BYTES, number


def _invoke(brain, data, connection, inherited_fds):
    # Do not let a helper retain the directory/worker lock after its parent
    # exits. The only return channel contains validated JSON, never pickle.
    for fd in inherited_fds:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        try:
            if data['schema'] == REPLAN_REQUEST_SCHEMA:
                result = brain.replan(data['text'], data['scene'], data['previous_plan'],
                                      data['feedback'], history=data['history'])
            else:
                result = brain.interpret(data['text'], data['scene'], data['history'])
            if not isinstance(result, BrainResult):
                raise ValueError('Invalid brain adapter result')
            safe = validate_result(result.to_dict(), data['scene']).to_dict()
        except BaseException:
            safe = BrainResult(reason='adapter_error').to_dict()
        raw = json.dumps(safe, allow_nan=False, separators=(',', ':')).encode()
        if len(raw) > MAX_BYTES:
            raw = json.dumps(BrainResult(reason='invalid_response').to_dict()).encode()
        connection.send_bytes(raw)
    except (OSError, ValueError, TypeError):
        pass
    finally:
        connection.close()


def _reap(process):
    """No helper is left running after cancellation or a bounded worker exit."""
    process.join(timeout=.02)
    if process.is_alive():
        process.terminate()
        process.join(timeout=.1)
    if process.is_alive():
        process.kill()
        process.join(timeout=.1)
    process.close()


def worker_run(brain, max_calls=MAX_CALLS, max_seconds=MAX_SECONDS):
    if type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
        raise ValueError('Call budget must be 1 to 10')
    if not number(max_seconds) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError('Time budget must be positive and at most 600 seconds')
    # Linux is explicit, rather than silently switching to an executor whose
    # serialization or lifetime rules differ. No worker-supplied code is run.
    context = multiprocessing.get_context('fork')
    store = BrainStore()
    lock_fd = None
    locked = False
    started = time.monotonic()
    deadline = started + max_seconds
    counters = {'status': 'ready', 'calls': 0, 'processed': 0, 'rejected': 0,
                'cancelled': 0, 'elapsed_seconds': 0.0, 'max_calls': max_calls,
                'replans': 0, 'max_replans': 1,
                'max_seconds': max_seconds, 'deadline_unix': time.time() + max_seconds}
    audit = []

    def publish(status):
        counters['status'] = status
        counters['elapsed_seconds'] = round(time.monotonic() - started, 3)
        store.write('ready.json', counters)

    try:
        lock_fd = os.open('worker.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                          0o600, dir_fd=store.fd)
        info = os.fstat(lock_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('Unsafe brain worker lock')
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A brain worker is already active') from None
        locked = True
        store.write('last-run-audit.json', [])
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
                    ttl = data['expires_at'] - time.time()
                    if not 0 < ttl <= REQUEST_SECONDS + .5:
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
                is_replan = data['schema'] == REPLAN_REQUEST_SCHEMA
                if is_replan and counters['replans'] >= 1:
                    # The parent worker owns this allowance, so fresh bridges,
                    # raw jobs and forked adapter copies cannot reset it.
                    counters['rejected'] += 1
                    store.write(job + '.result', {'schema': RESULT_SCHEMA, 'id': job,
                        'expires_at': data['expires_at'], 'result': BrainResult(
                            reason='replan_budget_exhausted', calls=counters['calls']).to_dict()})
                    store.delete(name)
                    continue
                receive, send = context.Pipe(duplex=False)
                process = context.Process(target=_invoke,
                    args=(brain, data, send,
                          (store.fd, lock_fd, receive.fileno())),
                    daemon=False, name='bounded-simulation-brain-call')
                result = None
                cancelled = False
                call_started = time.monotonic()
                try:
                    # A second check closes the interval spent validating and
                    # preparing the helper before invoking the remote adapter.
                    if store.exists(job + '.cancel'):
                        counters['cancelled'] += 1
                        store.delete(name)
                        continue
                    counters['calls'] += 1
                    if is_replan:
                        counters['replans'] += 1
                    publish('calling')
                    process.start()
                    send.close()
                    while time.monotonic() < deadline and time.time() < data['expires_at']:
                        if store.exists(job + '.cancel') or not store.exists(name):
                            cancelled = True
                            break
                        if receive.poll(min(POLL_SECONDS, max(0, deadline - time.monotonic()))):
                            try:
                                raw = receive.recv_bytes(MAX_BYTES)
                                result = validate_result(json.loads(raw), data['scene'])
                            except (EOFError, OSError, ValueError, TypeError, RecursionError):
                                result = BrainResult(reason='invalid_response')
                            break
                        if not process.is_alive():
                            result = BrainResult(reason='adapter_error')
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
                    result = BrainResult(reason='cancelled')
                    counters['cancelled'] += 1
                elif time.time() >= data['expires_at']:
                    result = BrainResult(reason='expired')
                elif time.monotonic() >= deadline:
                    result = BrainResult(reason='timeout')
                elif result is None:
                    result = BrainResult(reason='adapter_error')
                safe = result.to_dict()
                safe['calls'] = counters['calls']
                safe = validate_result(safe, data['scene']).to_dict()
                counters['processed'] += 1
                delivered = not cancelled and time.time() < data['expires_at']
                # No user request, conversation, model message, exception text,
                # HTTP data, endpoint, headers or credentials are logged.
                plan = safe['plan'] or {}
                audit.append({'id': job, 'accepted': safe['accepted'], 'reason': safe['reason'],
                              'operation': 'replan' if is_replan else 'interpret',
                              'model': MODEL, 'calls': counters['calls'],
                              'provenance': safe['provenance'], 'action': plan.get('action'),
                              'object_id': plan.get('object_id'), 'target_id': plan.get('target_id'),
                              'latency_seconds': round(time.monotonic() - call_started, 4),
                              'delivered': delivered})
                store.write('last-run-audit.json', audit)
                if delivered:
                    store.write(job + '.result', {'schema': RESULT_SCHEMA, 'id': job,
                                'expires_at': data['expires_at'], 'result': safe})
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
    parser = argparse.ArgumentParser(description='Run one bounded simulation-only Qwen brain window')
    parser.add_argument('--config', required=True, help='Explicit saved configuration outside the project')
    parser.add_argument('--max-calls', type=int, default=MAX_CALLS)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    # Validate limits before opening the explicit saved configuration. Never
    # discover credentials, echo paths/configuration or accept a destination.
    if not 1 <= args.max_calls <= MAX_CALLS or not number(args.max_seconds) or not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error('Limits must be 1 to 10 calls and greater than 0 to 600 seconds')
    try:
        from brain import QwenBrain
        brain = QwenBrain.from_config(args.config, max_calls=args.max_calls,
                                     timeout=min(30, args.max_seconds), use_environment_network=True)
        print(json.dumps(worker_run(brain, args.max_calls, args.max_seconds)))
    except Exception:
        print(json.dumps({'status': 'worker_error'}))
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
