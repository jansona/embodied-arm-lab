"""One explicit visual-loop session. No daemon renewal, retries or fallback.

A pooled inference child serves Qwen alone by default, or both fixed providers
only in explicit two_model mode. Stop terminates/reaps that
child; only a later explicit job may create another, within the SAME deadline
and cumulative budgets. Expiry or child failure closes the session window.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import multiprocessing
import os
import signal
import time
import uuid
from contextlib import ExitStack

from embodied_arm.providers.request_telemetry import worker_telemetry, request_audit_fields
from embodied_arm.runtime.visual_loop_ipc import (VisualStore, COMPONENTS, MODES, MAX_CALLS, MAX_SECONDS,
    MAX_BYTES, MAX_RESULT_BYTES, REQUEST_SECONDS, POLL_SECONDS, RESULT_SCHEMA, _json, _loads,
    number, failure, fresh, normalize_payload, validate_job, validate_result)


def _inference_loop(adapters, connection, inherited_fds):
    # Child termination must not inherit the CLI parent's graceful handler.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    for fd in inherited_fds:
        try: os.close(fd)
        except OSError: pass
    try:
        while True:
            command = _loads(connection.recv_bytes(MAX_BYTES))
            if command == {'stop': True}: break
            if not isinstance(command, dict) or set(command) != {'id', 'component', 'payload'}: break
            component = command['component']
            payload = normalize_payload(component, command['payload'])
            adapter = adapters[component]
            try:
                if hasattr(adapter, 'last_request'): adapter.last_request = None
                if component == 'qwen': result = adapter.observe(**payload)
                else: result = adapter.decide(**payload)
            except BaseException:
                result = failure(component, 'worker_error')
            try:
                result = validate_result(component, result, payload)
            except (ValueError, TypeError, RecursionError):
                result = failure(component, 'invalid_response')
            metadata_component = 'brain' if component == 'qwen' else 'cerebellum'
            raw = _json({'id': command['id'], 'result': result,
                         'telemetry': worker_telemetry(getattr(adapter, 'last_request', None), metadata_component)})
            if len(raw) > MAX_RESULT_BYTES:
                raw = _json({'id': command['id'], 'result': failure(component, 'invalid_response'), 'telemetry': None})
            connection.send_bytes(raw)
    except (EOFError, OSError, ValueError, TypeError, RecursionError): pass
    finally:
        connection.close()
        for adapter in adapters.values():
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


def worker_run(qwen, jev=None, max_calls=MAX_CALLS, max_seconds=MAX_SECONDS, *, mode='qwen_direct'):
    """max_calls bounds actions; Qwen gets one final observation beyond that.

    Direct mode never retains or invokes a supplied Jev adapter. Runtime action
    execution keeps its own action cap; this worker only budgets provider calls.
    """
    if not isinstance(mode, str) or mode not in MODES: raise ValueError('Invalid visual control mode')
    if mode == 'two_model' and jev is None: raise ValueError('Two-model mode requires Jev')
    if type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
        raise ValueError('Action budget must be 1 to 200')
    if not number(max_seconds) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError('Session time budget must be positive and at most 600 seconds')
    adapters = {'qwen': qwen}
    if mode == 'two_model': adapters['jev'] = jev
    context = multiprocessing.get_context('fork')
    store = VisualStore()
    lock_fd = None
    locked = False
    process = connection = None
    started = time.monotonic()
    deadline = started + max_seconds
    counters = {'status': 'ready', 'mode': mode, 'session_id': uuid.uuid4().hex, 'active_component': None,
                'calls': {c: 0 for c in COMPONENTS}, 'max_calls': {'qwen': max_calls + 1, 'jev': max_calls if mode == 'two_model' else 0},
                'processed': 0, 'rejected': 0, 'cancelled': 0, 'max_seconds': max_seconds,
                'elapsed_seconds': 0., 'deadline_unix': time.time() + max_seconds}
    audit = []
    terminal = None

    def publish(status, component=None):
        counters.update(status=status, active_component=component,
                        elapsed_seconds=round(time.monotonic() - started, 3))
        store.write('ready.json', counters)

    try:
        store.exists('worker.lock')
        lock_fd = os.open('worker.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                          0o600, dir_fd=store.fd)
        if not store._safe(os.fstat(lock_fd)): raise ValueError('Unsafe visual worker lock')
        lock_wait = min(deadline, time.monotonic() + .05)
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= lock_wait: raise ValueError('A visual worker is already active') from None
                time.sleep(.001)
        locked = True
        store.write('last-run-audit.json', [])
        publish('ready')
        while not terminal and time.monotonic() < deadline:
            if all(counters['calls'][c] >= counters['max_calls'][c] for c in COMPONENTS):
                terminal = 'call_budget'
                break
            jobs = store.jobs()
            if not jobs:
                time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
                continue
            for name in jobs:
                if terminal or time.monotonic() >= deadline: break
                job = name[:-4]
                try:
                    data = validate_job(store.read(name), job, counters['session_id'])
                    component = data['component']
                    if (not 0 < data['expires_at'] - time.time() <= REQUEST_SECONDS + .5
                            or counters['calls'][component] >= counters['max_calls'][component]):
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
                payload = data['payload']
                if store.exists(job + '.cancel') or not store.exists(name):
                    counters['cancelled'] += 1
                    store.delete(name)
                    continue
                if process is None:
                    connection, child_connection = context.Pipe(duplex=True)
                    process = context.Process(target=_inference_loop,
                        args=(adapters, child_connection,
                              (store.fd, lock_fd, connection.fileno())),
                        daemon=False, name='bounded-visual-models')
                    process.start()
                    child_connection.close()
                result = telemetry = None
                cancelled = False
                call_started = time.monotonic()
                counters['calls'][component] += 1
                publish('calling', component)
                try:
                    connection.send_bytes(_json({'id': job, 'component': component, 'payload': payload}))
                    while (time.monotonic() < deadline and time.time() < data['expires_at']
                           and fresh(component, payload)):
                        if store.exists(job + '.cancel') or not store.exists(name):
                            cancelled = True
                            break
                        if connection.poll(min(POLL_SECONDS, max(0, deadline - time.monotonic()))):
                            raw = _loads(connection.recv_bytes(MAX_RESULT_BYTES))
                            if not isinstance(raw, dict) or set(raw) != {'id', 'result', 'telemetry'} or raw['id'] != job:
                                raise ValueError('Invalid visual helper response')
                            result = validate_result(component, raw['result'], payload)
                            telemetry = worker_telemetry(raw['telemetry'], 'brain' if component == 'qwen' else 'cerebellum')
                            break
                        if not process.is_alive():
                            terminal = 'worker_error'
                            break
                except (EOFError, OSError, ValueError, TypeError, RecursionError): terminal = 'worker_error'
                cancelled = cancelled or store.exists(job + '.cancel') or not store.exists(name)
                if cancelled:
                    # Stop invalidates the in-flight helper, not the session budget.
                    # A replacement is created only after another explicit job.
                    connection.close()
                    _reap(process)
                    connection = process = None
                    terminal = None
                    counters['cancelled'] += 1
                    result = failure(component, 'cancelled')
                elif not fresh(component, payload):
                    terminal = 'stale_frame'
                    result = failure(component, 'stale_frame')
                elif time.time() >= data['expires_at']:
                    terminal = 'expired'
                    result = failure(component, 'expired')
                elif time.monotonic() >= deadline:
                    terminal = 'time_budget'
                    result = failure(component, 'timeout')
                elif result is None: result = failure(component, 'worker_error')
                if component == 'qwen': result['calls'] = counters['calls'][component]
                result = validate_result(component, result, payload)
                counters['processed'] += 1
                delivered = not cancelled and time.time() < data['expires_at'] and time.monotonic() < deadline
                audit.append({'id': job, 'component': component, 'payload_sha256': data['payload_sha256'],
                    'accepted': result['accepted'], 'reason': result['reason'], 'calls': counters['calls'][component],
                    'latency_seconds': round(time.monotonic() - call_started, 4), 'delivered': delivered,
                    **request_audit_fields(telemetry, 'brain' if component == 'qwen' else 'cerebellum')})
                store.write('last-run-audit.json', audit)
                if delivered:
                    binding = {k: data[k] for k in ('id', 'session_id', 'client_id', 'generation',
                                                    'component', 'expires_at', 'payload_sha256')}
                    store.write(job + '.result', {**binding, 'schema': RESULT_SCHEMA, 'result': result})
                store.delete(name)
                if not terminal: publish('ready')
        publish(terminal or 'time_budget')
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
    parser = argparse.ArgumentParser(description='Run one bounded Qwen-direct or explicit two-model simulation window')
    parser.add_argument('--config', required=True, help='Explicit approved external split configuration path')
    parser.add_argument('--mode', choices=MODES, default='qwen_direct')
    parser.add_argument('--max-calls', type=int, default=MAX_CALLS, help='Action cap; Qwen gets one additional final observation')
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    if (not 1 <= args.max_calls <= MAX_CALLS or not number(args.max_seconds)
            or not 0 < args.max_seconds <= MAX_SECONDS):
        parser.error('Limits must be 1 to 200 actions and greater than 0 to 600 seconds')
    def stopped(*_):
        raise KeyboardInterrupt
    old_handlers = {sig: signal.signal(sig, stopped) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        from embodied_arm.providers.visual_brain import QwenVisualBrain
        with ExitStack() as stack:
            qwen = QwenVisualBrain.from_config(args.config, max_calls=args.max_calls + 1,
                    timeout=min(30, args.max_seconds), use_environment_network=True)
            stack.callback(qwen.close)
            jev = None
            if args.mode == 'two_model':
                from embodied_arm.providers.visual_cerebellum import VisualJev
                jev = VisualJev.from_config(args.config, max_calls=args.max_calls,
                        timeout=min(15, args.max_seconds), use_environment_network=True)
                stack.callback(jev.close)
            print(json.dumps(worker_run(qwen, jev, args.max_calls, args.max_seconds, mode=args.mode)))
    except KeyboardInterrupt:
        print(json.dumps({'status': 'stopped'}))
    except Exception:
        print(json.dumps({'status': 'worker_error'}))
        raise SystemExit(1) from None
    finally:
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__': main()
