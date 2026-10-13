"""Opt-in background Jev controller for the local simulation only.

Configure from an explicitly selected JSON file outside the project; this module
never searches for credentials, loads environment variables, or calls an API
until start(). HTTP calls run in a daemon worker, never the server request thread.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time

from adapters import Decision, JevAdapter, OFFICIAL_ENDPOINT, ALIYUN_ENDPOINT, ALIYUN_MODEL
from control import apply_action, observation


class JevRunner:
    def __init__(self, sim, adapter=None, use_environment_network=False):
        self.sim = sim
        self.use_environment_network=use_environment_network
        self._adapter = adapter
        self._lock = threading.RLock()
        self._epoch = 0
        self._thread = None
        self._timer = None
        self._running = False
        self._started = None
        self._finished = None
        self._deadline = None
        self._reason = 'ready' if adapter is not None else 'not_configured'
        self._calls = 0
        self._applied = 0
        self._max_calls = 0
        self._max_seconds = 0
        self._latency = None
        self._last_action = None
        self._last_confidence = None
        self._last_decision = None

    def configure(self, path):
        """Read only this explicit external JSON file. Never return its contents.

        Format: {"provider":"aliyun_bailian", "model":"decision-model-preview",
                 "endpoint": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone",
                 "api_key": "...", "min_confidence": 0.65, "timeout": 8}
        Missing/empty api_key leaves the runner unconfigured. Do not put this
        file in a repository or expose this method to untrusted path input.
        """
        with self._lock:
            if self._running or (self._thread and self._thread.is_alive()):
                raise ValueError('Stop and wait for the pending request before configuring')
            if not isinstance(path, (str, Path)) or not str(path).strip():
                raise ValueError('An explicit external configuration path is required')
            try:
                config_path = Path(path).expanduser().resolve()
                project = Path(__file__).resolve().parent
                if config_path == project or project in config_path.parents:
                    raise ValueError('Configuration must be outside the project directory')
                with config_path.open('rb') as file:
                    raw = file.read(16385)
                if len(raw) > 16384:
                    raise ValueError('Configuration is too large')
                config = json.loads(raw)
            except (OSError, ValueError, TypeError, RecursionError):
                raise ValueError('Cannot load the explicit external configuration file') from None
            if not isinstance(config, dict) or set(config) - {'provider', 'endpoint', 'model', 'api_key', 'min_confidence', 'timeout'}:
                raise ValueError('Invalid Jev configuration fields')
            endpoint = config.get('endpoint')
            model = config.get('model')
            if config.get('provider') != 'aliyun_bailian' or endpoint != ALIYUN_ENDPOINT or model != ALIYUN_MODEL:
                raise ValueError('Configuration must explicitly select the approved Aliyun provider, endpoint and decision-model-preview')
            key = config.get('api_key')
            if key is None or key == '':
                self._adapter = None
                self._reason = 'not_configured'
            else:
                self._adapter = JevAdapter(api_key=key, endpoint=endpoint, model=model,
                                           min_confidence=config.get('min_confidence', 0.65),
                                           timeout=config.get('timeout', 8.0),use_environment_network=self.use_environment_network)
                self._reason = 'ready'
            return self.status()

    def smoke_test(self):
        """Explicit API-only check using the configured adapter; never actuates."""
        self.stop()
        with self._lock:
            if self._adapter is None:raise ValueError('Jev is not configured')
            if self._thread and self._thread.is_alive():raise ValueError('A request is pending')
            adapter=self._adapter
        return adapter.smoke_test()

    def status(self):
        with self._lock:
            elapsed = 0.0 if self._started is None else (self._finished or time.monotonic()) - self._started
            return {'provider':'aliyun_bailian','model':ALIYUN_MODEL,'endpoint':ALIYUN_ENDPOINT,'configured': self._adapter is not None, 'running': self._running,
                    'pending': bool(self._thread and self._thread.is_alive()),
                    'status': self._reason, 'reason': self._reason, 'origin': 'jev',
                    'calls': self._calls, 'actions_applied': self._applied,
                    'max_calls': self._max_calls, 'max_seconds': self._max_seconds,
                    'elapsed_seconds': round(max(0.0, elapsed), 3),
                    'last_call_seconds': self._latency,
                    'last_action': self._last_action,
                    'last_confidence': self._last_confidence,'last_decision':self._last_decision}

    def start(self, max_calls=30, max_seconds=60):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 1 <= max_calls <= 1000:
            raise ValueError('Call budget must be an integer between 1 and 1000')
        if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                or not 0 < max_seconds <= 300 or not math.isfinite(max_seconds)):
            raise ValueError('Time budget must be greater than zero and at most 300 seconds')
        with self._lock:
            if self._adapter is None:
                raise ValueError('Jev is not configured; supply an explicit server-side configuration first')
            if self._running or (self._thread and self._thread.is_alive()):
                raise ValueError('A Jev run or pending request is already active')
            with self.sim.lock:
                if self.sim.state()['busy']:
                    raise ValueError('Stop existing motion before starting Jev')
                self.sim.paused = False
                self.sim.mode = 'jev'
                self.sim.origin = 'jev'
            self._epoch += 1
            epoch = self._epoch
            self._running = True
            self._started = time.monotonic()
            self._finished = None
            self._deadline = self._started + max_seconds
            self._max_calls, self._max_seconds = max_calls, max_seconds
            self._calls = self._applied = 0
            self._latency = self._last_action = self._last_confidence = None
            self._last_decision = None
            self._reason = 'running'
            self._thread = threading.Thread(target=self._work, args=(epoch, self._adapter), daemon=True, name='jev-sim-runner')
            self._timer = threading.Timer(max_seconds, self._expire, args=(epoch,))
            self._timer.daemon = True
            self._thread.start()
            self._timer.start()
            return self.status()

    def _finish_locked(self, reason):
        self._running = False
        if hasattr(self._adapter,'cancel'):
            try:self._adapter.cancel()
            except (OSError,ValueError):pass  # epoch invalidation must always proceed
        self._epoch += 1
        self._reason = reason
        self._finished = time.monotonic()
        if self._timer:
            self._timer.cancel()
        # The same lock protects epoch checking and actuation. Once stop returns,
        # no result from the old epoch can enqueue another command.
        self.sim.stop()

    def _expire(self, epoch):
        with self._lock:
            if self._running and epoch == self._epoch:
                self._finish_locked('time_budget')

    def stop(self):
        with self._lock:
            self._finish_locked('stopped')
            return self.status()

    def pause(self):
        with self._lock:
            self._finish_locked('paused')
            return self.status()

    def _active(self, epoch):
        return self._running and epoch == self._epoch

    def _work(self, epoch, adapter):
        try:
            while True:
                with self._lock:
                    if not self._active(epoch):
                        return
                    if time.monotonic() >= self._deadline:
                        self._finish_locked('time_budget')
                        return
                    state = self.sim.state()
                    if state['paused']:
                        self._finish_locked('simulation_paused')
                        return
                    busy = state['busy']
                    if not busy:
                        if self._calls >= self._max_calls:
                            self._finish_locked('call_budget')
                            return
                        snapshot = observation(self.sim)
                        self._calls += 1
                if busy:
                    time.sleep(0.01)
                    continue
                before = time.monotonic()
                try:
                    decision = adapter.decide(snapshot)
                except Exception:
                    # Never reflect arbitrary exception text, URLs, state or keys.
                    decision = Decision(reason='adapter_error')
                latency = round(time.monotonic() - before, 4)
                with self._lock:
                    if not self._active(epoch):
                        return
                    self._latency = latency
                    if time.monotonic() >= self._deadline:
                        self._finish_locked('time_budget')
                        return
                    if not isinstance(decision, Decision):
                        self._finish_locked('invalid_decision')
                        return
                    self._last_decision=decision.to_dict()
                    self._last_action = decision.action if decision.action in ('x_plus','x_minus','y_plus','y_minus','z_plus','z_minus','open','close','hold') else 'hold'
                    confidence = decision.confidence
                    self._last_confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1 else None
                    if not decision.accepted:
                        safe_reasons = {'low_confidence', 'timeout', 'network_error', 'invalid_response', 'invalid_state', 'response_too_large', 'adapter_error'}
                        self._finish_locked(decision.reason if decision.reason in safe_reasons else 'api_error')
                        return
                    if decision.action == 'hold':
                        self._finish_locked('model_hold')
                        return
                    try:
                        with self.sim.lock:
                            apply_action(self.sim, decision.action, origin='jev')
                        self._applied += 1
                    except Exception:
                        self._finish_locked('action_rejected')
                        return
        except Exception:
            with self._lock:
                if self._active(epoch):
                    self._finish_locked('runner_error')
