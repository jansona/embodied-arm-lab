"""Runner tests use simulated local fakes and mock adapters, never the network."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import httpx
from unittest.mock import patch

from adapters import ACTIONS, Decision, JevAdapter, OFFICIAL_ENDPOINT, ALIYUN_ENDPOINT, ALIYUN_MODEL
from runner import JevRunner


class FakeSim:
    def __init__(self):
        self.lock = threading.RLock()
        self.paused = False
        self.queue = []
        self.mode = 'manual'
        self.origin = 'reset'
        self.moves = []
        self.position = [.4, 0, .3]
        self.busy = False
    def state(self):
        with self.lock:
            return {'busy': self.busy, 'paused': self.paused, 'ee': self.position[:],
                    'joints': [0]*7, 'gripper_width': .08, 'blocks': {'red': [.5, 0, .025]},
                    'goal': [.48, .21, .025], 'workspace': [], 'metrics': {}}
    def manual(self, action, values=None):
        with self.lock:
            if self.paused or self.busy:
                raise ValueError('Not idle')
            self.moves.append((action, values))
            if values:
                self.position = values[:]
            self.queue.append({'origin': 'manual'})
    def stop(self):
        with self.lock:
            self.paused = True
            self.queue.clear()
            self.busy = False


class FixedAdapter:
    def __init__(self, result): self.result = result; self.calls = 0
    def decide(self, state): self.calls += 1; return self.result


def wait_done(runner, timeout=2):
    deadline = time.monotonic() + timeout
    while runner.status()['pending'] and time.monotonic() < deadline:
        time.sleep(.005)
    assert not runner.status()['pending'], runner.status()


class RunnerTests(unittest.TestCase):
    def test_response_to_motion_and_call_budget(self):
        sim = FakeSim()
        adapter = FixedAdapter(Decision('x_plus', .9, {}, True, 'accepted'))
        runner = JevRunner(sim, adapter)
        self.assertEqual(adapter.calls, 0)
        runner.start(max_calls=2, max_seconds=1)
        wait_done(runner)
        self.assertEqual(adapter.calls, 2)
        self.assertEqual(len(sim.moves), 2)
        self.assertAlmostEqual(sim.position[0], .44)
        self.assertEqual(runner.status()['reason'], 'call_budget')
        self.assertEqual(runner.status()['origin'], 'jev')
        self.assertEqual(runner.status()['actions_applied'], 2)

    def test_mock_http_response_through_adapter_and_motion_boundary(self):
        seen = []
        def respond(request):
            seen.append(json.loads(request.content))
            probabilities = {action: 0.0 for action in ACTIONS}
            probabilities['z_plus'] = 1.0
            return httpx.Response(200, json={'answers': {'action': {
                'type': 'choice', 'choice': 'z_plus', 'confidence': .99,
                'probabilities': probabilities}}})
        adapter = JevAdapter(api_key='test-only-placeholder', transport=httpx.MockTransport(respond))
        sim = FakeSim(); runner = JevRunner(sim, adapter)
        runner.start(max_calls=1); wait_done(runner)
        self.assertAlmostEqual(sim.position[2], .32)
        self.assertTrue(seen[0]['state']['simulation_only'])
        self.assertEqual(runner.status()['status'], 'call_budget')
        self.assertEqual(runner.status()['actions_applied'], 1)

    def test_error_or_low_confidence_never_falls_back(self):
        for result in [Decision(reason='network_error'), Decision(confidence=.1, reason='low_confidence')]:
            sim = FakeSim(); runner = JevRunner(sim, FixedAdapter(result))
            runner.start(); wait_done(runner)
            self.assertEqual(sim.moves, [])
            self.assertEqual(runner.status()['reason'], result.reason)
            self.assertTrue(sim.paused)

    def test_stop_pending_discards_stale_response(self):
        entered, release = threading.Event(), threading.Event()
        class Pending:
            def decide(self, state):
                entered.set(); release.wait(2)
                return Decision('x_plus', .9, {}, True, 'accepted')
        sim = FakeSim(); runner = JevRunner(sim, Pending())
        runner.start(); self.assertTrue(entered.wait(1))
        runner.stop()
        with self.assertRaises(ValueError): runner.start()
        release.set(); wait_done(runner)
        self.assertEqual(sim.moves, [])
        self.assertEqual(runner.status()['reason'], 'stopped')

    def test_wall_budget_expires_even_while_request_pending(self):
        entered, release = threading.Event(), threading.Event()
        class Pending:
            def decide(self, state):
                entered.set(); release.wait(2)
                return Decision('x_plus', .9, {}, True, 'accepted')
        sim = FakeSim(); runner = JevRunner(sim, Pending())
        runner.start(max_seconds=.05); self.assertTrue(entered.wait(1))
        time.sleep(.1)
        self.assertFalse(runner.status()['running'])
        self.assertTrue(runner.status()['pending'])
        self.assertEqual(runner.status()['reason'], 'time_budget')
        self.assertTrue(sim.paused)
        release.set(); wait_done(runner)
        self.assertEqual(sim.moves, [])

    def test_waits_for_idle_before_next_call(self):
        sim = FakeSim()
        original = sim.manual
        def manual(action, values=None): original(action, values); sim.busy = True
        sim.manual = manual
        adapter = FixedAdapter(Decision('x_plus', .9, {}, True, 'accepted'))
        runner = JevRunner(sim, adapter); runner.start(max_calls=2, max_seconds=.08)
        wait_done(runner)
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(runner.status()['reason'], 'time_budget')

    def test_exception_is_sanitized_and_hold_stops(self):
        class Broken:
            def decide(self, state): raise RuntimeError('SECRET-KEY')
        for adapter, reason in [(Broken(), 'adapter_error'), (FixedAdapter(Decision('hold', .9, {}, True, 'accepted')), 'model_hold')]:
            runner = JevRunner(FakeSim(), adapter); runner.start(); wait_done(runner)
            self.assertEqual(runner.status()['reason'], reason)
            self.assertNotIn('SECRET', json.dumps(runner.status()))

    def test_configure_explicit_file_only_no_network_or_key_status(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'explicit.json'
            path.write_text(json.dumps({'provider':'aliyun_bailian','endpoint': ALIYUN_ENDPOINT,'model':ALIYUN_MODEL}))
            runner = JevRunner(FakeSim())
            self.assertFalse(runner.configure(path)['configured'])
            with self.assertRaises(ValueError): runner.start()
            path.write_text(json.dumps({'provider':'aliyun_bailian','endpoint':ALIYUN_ENDPOINT,'model':ALIYUN_MODEL,'api_key': 'test-only-placeholder'}))
            with patch('httpx.Client', side_effect=AssertionError('No network at configure')):
                self.assertTrue(runner.configure(path)['configured'])
            self.assertNotIn('test-only-placeholder', json.dumps(runner.status()))
            path.write_text(json.dumps({'api_key': 'test-only-placeholder', 'endpoint': 'https://evil.test'}))
            with self.assertRaises(ValueError): runner.configure(path)
        with self.assertRaises(ValueError): runner.configure(Path(__file__).parent / 'creds.json')

    def test_budgets_and_busy_start_rejected(self):
        runner = JevRunner(FakeSim(), FixedAdapter(Decision()))
        for kwargs in [{'max_calls': 0}, {'max_calls': True}, {'max_calls': 1001}, {'max_seconds': 0}, {'max_seconds': float('nan')}, {'max_seconds': 301}]:
            with self.assertRaises(ValueError): runner.start(**kwargs)
        runner.sim.busy = True
        with self.assertRaises(ValueError): runner.start()


if __name__ == '__main__': unittest.main()
