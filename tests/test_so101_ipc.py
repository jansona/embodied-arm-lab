"""Offline-only SO101 transport tests. No saved config or real provider calls."""
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import httpx
from embodied_arm.providers.adapters import ALIYUN_ENDPOINT, ALIYUN_MODEL
from embodied_arm.providers.so101_jev import SO101Decision, SO101JevAdapter, build_observation, parse_response, validate_observation
from embodied_arm.runtime.so101_ipc import (FileSO101Bridge, SO101Store, REQUEST_SCHEMA, RESULT_SCHEMA,
                      observation_hash, validate_job, worker_status)
from embodied_arm.runtime.so101_worker import main, worker_run

MP = multiprocessing.get_context('fork')


def observation():
    robot = {'name': 'SO-ARM101', 'arm_dof': 5, 'ee': [.2, 0, .06], 'joints': [0] * 5,
             'joint_limits': [[-2, 2]] * 5, 'gripper_openness': .4, 'gripper_angle': .2,
             'paused': True, 'busy': False, 'epoch': 1, 'actions': 0,
             'workspace': [[.06, .38], [-.25, .25], [.006, .36]], 'cube_edge_m': .02}
    target = {'source': 'rgb_only', 'kind': 'cartesian', 'target_xyz': [.22, 0, .06],
              'tolerance_m': .005, 'step_m': .008}
    return build_observation(target, robot, 0)


def answer(choice='positive', confidence=.9):
    return {'model': ALIYUN_MODEL, 'answers': {'action': {'type': 'choice', 'choice': choice,
        'confidence': confidence, 'probabilities': {c: .9 if c == choice else .05
                                                  for c in ('positive', 'negative', 'hold')}}}}


class FakeAdapter:
    def __init__(self, mode='ok'):
        self.mode = mode
        self.calls = MP.Value('i', 0)
        self.pids = MP.Array('i', 12)
        self.closed = MP.Value('i', 0)
        self.entered = MP.Event()

    def decide(self, current):
        with self.calls.get_lock():
            index = self.calls.value
            self.calls.value += 1
            self.pids[index] = os.getpid()
        self.entered.set()
        if self.mode == 'slow': time.sleep(3)
        if self.mode == 'error': raise RuntimeError('SECRET Authorization Bearer https://bad.invalid')
        if self.mode == 'invalid': return {'action': 'x_plus', 'accepted': True}
        if self.mode == 'low': return parse_response(answer(confidence=.3), current)
        if self.mode == 'wrong': return parse_response(answer('negative'), current)
        return parse_response(answer(), current)

    def close(self):
        with self.closed.get_lock(): self.closed.value += 1


class SO101IPCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / '.runtime' / 'so101-jev-ipc'
        self.patcher = patch('embodied_arm.runtime.so101_ipc.SO101_IPC_ROOT', self.root)
        self.patcher.start()
        self.threads = []

    def tearDown(self):
        for thread in self.threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.patcher.stop()
        self.temp.cleanup()

    def start_worker(self, adapter=None, calls=1, seconds=3.):
        adapter = adapter or FakeAdapter()
        done = []
        thread = threading.Thread(target=lambda: done.append(worker_run(adapter, calls, seconds)))
        self.threads.append(thread)
        thread.start()
        until = time.monotonic() + 3
        while not worker_status()['available'] and time.monotonic() < until:
            time.sleep(.002)
        self.assertTrue(worker_status()['available'])
        return adapter, thread, done

    def read(self, name):
        store = SO101Store()
        try: return store.read(name)
        finally: store.close()

    def job(self, job='a' * 32, **changes):
        obs = observation()
        data = {'schema': REQUEST_SCHEMA, 'simulation_only': True, 'id': job,
                'expires_at': time.time() + .5, 'observation_sha256': observation_hash(obs),
                'observation': obs}
        data.update(changes)
        store = SO101Store()
        try: store.write(job + '.job', data)
        finally: store.close()
        return job

    def test_atomic_ready_replacement_reopens_unlinked_snapshot(self):
        store = SO101Store()
        store.write('ready.json', {'version': 1})
        original = os.fstat
        unlinked_counts = []
        def replace_after_open(fd):
            if not unlinked_counts:
                store.write('ready.json', {'version': 2})
                info = original(fd)
                unlinked_counts.append(info.st_nlink)
                return info
            return original(fd)
        try:
            with patch('embodied_arm.runtime.so101_ipc.os.fstat', side_effect=replace_after_open):
                self.assertEqual(store.read('ready.json'), {'version': 2})
            self.assertEqual(unlinked_counts, [0])
        finally:
            store.close()

    def test_atomic_ready_retry_is_bounded_and_never_accepts_unlinked_inode(self):
        store = SO101Store()
        store.write('ready.json', {'version': 1})
        original = os.fstat
        attempts = []
        def replace_every_open(fd):
            store.write('ready.json', {'version': len(attempts) + 2})
            info = original(fd)
            attempts.append(info.st_nlink)
            return info
        try:
            with patch('embodied_arm.runtime.so101_ipc.os.fstat', side_effect=replace_every_open):
                with self.assertRaisesRegex(ValueError, 'Unsafe SO101 IPC file'):
                    store.read('ready.json')
            self.assertEqual(attempts, [0, 0])
        finally:
            store.close()

    def test_two_roundtrips_use_one_pooled_child(self):
        adapter, worker, done = self.start_worker(calls=2)
        bridge = FileSO101Bridge()
        first = bridge.decide(observation())
        second = bridge.decide(observation())
        worker.join(5)
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(first.action, 'x_plus')
        self.assertEqual(adapter.calls.value, 2)
        self.assertEqual(adapter.pids[0], adapter.pids[1])
        self.assertNotEqual(adapter.pids[0], os.getpid())
        self.assertEqual(adapter.closed.value, 1)
        self.assertEqual(done[0]['calls'], 2)
        self.assertEqual(done[0]['status'], 'call_budget')
        self.assertFalse(worker_status()['available'])
        self.assertFalse(multiprocessing.active_children())
        audit = self.read('last-run-audit.json')
        self.assertEqual(len(audit), 2)
        self.assertNotIn('joints', json.dumps(audit))
        self.assertNotIn('target_xyz', json.dumps(audit))
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)

    def test_real_adapter_with_fake_http_provider(self):
        calls = MP.Value('i', 0)
        def handler(request):
            self.assertEqual(str(request.url), ALIYUN_ENDPOINT)
            body = json.loads(request.content)
            self.assertEqual(body['model'], ALIYUN_MODEL)
            self.assertNotIn('Panda', json.dumps(body))
            with calls.get_lock(): calls.value += 1
            return httpx.Response(200, json=answer())
        with SO101JevAdapter(api_key='FAKE-TEST-KEY', max_calls=2, transport=httpx.MockTransport(handler)) as adapter:
            _, worker, _ = self.start_worker(adapter, calls=2)
            bridge = FileSO101Bridge()
            self.assertTrue(bridge.decide(observation()).accepted)
            self.assertTrue(bridge.decide(observation()).accepted)
            worker.join(5)
        self.assertEqual(calls.value, 2)

    def test_rejection_never_becomes_rule_selected_action(self):
        for mode, reason in (('low', 'low_confidence'), ('wrong', 'criterion_mismatch'),
                             ('invalid', 'invalid_response'), ('error', 'adapter_error')):
            with self.subTest(mode=mode):
                _, worker, _ = self.start_worker(FakeAdapter(mode))
                result = FileSO101Bridge().decide(observation())
                worker.join(5)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.action)
                self.assertEqual(result.reason, reason)
                self.assertNotIn('SECRET', (self.root / 'last-run-audit.json').read_text())

    def test_offline_bridge_never_loads_configuration(self):
        with patch.object(SO101JevAdapter, 'from_config', side_effect=AssertionError('No loading')):
            bridge = FileSO101Bridge()
            self.assertFalse(bridge.status()['available'])
            self.assertEqual(bridge.decide(observation()).reason, 'unavailable')

    def test_busy_or_unpaused_robot_cannot_be_queued(self):
        for field, value in (('busy', True), ('paused', False)):
            obs = observation()
            obs['robot'][field] = value
            self.assertEqual(FileSO101Bridge().decide(obs).reason, 'busy')
        self.assertFalse(list(self.root.glob('*.job')))

    def test_stop_kills_pooled_child_and_ends_window_without_restart(self):
        adapter, worker, done = self.start_worker(FakeAdapter('slow'), calls=3)
        bridge = FileSO101Bridge()
        results = []
        client = threading.Thread(target=lambda: results.append(bridge.decide(observation())))
        client.start()
        self.assertTrue(adapter.entered.wait(1))
        self.assertEqual(bridge.decide(observation()).reason, 'busy')
        before = time.monotonic()
        bridge.cancel()
        client.join(1)
        worker.join(5)
        self.assertLess(time.monotonic() - before, .4)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertIsNone(results[0].action)
        self.assertEqual(done[0]['status'], 'cancelled')
        self.assertEqual(adapter.calls.value, 1)
        self.assertFalse(worker_status()['available'])
        self.assertFalse(multiprocessing.active_children())
        self.assertFalse(list(self.root.glob('*.result')))

    def test_cancel_before_queue_prevents_job(self):
        entered, release = threading.Event(), threading.Event()
        bridge = FileSO101Bridge()
        results = []
        def slow(obs):
            entered.set(); release.wait(1)
            return validate_observation(obs)
        with patch('embodied_arm.runtime.so101_ipc.validate_observation', side_effect=slow):
            client = threading.Thread(target=lambda: results.append(bridge.decide(observation())))
            client.start()
            self.assertTrue(entered.wait(1))
            bridge.cancel(); release.set(); client.join(1)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertFalse(list(self.root.glob('*.job')))

    def test_deadline_kills_pending_call(self):
        self.job()
        adapter = FakeAdapter('slow')
        before = time.monotonic()
        done = worker_run(adapter, max_calls=3, max_seconds=.07)
        self.assertLess(time.monotonic() - before, .4)
        self.assertEqual(done['status'], 'time_budget')
        result = self.read('a' * 32 + '.result')['result']
        self.assertEqual(result['reason'], 'timeout')
        self.assertIsNone(result['action'])
        self.assertEqual(adapter.calls.value, 1)
        self.assertFalse(multiprocessing.active_children())

    def test_expiry_ends_window_and_cannot_deliver_late_motion(self):
        self.job(expires_at=time.time() + .06)
        done = worker_run(FakeAdapter('slow'), max_calls=3, max_seconds=.3)
        self.assertEqual(done['status'], 'expired')
        self.assertFalse(list(self.root.glob('*.result')))
        self.assertFalse(multiprocessing.active_children())

    def test_no_panda_oracle_urls_headers_or_keys_in_schema(self):
        obs = observation()
        mutations = [lambda o: o.update(headers={}), lambda o: o.update(api_key='SECRET'),
            lambda o: o.update(endpoint='https://x.invalid'), lambda o: o.update(schema='arm-observation-v1'),
            lambda o: o['robot'].update(name='Franka Panda', arm_dof=7),
            lambda o: o['robot'].update(body_id=7), lambda o: o['target'].update(source='oracle'),
            lambda o: o['target'].update(chosen_action='x_plus')]
        for mutate in mutations:
            bad = json.loads(json.dumps(obs)); mutate(bad)
            self.assertEqual(FileSO101Bridge().decide(bad).reason, 'invalid_input')
        job = self.job()
        data = self.read(job + '.job')
        self.assertEqual(validate_job(data, job), data)
        for changes in ({'observation_sha256': '0' * 64}, {'headers': {}}, {'simulation_only': False}):
            with self.assertRaises(ValueError): validate_job({**data, **changes}, job)

    def test_malformed_expired_and_cancelled_jobs_use_no_api(self):
        self.job(expires_at=time.time() - 1)
        self.job('b' * 32, endpoint='https://x.invalid')
        self.job('c' * 32)
        store = SO101Store()
        try: store.write('c' * 32 + '.cancel', {'cancelled': True})
        finally: store.close()
        adapter = FakeAdapter()
        done = worker_run(adapter, max_seconds=.05)
        self.assertEqual(adapter.calls.value, 0)
        self.assertEqual(done['rejected'], 2)
        self.assertEqual(done['cancelled'], 1)

    def test_response_digest_rejects_stale_observation(self):
        store = SO101Store(); store.close()
        def forge():
            until = time.monotonic() + 3
            while not list(self.root.glob('*.job')) and time.monotonic() < until: time.sleep(.002)
            jobs = list(self.root.glob('*.job'))
            if not jobs: return
            job = jobs[0].stem; data = self.read(job + '.job')
            store = SO101Store()
            try:
                store.write(job + '.result', {'schema': RESULT_SCHEMA, 'id': job,
                    'expires_at': data['expires_at'], 'observation_sha256': '0' * 64,
                    'result': parse_response(answer(), data['observation']).to_dict()})
            finally: store.close()
        thread = threading.Thread(target=forge); thread.start()
        with patch('embodied_arm.runtime.so101_ipc.worker_status', return_value={'available': True, 'remaining_calls': 1}):
            result = FileSO101Bridge().decide(observation())
        thread.join(1)
        self.assertEqual(result.reason, 'invalid_response')
        self.assertIsNone(result.action)

    def test_store_and_worker_locks_stay_private_and_bounded(self):
        store = SO101Store()
        name = 'a' * 32 + '.job'
        outside = Path(self.temp.name) / 'outside'; outside.write_text('{}'); outside.chmod(0o600)
        try:
            with self.assertRaises(ValueError): store.read('../outside')
            (self.root / name).symlink_to(outside)
            with self.assertRaises(OSError): store.read(name)
            with self.assertRaises(ValueError): store.write(name, {})
            (self.root / name).unlink()
            os.link(outside, self.root / name)
            with self.assertRaises(ValueError): store.read(name)
            (self.root / name).unlink()
            os.mkfifo(self.root / name, mode=0o600)
            with self.assertRaises(ValueError): store.read(name)
            (self.root / name).unlink()
            with self.assertRaises(ValueError): store.write(name, {'x': 'A' * 32768})
        finally: store.close()
        adapter, worker, _ = self.start_worker(seconds=.1)
        with self.assertRaises(ValueError): worker_run(adapter, max_seconds=.01)
        worker.join(5)
        self.assertFalse(worker_status()['available'])
        for args in ({'max_calls': 0}, {'max_calls': 201}, {'max_calls': True},
                     {'max_seconds': 421}, {'max_seconds': 0}, {'max_seconds': float('nan')}):
            with self.assertRaises(ValueError): worker_run(FakeAdapter(), **args)
        with patch.object(SO101JevAdapter, 'from_config', side_effect=AssertionError('No loading')):
            with self.assertRaises(SystemExit): main(['--config', '/fake/config', '--max-calls', '201'])

    def test_transient_status_probe_does_not_claim_existing_worker(self):
        store = SO101Store(); store.write('worker.lock', {})
        fd = os.open('worker.lock', os.O_RDWR | os.O_NOFOLLOW, dir_fd=store.fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        timer = threading.Timer(.02, lambda: fcntl.flock(fd, fcntl.LOCK_UN)); timer.start()
        try:
            done = worker_run(FakeAdapter(), max_seconds=.08)
            self.assertEqual(done['status'], 'time_budget')
            self.assertEqual(done['calls'], 0)
        finally:
            timer.join(1); os.close(fd); store.close()


if __name__ == '__main__': unittest.main()
