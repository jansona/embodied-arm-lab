"""Offline-only protocol/security tests. No saved config or live API calls."""
import json
import fcntl
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from embodied_arm.providers.brain import (BrainPlan, BrainResult, MODEL, PROVIDER_ERROR_REASONS,
                   REPLAN_ERROR_REASONS, validate_plan)
import embodied_arm.runtime.brain_ipc as brain_ipc
from embodied_arm.runtime.brain_ipc import (BrainStore, FileBrainBridge, REQUEST_SCHEMA, RESULT_SCHEMA,
                       REPLAN_REQUEST_SCHEMA, normalize_input, validate_job,
                       validate_result, worker_status)
from embodied_arm.runtime.brain_worker import main, worker_run
from embodied_arm.runtime.ipc import Store


MP = multiprocessing.get_context('fork')


def scene():
    return {'blocks': {'red': [.5, 0, .025], 'blue': [.58, -.19, .025]},
            'ee': [.4, 0, .3], 'goal': [.48, .21, .025], 'busy': False,
            'paused': False, 'gripper_width': .08, 'status': 'SENSITIVE STATUS',
            'headers': {'Authorization': 'SECRET'}, 'api_key': 'SECRET'}


class MockBrain:
    def __init__(self, mode='ok'):
        self.calls = MP.Value('i', 0)
        self.entered = MP.Event()
        self.mode = mode

    def interpret(self, text, current_scene, history=None):
        with self.calls.get_lock():
            self.calls.value += 1
        self.entered.set()
        if self.mode == 'slow':
            time.sleep(5)
        if self.mode == 'exception':
            raise RuntimeError('SECRET https://unapproved.invalid Authorization')
        if self.mode == 'invalid':
            return BrainResult(plan=BrainPlan('pick', 'other', None, 'SECRET'),
                               accepted=True, reason='accepted', provenance='mock')
        if self.mode == 'failure':
            return BrainResult(reason='network_error', calls=1, provenance='mock')
        return BrainResult(plan=BrainPlan('pick', 'red', 'goal', 'MODEL MESSAGE SECRET'),
                           accepted=True, reason='accepted', calls=1, provenance='mock')


def sequence_plan():
    return {'action': 'sequence', 'object_id': None, 'target_id': None, 'message': '把红块叠到蓝块上',
            'steps': [{'step_id': 's1', 'action': 'pick', 'object_id': 'red', 'target_id': None},
                      {'step_id': 's2', 'action': 'place', 'object_id': 'red', 'target_id': 'on:blue'}]}


def repair_feedback():
    return {'completed_step_ids': [], 'failed_step_id': 's1', 'reason': 'grasp_failed'}


class RepairBrain(MockBrain):
    def __init__(self, mode='ok'):
        super().__init__(mode)
        self.repair_calls = MP.Value('i', 0)

    def interpret(self, text, current_scene, history=None):
        result = super().interpret(text, current_scene, history)
        if not result.accepted:
            return result
        return BrainResult(plan=validate_plan(sequence_plan(), current_scene), accepted=True,
                           reason='accepted', calls=1, provenance='mock')

    def replan(self, text, current_scene, previous_plan, feedback, history=None):
        if previous_plan != sequence_plan() or feedback != repair_feedback():
            return BrainResult(reason='invalid_feedback', provenance='mock')
        with self.repair_calls.get_lock():
            self.repair_calls.value += 1
        return self.interpret(text, current_scene, history)


class BrainIPCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / '.runtime' / 'brain-ipc'
        self.patcher = patch('embodied_arm.runtime.brain_ipc.BRAIN_IPC_ROOT', self.root)
        self.patcher.start()
        self.threads = []
        self.brains = []

    def tearDown(self):
        for thread in self.threads:
            thread.join(2)
            self.assertFalse(thread.is_alive(), 'Bounded worker did not exit')
        self.patcher.stop()
        self.temp.cleanup()

    def start_worker(self, brain=None, calls=1, seconds=.8):
        brain = brain or MockBrain()
        self.brains.append(brain)
        results = []
        thread = threading.Thread(target=lambda: results.append(worker_run(brain, calls, seconds)))
        self.threads.append(thread)
        thread.start()
        limit = time.monotonic() + 1
        while not worker_status()['available'] and time.monotonic() < limit:
            time.sleep(.002)
        self.assertTrue(worker_status()['available'])
        return brain, thread, results

    def write_job(self, job='a' * 32, **changes):
        data = {'schema': REQUEST_SCHEMA, 'simulation_only': True, 'id': job,
                'expires_at': time.time() + .5,
                **normalize_input('拿起红色方块放到绿色目标区域', scene())}
        data.update(changes)
        store = BrainStore()
        try:
            store.write(job + '.job', data)
        finally:
            store.close()
        return job

    def read(self, name):
        store = BrainStore()
        try:
            return store.read(name)
        finally:
            store.close()

    def test_mock_roundtrip_and_sanitized_audit(self):
        brain, thread, results = self.start_worker()
        result = FileBrainBridge().interpret('拿起红色方块', scene(), [{'role': 'user', 'content': 'TEXT SECRET'}])
        thread.join(1)
        self.assertTrue(result.accepted)
        self.assertEqual(result.plan.object_id, 'red')
        self.assertEqual(result.provenance, 'mock')
        self.assertEqual(brain.calls.value, 1)
        self.assertEqual(results[0]['calls'], 1)
        self.assertEqual(results[0]['status'], 'call_budget')
        self.assertFalse(worker_status()['available'])
        content = (self.root / 'last-run-audit.json').read_text()
        self.assertNotIn('SECRET', content)
        self.assertNotIn('Authorization', content)
        self.assertNotIn('TEXT', content)
        self.assertEqual(json.loads(content)[0]['action'], 'pick')
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        for path in self.root.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_offline_is_fast_and_loads_no_config(self):
        before = time.monotonic()
        with patch('embodied_arm.providers.brain.QwenBrain.from_config', side_effect=AssertionError('Must not load')):
            bridge = FileBrainBridge()
            self.assertFalse(bridge.status()['configured'])
            self.assertEqual(bridge.interpret('拿红块', scene()).reason, 'unavailable')
        self.assertLess(time.monotonic() - before, .2)

    def test_error_roundtrip_exposes_no_exception_text(self):
        brain, thread, _ = self.start_worker(MockBrain('exception'))
        result = FileBrainBridge().interpret('拿红块', scene())
        thread.join(1)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.plan)
        self.assertEqual(result.reason, 'adapter_error')
        self.assertNotIn('SECRET', (self.root / 'last-run-audit.json').read_text())

    def test_rejected_provider_result_has_no_plan(self):
        _, thread, _ = self.start_worker(MockBrain('failure'))
        result = FileBrainBridge().interpret('拿红块', scene())
        thread.join(1)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.plan)
        self.assertEqual(result.reason, 'network_error')

    def test_invalid_adapter_plan_never_accepted(self):
        _, thread, _ = self.start_worker(MockBrain('invalid'))
        result = FileBrainBridge().interpret('拿红块', scene())
        thread.join(1)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.plan)

    def test_cancel_terminates_pending_helper_and_discards_result(self):
        brain, worker, done = self.start_worker(MockBrain('slow'))
        bridge = FileBrainBridge()
        results = []
        client = threading.Thread(target=lambda: results.append(bridge.interpret('拿红块', scene())))
        client.start()
        self.assertTrue(brain.entered.wait(1))
        before = time.monotonic()
        bridge.cancel()
        client.join(1)
        worker.join(1)
        self.assertFalse(client.is_alive())
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - before, .4)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertEqual(done[0]['cancelled'], 1)
        self.assertFalse(list(self.root.glob('*.result')))
        self.assertFalse(multiprocessing.active_children())

    def test_cancel_marker_before_worker_makes_zero_calls(self):
        job = self.write_job()
        store = BrainStore()
        try:
            store.write(job + '.cancel', {'cancelled': True})
        finally:
            store.close()
        brain = MockBrain()
        result = worker_run(brain, max_calls=1, max_seconds=.05)
        self.assertEqual(brain.calls.value, 0)
        self.assertEqual(result['cancelled'], 1)

    def test_cancel_during_input_preparation_prevents_job(self):
        bridge = FileBrainBridge()
        entered, release = threading.Event(), threading.Event()
        results = []
        original = normalize_input

        def blocked(*args):
            entered.set()
            release.wait(1)
            return original(*args)

        with patch('embodied_arm.runtime.brain_ipc.normalize_input', side_effect=blocked):
            client = threading.Thread(target=lambda: results.append(bridge.interpret('拿红块', scene())))
            client.start()
            self.assertTrue(entered.wait(1))
            bridge.cancel()
            release.set()
            client.join(1)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertFalse(list(self.root.glob('*.job')))

    def test_cancel_is_local_even_if_cancel_write_fails(self):
        brain, worker, _ = self.start_worker(MockBrain('slow'))
        bridge = FileBrainBridge()
        results = []
        client = threading.Thread(target=lambda: results.append(bridge.interpret('拿红块', scene())))
        client.start()
        self.assertTrue(brain.entered.wait(1))
        with patch('embodied_arm.runtime.brain_ipc.BrainStore.write', side_effect=OSError('unavailable')):
            bridge.cancel()
        client.join(1)
        worker.join(1)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertFalse(list(self.root.glob('*.result')))

    def test_wrong_response_envelope_is_rejected(self):
        store = BrainStore()
        store.close()
        done = []

        def forge():
            limit = time.monotonic() + 1
            while not list(self.root.glob('*.job')) and time.monotonic() < limit:
                time.sleep(.002)
            jobs = list(self.root.glob('*.job'))
            if not jobs:
                return
            job = jobs[0].stem
            data = self.read(job + '.job')
            fake = {'schema': RESULT_SCHEMA, 'id': 'b' * 32, 'expires_at': data['expires_at'],
                    'result': BrainResult(plan=BrainPlan('pick', 'red', None, '拿红块'),
                        accepted=True, reason='accepted', provenance='mock').to_dict()}
            store = BrainStore()
            try:
                store.write(job + '.result', fake)
            finally:
                store.close()

        thread = threading.Thread(target=forge)
        thread.start()
        with patch('embodied_arm.runtime.brain_ipc.worker_status', return_value={'available': True, 'remaining_calls': 1}):
            done.append(FileBrainBridge().interpret('拿红块', scene()))
        thread.join(1)
        self.assertEqual(done[0].reason, 'invalid_response')
        self.assertIsNone(done[0].plan)

    def test_second_bridge_call_is_busy(self):
        brain, worker, _ = self.start_worker(MockBrain('slow'))
        bridge = FileBrainBridge()
        results = []
        client = threading.Thread(target=lambda: results.append(bridge.interpret('拿红块', scene())))
        client.start()
        self.assertTrue(brain.entered.wait(1))
        self.assertEqual(bridge.interpret('拿蓝块', scene()).reason, 'busy')
        bridge.cancel()
        client.join(1)
        worker.join(1)

    def test_strict_filtered_job_schema(self):
        safe = normalize_input('拿红块', scene(), [{'role': 'user', 'content': '好', 'secret': 'SECRET'}])
        self.assertNotIn('SECRET', json.dumps(safe))
        self.assertEqual(set(safe['scene']), {'blocks', 'ee', 'goal', 'busy', 'paused', 'gripper_width'})
        data = {'schema': REQUEST_SCHEMA, 'simulation_only': True, 'id': 'a' * 32,
                'expires_at': time.time() + 1, **safe}
        self.assertEqual(validate_job(data, 'a' * 32), data)
        for changes in ({'endpoint': 'https://x.invalid'}, {'api_key': 'SECRET'},
                        {'headers': {}}, {'config': '/somewhere'}, {'simulation_only': False},
                        {'id': 'b' * 32}, {'expires_at': float('nan')}):
            with self.assertRaises(ValueError):
                validate_job({**data, **changes}, 'a' * 32)
        data['scene']['api_key'] = 'SECRET'
        with self.assertRaises(ValueError):
            validate_job(data, 'a' * 32)

    def test_limits_urls_headers_credentials_and_nonfinite_rejected(self):
        for text in ('', ' ' * 4, 'x' * 2001, 'visit https://example.com', 'Authorization: Bearer abc',
                     'api_key=SECRET', 'file:///tmp/x', 'okay\x00bad'):
            self.assertEqual(FileBrainBridge().interpret(text, scene()).reason, 'invalid_input')
        for bad_scene in ({}, {'blocks': {}, 'ee': [0, 0, 0], 'goal': [0, 0, 0]},
                          {**scene(), 'ee': [float('inf'), 0, 0]}):
            self.assertEqual(FileBrainBridge().interpret('拿红块', bad_scene).reason, 'invalid_input')
        with self.assertRaises(ValueError):
            normalize_input('拿红块', scene(), [{'role': 'system', 'content': 'bad'}])

    def test_multibyte_input_and_history_fit_fixed_byte_limit(self):
        text = '🤖' * 2000
        safe = normalize_input(text, scene(), [{'role': 'user', 'content': '🤖' * 1000}] * 6)
        job = self.write_job(**safe)
        data = self.read(job + '.job')
        self.assertEqual(data['text'], text)
        self.assertLessEqual((self.root / (job + '.job')).stat().st_size, 32767)
        validate_job(data, job)

    def test_expired_and_unfiltered_jobs_make_no_calls(self):
        brain = MockBrain()
        self.write_job(expires_at=time.time() - 1)
        self.write_job('b' * 32, endpoint='https://example.com')
        self.write_job('c' * 32, scene=scene())
        self.write_job('d' * 32, expires_at=time.time() + 1000)
        result = worker_run(brain, max_calls=1, max_seconds=.05)
        self.assertEqual(brain.calls.value, 0)
        self.assertEqual(result['rejected'], 4)

    def test_result_plan_and_metadata_validation(self):
        good = BrainResult(plan=BrainPlan('pick', 'red', None, '拿起红块'), accepted=True,
                           reason='accepted', provenance='mock').to_dict()
        self.assertTrue(validate_result(good, scene()).accepted)
        for changes in ({'headers': {}}, {'model': 'other'}, {'provenance': 'unknown'},
                        {'calls': True}, {'calls': 11}, {'reason': 'SECRET'},
                        {'accepted': False}, {'plan': {**good['plan'], 'joint_angles': [1]}}):
            with self.assertRaises(ValueError):
                validate_result({**good, **changes}, scene())
        missing = {**scene(), 'blocks': {'blue': [0, 0, 0]}}
        with self.assertRaises(ValueError):
            validate_result(good, missing)

    def test_deadline_kills_and_reaps_stuck_inference(self):
        brain = MockBrain('slow')
        self.write_job()
        before = time.monotonic()
        result = worker_run(brain, max_calls=1, max_seconds=.07)
        self.assertLess(time.monotonic() - before, .4)
        self.assertEqual(result['calls'], 1)
        self.assertFalse(multiprocessing.active_children())
        self.assertEqual(self.read('a' * 32 + '.result')['result']['reason'], 'timeout')

    def test_expiry_in_flight_never_delivers_late_plan(self):
        brain = MockBrain('slow')
        self.write_job(expires_at=time.time() + .06)
        result = worker_run(brain, max_calls=1, max_seconds=.3)
        self.assertEqual(result['calls'], 1)
        self.assertFalse(list(self.root.glob('*.result')))
        self.assertEqual(self.read('last-run-audit.json')[0]['reason'], 'expired')
        self.assertFalse(multiprocessing.active_children())

    def test_worker_lock_and_status_are_independent_of_cerebellum(self):
        other = Path(self.temp.name) / '.runtime' / 'jev-ipc'
        store = Store(other)
        store.write('ready.json', {'status': 'ready', 'calls': 0, 'max_calls': 8,
                                  'deadline_unix': time.time() + 30})
        store.close()
        self.assertFalse(worker_status()['available'])
        brain, thread, done = self.start_worker(seconds=.12)
        status = worker_status()
        self.assertTrue(status['configured'])
        self.assertEqual(status['model'], MODEL)
        with self.assertRaises(ValueError):
            worker_run(brain, max_seconds=.01)
        thread.join(1)
        self.assertFalse(worker_status()['available'])
        self.assertEqual(done[0]['status'], 'time_budget')

    def test_store_rejects_symlinks_and_traversal(self):
        store = BrainStore()
        target = Path(self.temp.name) / 'target'
        target.write_text('{}')
        try:
            with self.assertRaises(ValueError):
                store.read('../target')
            (self.root / ('a' * 32 + '.job')).symlink_to(target)
            with self.assertRaises(OSError):
                store.read('a' * 32 + '.job')
            with self.assertRaises(ValueError):
                store.write('a' * 32 + '.job', {})
        finally:
            store.close()
        brain = MockBrain()
        result = worker_run(brain, max_seconds=.03)
        self.assertEqual(brain.calls.value, 0)
        self.assertEqual(result['rejected'], 1)
        link = Path(self.temp.name) / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        with patch('embodied_arm.runtime.brain_ipc.BRAIN_IPC_ROOT', link), self.assertRaises(ValueError):
            BrainStore()

    def test_invalid_worker_budgets_and_cli_never_load_config(self):
        for args in ({'max_calls': 0}, {'max_calls': 11}, {'max_calls': True},
                     {'max_seconds': 601}, {'max_seconds': 0}, {'max_seconds': float('inf')}):
            with self.assertRaises(ValueError):
                worker_run(MockBrain(), **args)
        with patch('embodied_arm.providers.brain.QwenBrain.from_config', side_effect=AssertionError('Must not load')):
            with self.assertRaises(SystemExit):
                main(['--config', '/never/read/config', '--max-calls', '11'])

    def test_stale_ready_file_cannot_claim_available(self):
        store = BrainStore()
        try:
            store.write('ready.json', {'status': 'ready', 'calls': 0, 'max_calls': 10,
                                      'deadline_unix': time.time() + 120})
        finally:
            store.close()
        self.assertFalse(worker_status()['available'])

    def test_status_probe_lock_does_not_reject_worker_startup(self):
        store = BrainStore()
        store.write('worker.lock', {})
        fd = os.open('worker.lock', os.O_RDWR | os.O_NOFOLLOW, dir_fd=store.fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        timer = threading.Timer(.02, lambda: fcntl.flock(fd, fcntl.LOCK_UN))
        timer.start()
        try:
            done = worker_run(MockBrain(), max_seconds=.08)
            self.assertEqual(done['status'], 'time_budget')
            self.assertEqual(done['calls'], 0)
        finally:
            timer.join(1)
            os.close(fd)
            store.close()

    def test_sequence_interpret_then_one_repair_share_worker_budget(self):
        brain, worker, done = self.start_worker(RepairBrain(), calls=2)
        bridge = FileBrainBridge()
        original = bridge.interpret('把方块垒起来', scene())
        self.assertTrue(original.accepted)
        self.assertEqual(original.plan.to_dict(), sequence_plan())
        repaired = bridge.replan('把方块垒起来', scene(), original.plan, repair_feedback())
        worker.join(1)
        self.assertTrue(repaired.accepted)
        self.assertEqual(repaired.calls, 2)
        self.assertEqual(repaired.plan.to_dict(), sequence_plan())
        self.assertEqual(brain.calls.value, 2)
        self.assertEqual(brain.repair_calls.value, 1)
        self.assertEqual(done[0]['calls'], 2)
        self.assertEqual(done[0]['replans'], 1)
        again = bridge.replan('再试一次', scene(), original.plan, repair_feedback())
        self.assertEqual(again.reason, 'replan_budget_exhausted')
        audit = self.read('last-run-audit.json')
        self.assertEqual([item['operation'] for item in audit], ['interpret', 'replan'])
        self.assertNotIn('把方块', json.dumps(audit, ensure_ascii=False))

    def test_invalid_feedback_never_uses_repair_allowance_or_calls(self):
        brain, worker, done = self.start_worker(RepairBrain())
        bridge = FileBrainBridge()
        for feedback in ({**repair_feedback(), 'reason': 'raw SECRET'},
                         {**repair_feedback(), 'url': 'https://example.com'},
                         {**repair_feedback(), 'completed_step_ids': ['s2']},
                         {**repair_feedback(), 'failed_step_id': 's2'}, None):
            result = bridge.replan('修复任务', scene(), sequence_plan(), feedback)
            self.assertEqual(result.reason, 'invalid_feedback')
        self.assertEqual(brain.calls.value, 0)
        result = bridge.replan('修复任务', scene(), sequence_plan(), repair_feedback())
        worker.join(1)
        self.assertTrue(result.accepted)
        self.assertEqual(done[0]['replans'], 1)
        self.assertEqual(brain.repair_calls.value, 1)

    def test_raw_jobs_cannot_reset_worker_repair_allowance(self):
        brain = RepairBrain()
        for job in ('a' * 32, 'b' * 32):
            self.write_job(job, schema=REPLAN_REQUEST_SCHEMA,
                           previous_plan=sequence_plan(), feedback=repair_feedback())
        self.write_job('c' * 32)
        done = worker_run(brain, max_calls=2, max_seconds=.5)
        self.assertEqual(done['calls'], 2)
        self.assertEqual(done['replans'], 1)
        self.assertEqual(done['rejected'], 1)
        self.assertEqual(brain.repair_calls.value, 1)
        self.assertEqual(self.read('b' * 32 + '.result')['result']['reason'], 'replan_budget_exhausted')

    def test_repair_schema_rejects_unknown_inputs_and_wrong_operation(self):
        safe = normalize_input('修复任务', scene(), previous_plan=sequence_plan(), feedback=repair_feedback())
        data = {'schema': REPLAN_REQUEST_SCHEMA, 'simulation_only': True, 'id': 'a' * 32,
                'expires_at': time.time() + 1, **safe}
        self.assertEqual(validate_job(data, 'a' * 32), data)
        for bad in ({**data, 'schema': REQUEST_SCHEMA}, {**data, 'config': '/tmp/key'},
                    {**data, 'feedback': {**repair_feedback(), 'headers': {}}},
                    {**data, 'previous_plan': None, 'feedback': None}):
            with self.assertRaises(ValueError):
                validate_job(bad, 'a' * 32)
        content = {**sequence_plan(), 'message': 'ftp://unapproved.invalid'}
        with self.assertRaises(ValueError):
            normalize_input('修复任务', scene(), previous_plan=content, feedback=repair_feedback())

    def test_repair_cancel_drops_result_and_does_not_reset_allowance(self):
        brain, worker, done = self.start_worker(RepairBrain('slow'))
        bridge = FileBrainBridge()
        results = []
        client = threading.Thread(target=lambda: results.append(
            bridge.replan('修复任务', scene(), sequence_plan(), repair_feedback())))
        client.start()
        self.assertTrue(brain.entered.wait(1))
        bridge.cancel()
        client.join(1)
        worker.join(1)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertEqual(done[0]['replans'], 1)
        self.assertEqual(bridge.replan('修复任务', scene(), sequence_plan(), repair_feedback()).reason,
                         'replan_budget_exhausted')
        self.assertFalse(list(self.root.glob('*.result')))
        self.assertFalse(multiprocessing.active_children())

    def test_provider_and_repair_reason_categories_survive_validation(self):
        for reason in PROVIDER_ERROR_REASONS | REPLAN_ERROR_REASONS:
            with self.subTest(reason=reason):
                result = validate_result(BrainResult(reason=reason).to_dict(), scene())
                self.assertEqual(result.reason, reason)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.plan)


if __name__ == '__main__':
    unittest.main()
