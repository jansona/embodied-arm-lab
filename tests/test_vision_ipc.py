"""Offline-only image IPC tests. Fake providers; never load saved credentials."""
import base64
import fcntl
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

import ipc
from brain import ENDPOINT, MODEL
from vision_adapter import QwenVision, VisionResult, validate_detections
from vision_ipc import (FileVisionBridge, VisionStore, MAX_BYTES, REQUEST_SCHEMA, RESULT_SCHEMA,
    normalize_packet, packet_hash, validate_job, validate_result, worker_status)
from vision_worker import main, worker_run

MP = multiprocessing.get_context('fork')


def jpeg(mode='RGB', fmt='JPEG'):
    image = Image.frombytes('RGB', (320, 240), random.Random(7).randbytes(320 * 240 * 3)).convert(mode)
    buffer = BytesIO()
    image.save(buffer, format=fmt, quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


IMAGE = jpeg()


def packet():
    return {'frame_id': 'frame-test-1', 'captured_at': time.monotonic(), 'images': [
        {'camera_id': cid, 'width': 320, 'height': 240, 'data_url': IMAGE} for cid in ('left', 'right')]}


def payload(frame):
    return {'frame_id': frame['frame_id'], 'detections': [
        {'object_id': obj, 'camera_id': cid, 'frame_id': frame['frame_id'],
         'uv': [100, 100], 'confidence': .95, 'pixel_sigma_px': 3,
         'visibility': 'visible', 'landmark': 'goal_center' if obj == 'goal' else 'top_face_center'}
        for obj in ('red', 'blue', 'goal') for cid in ('left', 'right')]}


def provider_payload(frame, data=None):
    data = json.loads(json.dumps(payload(frame) if data is None else data))
    data['coordinate_space'] = 'normalized_1000'
    dimensions = {i['camera_id']: (i['width'] - 1, i['height'] - 1) for i in frame['images']}
    for item in data['detections']:
        width, height = dimensions[item['camera_id']]
        uv = item.pop('uv')
        item['uv_1000'] = [uv[0] * 1000 / width, uv[1] * 1000 / height] if uv is not None else None
        item['sigma_1000'] = item.pop('pixel_sigma_px') * 1000 / max(width, height)
    return data


def success(frame):
    return VisionResult(True, 'accepted', frame['frame_id'], frame['captured_at'],
                        validate_detections(payload(frame), frame), calls=1)


class FakeVision:
    def __init__(self, mode='ok'):
        self.mode = mode
        self.calls = MP.Value('i', 0)
        self.entered = MP.Event()

    def perceive(self, frame):
        with self.calls.get_lock():
            self.calls.value += 1
        self.entered.set()
        if self.mode == 'slow':
            time.sleep(3)
        if self.mode == 'error':
            raise RuntimeError('SECRET Authorization: Bearer SECRET https://unapproved.invalid')
        if self.mode == 'failure':
            return VisionResult(reason='network_error', frame_id=frame['frame_id'], captured_at=frame['captured_at'])
        if self.mode == 'mismatch':
            result = success(frame)
            return VisionResult(True, 'accepted', 'other-frame', frame['captured_at'], result.detections)
        if self.mode == 'low':
            raw = payload(frame)
            raw['detections'][0]['confidence'] = .1
            return VisionResult(True, 'accepted', frame['frame_id'], frame['captured_at'],
                                validate_detections(raw, frame))
        return success(frame)


class VisionIPCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / '.runtime' / 'vision-ipc'
        self.patcher = patch('vision_ipc.VISION_IPC_ROOT', self.root)
        self.patcher.start()
        self.threads = []

    def tearDown(self):
        for thread in self.threads:
            thread.join(2)
            self.assertFalse(thread.is_alive(), 'Vision worker did not exit')
        self.patcher.stop()
        self.temp.cleanup()

    def start_worker(self, adapter=None, calls=1, seconds=2):
        adapter = adapter or FakeVision()
        done = []
        thread = threading.Thread(target=lambda: done.append(worker_run(adapter, calls, seconds)))
        self.threads.append(thread)
        thread.start()
        until = time.monotonic() + 1
        while not worker_status()['available'] and time.monotonic() < until:
            time.sleep(.002)
        self.assertTrue(worker_status()['available'])
        return adapter, thread, done

    def read(self, name):
        store = VisionStore()
        try:
            return store.read(name)
        finally:
            store.close()

    def job(self, job='a' * 32, frame=None, **changes):
        frame = normalize_packet(frame or packet())
        data = {'schema': REQUEST_SCHEMA, 'simulation_only': True, 'id': job,
                'expires_at': time.time() + .5, 'frame_sha256': packet_hash(frame), 'packet': frame}
        data.update(changes)
        store = VisionStore()
        try:
            store.write(job + '.job', data)
        finally:
            store.close()
        return job

    def test_large_rgb_roundtrip_and_safe_audit(self):
        frame = packet()
        self.assertGreater(len(json.dumps(frame)), ipc.MAX_BYTES)
        adapter, worker, done = self.start_worker()
        result = FileVisionBridge().perceive(frame)
        worker.join(1)
        self.assertTrue(result.accepted)
        self.assertEqual(result.frame_id, frame['frame_id'])
        self.assertEqual(result.captured_at, frame['captured_at'])
        self.assertEqual(len(result.detections), 6)
        self.assertEqual(adapter.calls.value, 1)
        self.assertEqual(done[0]['status'], 'call_budget')
        self.assertFalse(worker_status()['available'])
        audit = self.read('last-run-audit.json')
        self.assertEqual(audit[0]['detection_count'], 6)
        self.assertNotIn('data_url', json.dumps(audit))
        self.assertNotIn('detections', json.dumps(audit))
        self.assertNotIn('uv', json.dumps(audit))
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in self.root.iterdir()))

    def test_real_adapter_fake_http_transport_end_to_end(self):
        frame = packet()
        calls = MP.Value('i', 0)

        def handler(request):
            self.assertEqual(str(request.url), ENDPOINT)
            body = json.loads(request.content)
            self.assertEqual(body['model'], MODEL)
            self.assertNotIn('world_xyz', json.dumps(body))
            self.assertEqual(sum(c['type'] == 'image_url' for c in body['messages'][1]['content']), 2)
            with calls.get_lock():
                calls.value += 1
            return httpx.Response(200, json={'model': MODEL, 'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(provider_payload(frame))}}]})

        adapter = QwenVision(api_key='FAKE-TEST-KEY', max_calls=1, transport=httpx.MockTransport(handler))
        _, worker, _ = self.start_worker(adapter)
        result = FileVisionBridge().perceive(frame)
        worker.join(1)
        self.assertTrue(result.accepted)
        self.assertEqual(calls.value, 1)

    def test_offline_is_fast_without_configuration(self):
        before = time.monotonic()
        with patch.object(QwenVision, 'from_config', create=True, side_effect=AssertionError('No config load')):
            bridge = FileVisionBridge()
            self.assertFalse(bridge.status()['available'])
            self.assertEqual(bridge.perceive(packet()).reason, 'unavailable')
        self.assertLess(time.monotonic() - before, .3)

    def test_provider_exception_is_sanitized(self):
        _, worker, _ = self.start_worker(FakeVision('error'))
        result = FileVisionBridge().perceive(packet())
        worker.join(1)
        self.assertEqual(result.reason, 'adapter_error')
        self.assertFalse(result.accepted)
        self.assertEqual(result.detections, ())
        self.assertNotIn('SECRET', (self.root / 'last-run-audit.json').read_text())

    def test_provider_failure_remains_rejected(self):
        _, worker, _ = self.start_worker(FakeVision('failure'))
        result = FileVisionBridge().perceive(packet())
        worker.join(1)
        self.assertEqual(result.reason, 'network_error')
        self.assertFalse(result.accepted)

    def test_rejected_provider_diagnostic_persists_independently(self):
        frame = packet()
        raw = payload(frame)
        raw['detections'][0]['pixel_sigma_px'] = 1
        def handler(_):
            return httpx.Response(200, json={'model': MODEL, 'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(provider_payload(frame, raw)), 'reasoning_content': 'SECRET_REASONING'}}]})
        adapter = QwenVision(api_key='FAKE-TEST-KEY', transport=httpx.MockTransport(handler))
        _, worker, _ = self.start_worker(adapter)
        result = FileVisionBridge().perceive(frame)
        worker.join(1)
        self.assertFalse(result.accepted)
        diagnostic = self.read('last-diagnostic.json')
        self.assertEqual(diagnostic['stage'], 'detections')
        self.assertEqual(diagnostic['error'], 'invalid_pixel_uncertainty')
        self.assertAlmostEqual(diagnostic['detections'][0]['sigma_1000'], 1000 / 319)
        self.assertNotIn('SECRET', json.dumps(diagnostic))
        self.assertNotIn('FAKE-TEST-KEY', json.dumps(diagnostic))

    def test_worker_sanitizes_adapter_diagnostic_again(self):
        adapter = FakeVision('failure')
        adapter.last_diagnostic = {'stage': 'SECRET', 'error': 'SECRET', 'headers': {'key': 'SECRET'},
                                   'detections': [{'object_id': 'SECRET', 'uv': [float('inf'), 0]}]}
        _, worker, _ = self.start_worker(adapter)
        result = FileVisionBridge().perceive(packet())
        worker.join(1)
        self.assertFalse(result.accepted)
        diagnostic = self.read('last-diagnostic.json')
        self.assertNotIn('SECRET', json.dumps(diagnostic, allow_nan=False))
        self.assertEqual(diagnostic['error'], 'invalid_response')

    def test_wrong_frame_and_low_confidence_never_accepted(self):
        for mode in ('mismatch', 'low'):
            with self.subTest(mode=mode):
                _, worker, _ = self.start_worker(FakeVision(mode))
                result = FileVisionBridge().perceive(packet())
                worker.join(1)
                self.assertEqual(result.reason, 'invalid_response')
                self.assertFalse(result.accepted)

    def test_cancel_terminates_helper_and_discards_pixels(self):
        adapter, worker, done = self.start_worker(FakeVision('slow'))
        bridge = FileVisionBridge()
        results = []
        client = threading.Thread(target=lambda: results.append(bridge.perceive(packet())))
        client.start()
        self.assertTrue(adapter.entered.wait(1))
        self.assertEqual(bridge.perceive(packet()).reason, 'busy')
        before = time.monotonic()
        bridge.cancel()
        client.join(1)
        worker.join(1)
        self.assertLess(time.monotonic() - before, .4)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertEqual(done[0]['cancelled'], 1)
        self.assertFalse(list(self.root.glob('*.result')))
        self.assertFalse(multiprocessing.active_children())

    def test_cancel_during_jpeg_preparation_creates_no_job(self):
        bridge = FileVisionBridge()
        entered, release = threading.Event(), threading.Event()
        results = []

        def slow(packet):
            entered.set()
            release.wait(1)
            return normalize_packet(packet)

        with patch('vision_ipc.normalize_packet', side_effect=slow):
            client = threading.Thread(target=lambda: results.append(bridge.perceive(packet())))
            client.start()
            self.assertTrue(entered.wait(1))
            bridge.cancel()
            release.set()
            client.join(1)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertFalse(list(self.root.glob('*.job')))

    def test_cancel_before_worker_makes_no_provider_call(self):
        job = self.job()
        store = VisionStore()
        try:
            store.write(job + '.cancel', {'cancelled': True})
        finally:
            store.close()
        adapter = FakeVision()
        done = worker_run(adapter, max_seconds=.12)
        self.assertEqual(adapter.calls.value, 0)
        self.assertEqual(done['cancelled'], 1)

    def test_cancel_survives_failed_cancel_marker_write(self):
        adapter, worker, _ = self.start_worker(FakeVision('slow'))
        bridge = FileVisionBridge()
        results = []
        client = threading.Thread(target=lambda: results.append(bridge.perceive(packet())))
        client.start()
        self.assertTrue(adapter.entered.wait(1))
        with patch('vision_ipc.VisionStore.write', side_effect=OSError('unavailable')):
            bridge.cancel()
        client.join(1)
        worker.join(1)
        self.assertEqual(results[0].reason, 'cancelled')
        self.assertFalse(list(self.root.glob('*.result')))

    def test_response_digest_is_checked_against_original_images(self):
        store = VisionStore()
        store.close()

        def forge():
            until = time.monotonic() + 1
            while not list(self.root.glob('*.job')) and time.monotonic() < until:
                time.sleep(.002)
            jobs = list(self.root.glob('*.job'))
            if not jobs:
                return
            job = jobs[0].stem
            data = self.read(job + '.job')
            store = VisionStore()
            try:
                store.write(job + '.result', {'schema': RESULT_SCHEMA, 'id': job,
                    'expires_at': data['expires_at'], 'frame_sha256': '0' * 64,
                    'result': success(data['packet']).to_dict()})
            finally:
                store.close()

        thread = threading.Thread(target=forge)
        thread.start()
        with patch('vision_ipc.worker_status', return_value={'available': True, 'remaining_calls': 1}):
            result = FileVisionBridge().perceive(packet())
        thread.join(1)
        self.assertEqual(result.reason, 'invalid_response')
        self.assertFalse(result.accepted)

    def test_stale_expired_and_extra_field_jobs_use_no_calls(self):
        first = self.job()
        data = self.read(first + '.job')
        data['packet']['captured_at'] -= 61
        data['frame_sha256'] = packet_hash(data['packet'])
        store = VisionStore()
        try:
            store.write(first + '.job', data)
        finally:
            store.close()
        self.job('b' * 32, expires_at=time.time() - 1)
        self.job('c' * 32, headers={'Authorization': 'SECRET'})
        adapter = FakeVision()
        done = worker_run(adapter, max_seconds=.12)
        self.assertEqual(adapter.calls.value, 0)
        self.assertEqual(done['rejected'], 3)

    def test_stuck_provider_is_killed_at_worker_deadline(self):
        self.job()
        before = time.monotonic()
        done = worker_run(FakeVision('slow'), max_calls=1, max_seconds=.07)
        self.assertLess(time.monotonic() - before, .4)
        self.assertEqual(done['calls'], 1)
        self.assertEqual(self.read('a' * 32 + '.result')['result']['reason'], 'timeout')
        self.assertFalse(multiprocessing.active_children())

    def test_frame_that_ages_in_flight_never_delivers_accepted_pixels(self):
        self.job()
        with patch('vision_ipc.MAX_FRAME_AGE', .08):
            done = worker_run(FakeVision('slow'), max_calls=1, max_seconds=.4)
        result = self.read('a' * 32 + '.result')['result']
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'stale_frame')
        self.assertEqual(result['detections'], [])
        self.assertEqual(done['calls'], 1)
        self.assertFalse(multiprocessing.active_children())

    def test_request_expiry_discards_provider_result(self):
        self.job(expires_at=time.time() + .06)
        worker_run(FakeVision('slow'), max_calls=1, max_seconds=.3)
        self.assertFalse(list(self.root.glob('*.result')))
        self.assertEqual(self.read('last-run-audit.json')[0]['reason'], 'expired')

    def test_strict_packet_fields_cannot_carry_state_or_config(self):
        for key, value in (('world_xyz', [1, 2, 3]), ('depth', []), ('calibration', {}),
                           ('api_key', 'SECRET'), ('headers', {}), ('endpoint', 'https://x.invalid')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_packet({**packet(), key: value})
        bad = packet()
        bad['images'][0]['camera_to_world'] = []
        with self.assertRaises(ValueError):
            normalize_packet(bad)

    def test_only_complete_rgb_jpeg_with_matching_dimensions(self):
        wrong = ['https://x.invalid/image.jpg', 'file:///tmp/image.jpg',
                 'data:image/png;base64,AAAA', 'data:image/jpeg;base64,!!!!', jpeg('L'), jpeg(fmt='PNG')]
        trailing = base64.b64decode(IMAGE.split(',', 1)[1]) + b'EXTRA SECRET'
        wrong.append('data:image/jpeg;base64,' + base64.b64encode(trailing).decode())
        for url in wrong:
            with self.subTest(url=url[:30]):
                bad = packet()
                bad['images'][0]['data_url'] = url
                with self.assertRaises(ValueError):
                    normalize_packet(bad)
        bad = packet()
        bad['images'][0]['width'] = 321
        with self.assertRaises(ValueError):
            normalize_packet(bad)
        bad = packet()
        bad['images'][1]['camera_id'] = 'left'
        with self.assertRaises(ValueError):
            normalize_packet(bad)

    def test_stale_future_nonfinite_and_oversized_frames_rejected(self):
        for stamp in (time.monotonic() - 61, time.monotonic() + 1, float('nan'), True):
            with self.assertRaises(ValueError):
                normalize_packet({**packet(), 'captured_at': stamp})
        bad = packet()
        bad['images'][0]['data_url'] = 'data:image/jpeg;base64,' + 'A' * MAX_BYTES
        with self.assertRaises(ValueError):
            normalize_packet(bad)

    def test_job_schema_and_image_digest_cannot_be_forged(self):
        job = self.job()
        data = self.read(job + '.job')
        self.assertEqual(validate_job(data, job), data)
        for changes in ({'frame_sha256': '0' * 64}, {'id': 'b' * 32}, {'simulation_only': False},
                        {'headers': {}}, {'url': 'https://x.invalid'}, {'expires_at': float('inf')}):
            with self.assertRaises(ValueError):
                validate_job({**data, **changes}, job)

    def test_result_validation_binds_frame_time_ids_and_safe_pixels(self):
        frame = normalize_packet(packet())
        result = success(frame).to_dict()
        self.assertTrue(validate_result(result, frame).accepted)
        for changes in ({'frame_id': 'different'}, {'captured_at': frame['captured_at'] + .001},
                        {'model': 'other'}, {'provenance': 'synthetic'}, {'calls': True},
                        {'calls': 7}, {'world_xyz': []}, {'reason': 'SECRET'}, {'accepted': False}):
            with self.assertRaises(ValueError):
                validate_result({**result, **changes}, frame)
        for changes in ({'frame_id': 'other'}, {'uv': [400, 100]}, {'pixel_sigma_px': 1},
                        {'confidence': .2}, {'visibility': 'occluded'}, {'camera_id': 'other'},
                        {'landmark': 'bbox_center'}, {'world_xyz': [0, 0, 0]}):
            bad = json.loads(json.dumps(result))
            bad['detections'][0].update(changes)
            with self.assertRaises(ValueError):
                validate_result(bad, frame)

    def test_symlinks_hardlinks_fifos_permissions_and_traversal_rejected(self):
        store = VisionStore()
        outside = Path(self.temp.name) / 'outside'
        outside.write_text('{}')
        outside.chmod(0o600)
        name = 'a' * 32 + '.job'
        try:
            with self.assertRaises(ValueError):
                store.read('../outside')
            (self.root / name).symlink_to(outside)
            with self.assertRaises(OSError):
                store.read(name)
            with self.assertRaises(ValueError):
                store.write(name, {})
            (self.root / name).unlink()
            os.link(outside, self.root / name)
            with self.assertRaises(ValueError):
                store.read(name)
            (self.root / name).unlink()
            os.mkfifo(self.root / name, mode=0o600)
            with self.assertRaises(ValueError):
                store.read(name)
            (self.root / name).unlink()
            store.write(name, {})
            (self.root / name).chmod(0o644)
            with self.assertRaises(ValueError):
                store.read(name)
        finally:
            store.close()
        link = Path(self.temp.name) / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        with patch('vision_ipc.VISION_IPC_ROOT', link), self.assertRaises(ValueError):
            VisionStore()

    def test_file_size_duplicate_json_and_legacy_limits_unchanged(self):
        store = VisionStore()
        name = 'a' * 32 + '.job'
        try:
            with self.assertRaises(ValueError):
                store.write(name, {'huge': 'A' * MAX_BYTES})
            store.write(name, {})
            (self.root / name).write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):
                store.read(name)
            (self.root / name).write_bytes(b'A' * (MAX_BYTES + 1))
            with self.assertRaises(ValueError):
                store.read(name)
        finally:
            store.close()
        self.assertEqual(ipc.MAX_BYTES, 32767)
        legacy = ipc.Store(Path(self.temp.name) / '.runtime' / 'legacy')
        try:
            with self.assertRaises(ValueError):
                legacy.write(name, {'x': 'A' * 32768})
        finally:
            legacy.close()

    def test_worker_lock_and_bounds(self):
        adapter, worker, done = self.start_worker(seconds=.1)
        self.assertTrue(worker_status()['configured'])
        with self.assertRaises(ValueError):
            worker_run(adapter, max_seconds=.01)
        worker.join(1)
        self.assertFalse(worker_status()['available'])
        for args in ({'max_calls': 0}, {'max_calls': 7}, {'max_calls': True},
                     {'max_seconds': 601}, {'max_seconds': 0}, {'max_seconds': float('inf')}):
            with self.assertRaises(ValueError):
                worker_run(FakeVision(), **args)
        with patch.object(QwenVision, 'from_config', create=True, side_effect=AssertionError('No config load')):
            with self.assertRaises(SystemExit):
                main(['--config', '/never-read-config', '--max-calls', '7'])

    def test_transient_status_probe_does_not_block_worker_startup(self):
        store = VisionStore()
        store.write('worker.lock', {})
        fd = os.open('worker.lock', os.O_RDWR | os.O_NOFOLLOW, dir_fd=store.fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        timer = threading.Timer(.02, lambda: fcntl.flock(fd, fcntl.LOCK_UN))
        timer.start()
        try:
            done = worker_run(FakeVision(), max_seconds=.08)
            self.assertEqual(done['status'], 'time_budget')
            self.assertEqual(done['calls'], 0)
        finally:
            timer.join(1)
            os.close(fd)
            store.close()

    def test_explicit_config_uses_approved_loader_and_never_calls_api(self):
        with patch('brain.QwenBrain.from_config', return_value=SimpleNamespace(_api_key='FAKE-TEST-KEY')) as loader:
            with patch('httpx.Client', side_effect=AssertionError('No API call')):
                adapter = QwenVision.from_config('/fake/explicit/config', max_calls=3, timeout=9,
                                                 use_environment_network=True)
        loader.assert_called_once_with('/fake/explicit/config', max_calls=3, timeout=9,
                                       use_environment_network=True)
        self.assertEqual(adapter.max_calls, 3)
        self.assertEqual(adapter.timeout, 9)
        self.assertEqual(adapter.status()['endpoint'], ENDPOINT)
        self.assertNotIn('FAKE-TEST-KEY', json.dumps(adapter.status()))
        with patch('brain.QwenBrain.from_config', side_effect=ValueError('Cannot load approved config')):
            with self.assertRaisesRegex(ValueError, 'Cannot load approved config'):
                QwenVision.from_config('/fake/rejected/config')


if __name__ == '__main__':
    unittest.main()
