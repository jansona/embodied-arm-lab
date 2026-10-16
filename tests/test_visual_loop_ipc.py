"""Offline IPC checks; providers, configuration loading and inference are fake."""
import base64
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time
from unittest.mock import Mock, patch

from PIL import Image
import pytest

import embodied_arm.runtime.visual_loop_ipc as ipc
from embodied_arm.runtime.visual_loop_worker import worker_run, main
from embodied_arm.providers.visual_brain import ACTIONS, SCHEMA, QwenVisualBrain
from embodied_arm.providers.visual_cerebellum import VisualJev

MP = multiprocessing.get_context('fork')


def robot():
    return {'name': 'SO-ARM101', 'arm_dof': 5, 'ee': [.2, 0., .3], 'joints': [0.] * 5,
            'joint_limits': [[-2., 2.]] * 5, 'gripper_openness': 1., 'paused': True,
            'busy': False, 'epoch': 2, 'actions': 4, 'workspace': [[-.5, .5]] * 3}


def packet():
    out = BytesIO()
    Image.new('RGB', (320, 240), 'orange').save(out, format='JPEG')
    url = 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()
    return {'frame_id': 'frame-1', 'captured_at': time.monotonic(), 'images': [
        {'camera_id': cid, 'width': 320, 'height': 240, 'data_url': url} for cid in ('observer', 'wrist')]}


def observation(frame='frame-1'):
    return {'schema': SCHEMA, 'frame_id': frame, 'visual_summary': 'A patterned cup is visible.',
            'local_objective': 'Move slightly toward the visible cup.', 'decision': 'continue',
            'evidence': 'The gripper is visibly above the cup.', 'confidence': .95}


def qwen_payload():
    return {'task': 'Pick up the patterned cup', 'packet': packet(), 'robot': robot(), 'history': None, 'review': False}


class FakeQwen:
    def __init__(self, mode='ok'):
        self.mode = mode
        self.entered = MP.Event()
        self.count = MP.Value('i', 0)
        self.pid = MP.Value('i', 0)

    def observe(self, task, packet, robot, history=None, review=False):
        self.count.value += 1
        self.pid.value = os.getpid()
        self.entered.set()
        if self.mode == 'slow' or (self.mode == 'slow_first' and self.count.value == 1): time.sleep(5)
        if self.mode == 'error': raise RuntimeError('SECRET https://private.invalid Authorization: KEY')
        value = observation(packet['frame_id'])
        if self.mode == 'mismatch': value['frame_id'] = 'wrong-frame'
        if self.mode == 'extra': value['api_key'] = 'SECRET'
        return {'accepted': True, 'reason': 'accepted', 'observation': value, 'calls': self.count.value}


    def close(self): pass


class FakeJev:
    def __init__(self):
        self.count = MP.Value('i', 0)
        self.pid = MP.Value('i', 0)

    def decide(self, observation, robot):
        self.count.value += 1
        self.pid.value = os.getpid()
        probs = {action: float(action == ACTIONS[0]) for action in ACTIONS}
        return {'accepted': True, 'reason': 'accepted', 'action': ACTIONS[0], 'confidence': .9,
                'probabilities': probs, 'proposed_action': ACTIONS[0]}


    def close(self): pass


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / '.runtime' / 'visual-loop-ipc'
    monkeypatch.setattr(ipc, 'VISUAL_IPC_ROOT', root)
    monkeypatch.setattr('embodied_arm.providers.configuration.load_configuration',
                        Mock(side_effect=AssertionError('Actual configuration must never be read')))
    threads = []
    def start(qwen=None, jev=None, calls=1, seconds=1.2):
        qwen, jev = qwen or FakeQwen(), jev or FakeJev()
        done = []
        errors = []
        def run():
            try: done.append(worker_run(qwen, jev, calls, seconds))
            except BaseException as exc: errors.append(type(exc).__name__)
        thread = threading.Thread(target=run)
        threads.append((thread, errors))
        thread.start()
        until = time.monotonic() + .8
        while not ipc.worker_status()['available'] and time.monotonic() < until: time.sleep(.002)
        assert ipc.worker_status()['available'], errors
        return qwen, jev, thread, done
    yield root, start
    for thread, errors in threads:
        thread.join(2)
        assert not thread.is_alive()
        assert errors == []


def read(name):
    store = ipc.VisualStore()
    try: return store.read(name)
    finally: store.close()


def test_both_components_pooled_full_image_roundtrip_and_budgets(runtime):
    root, start = runtime
    qwen, jev, thread, done = start()
    brain, motor = ipc.FileVisualBridge('qwen'), ipc.FileVisualBridge('jev')
    assert brain.status()['max_calls'] == 2
    assert motor.status()['max_calls'] == 1
    frame = packet()
    result = brain.observe('Pick up the patterned cup', frame, robot())
    assert result['accepted'] and result['observation']['frame_id'] == frame['frame_id']
    move = motor.decide(result['observation'], robot())
    assert move['accepted'] and move['action'] == ACTIONS[0]
    assert not motor.status()['available']
    final = brain.observe('Pick up the patterned cup', packet(), robot(),
                         {'previous_observation': result['observation'], 'applied_actions': [{'action': ACTIONS[0], 'applied': True}]}, review=True)
    assert final['accepted'] and final['calls'] == 2
    thread.join(1)
    assert done[0]['calls'] == {'qwen': 2, 'jev': 1}
    assert done[0]['status'] == 'call_budget'
    assert qwen.pid.value == jev.pid.value != os.getpid()
    assert qwen.count.value == 2 and jev.count.value == 1
    audit = json.dumps(read('last-run-audit.json'))
    assert all(word not in audit for word in ('data_url', 'visual_summary', 'patterned cup', 'Authorization'))
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.iterdir())


def test_offline_and_closed_do_not_load_config(runtime):
    bridge = ipc.FileVisualBridge()
    assert not bridge.status()['available']
    assert bridge.observe(**qwen_payload())['reason'] == 'unavailable'
    bridge.close()
    assert bridge.status()['status'] == 'closed'
    assert bridge.observe(**qwen_payload())['reason'] == 'closed'
    assert ipc.FileVisualBridge('jev').observe(**qwen_payload())['reason'] == 'invalid_input'
    with pytest.raises(ValueError): ipc.FileVisualBridge('other')


@pytest.mark.parametrize('mode', ['error', 'mismatch', 'extra'])
def test_bad_provider_output_rejected_and_sanitized(runtime, mode):
    _, start = runtime
    start(FakeQwen(mode), seconds=.3)
    result = ipc.FileVisualBridge().observe(**qwen_payload())
    assert not result['accepted'] and result['observation'] is None
    assert result['reason'] in ('worker_error', 'invalid_response')
    assert 'SECRET' not in json.dumps(result) + json.dumps(read('last-run-audit.json'))


def test_cancel_reaps_child_then_new_job_keeps_session_and_budget(runtime):
    _, start = runtime
    qwen, _, thread, done = start(FakeQwen('slow_first'))
    bridge = ipc.FileVisualBridge()
    original_status = bridge.status()
    result = []
    caller = threading.Thread(target=lambda: result.append(bridge.observe(**qwen_payload())))
    caller.start()
    assert qwen.entered.wait(.8)
    old_pid = qwen.pid.value
    bridge.cancel()
    caller.join(.7)
    until = time.monotonic() + .7
    while bridge.status()['busy'] and time.monotonic() < until: time.sleep(.002)
    assert result[0]['reason'] == 'cancelled'
    assert bridge.status()['available']
    assert bridge.status()['session_id'] == original_status['session_id']
    assert bridge.status()['remaining_calls'] == 1
    assert bridge.status()['remaining_seconds'] < original_status['remaining_seconds']
    with pytest.raises(ProcessLookupError): os.kill(old_pid, 0)
    assert not read('last-run-audit.json')[0]['delivered']
    resumed = bridge.observe(**qwen_payload())
    assert resumed['accepted'] and resumed['calls'] == 2
    assert qwen.pid.value != old_pid
    assert ipc.FileVisualBridge('jev').decide(observation(), robot())['accepted']
    thread.join(.7)
    assert done[0]['status'] == 'call_budget'
    assert done[0]['calls'] == {'qwen': 2, 'jev': 1}


def test_close_cancels_inflight_and_busy_rejects(runtime):
    _, start = runtime
    qwen, _, thread, _ = start(FakeQwen('slow'))
    bridge = ipc.FileVisualBridge()
    result = []
    caller = threading.Thread(target=lambda: result.append(bridge.observe(**qwen_payload())))
    caller.start()
    assert qwen.entered.wait(.8)
    assert bridge.observe(**qwen_payload())['reason'] == 'busy'
    bridge.close()
    caller.join(.7)
    thread.join(.7)
    assert result[0]['reason'] == 'cancelled'
    assert bridge.status()['status'] == 'closed'


def test_cancel_during_validation_cannot_enqueue(runtime, monkeypatch):
    _, start = runtime
    qwen, _, _, _ = start(seconds=.2)
    bridge = ipc.FileVisualBridge()
    original = ipc.normalize_payload
    def cancel(component, payload):
        bridge.cancel()
        return original(component, payload)
    monkeypatch.setattr(ipc, 'normalize_payload', cancel)
    assert bridge.observe(**qwen_payload())['reason'] == 'cancelled'
    assert qwen.count.value == 0


def test_session_deadline_kills_slow_helper(runtime):
    _, start = runtime
    qwen, _, thread, done = start(FakeQwen('slow'), seconds=.2)
    result = ipc.FileVisualBridge().observe(**qwen_payload())
    thread.join(.8)
    assert not result['accepted']
    assert done[0]['status'] in ('expired', 'time_budget', 'cancelled')
    with pytest.raises(ProcessLookupError): os.kill(qwen.pid.value, 0)


def test_idle_budget_does_not_renew_or_call(runtime):
    _, start = runtime
    qwen, jev, thread, done = start(seconds=.08)
    thread.join(.5)
    assert done[0]['status'] == 'time_budget'
    assert qwen.count.value == jev.count.value == 0
    assert not ipc.worker_status()['available']


@pytest.mark.parametrize('mutation', ['url', 'metadata', 'key', 'old', 'future', 'dimensions', 'task', 'robot', 'history', 'review'])
def test_invalid_requests_never_queue(runtime, mutation):
    data = qwen_payload()
    if mutation == 'url': data['packet']['images'][0]['data_url'] = 'https://unapproved.invalid/image.jpg'
    if mutation == 'metadata': data['packet']['ground_truth'] = {'object': [1, 2, 3]}
    if mutation == 'key': data['packet']['images'][0]['api_key'] = 'SECRET'
    if mutation == 'old': data['packet']['captured_at'] -= 100
    if mutation == 'future': data['packet']['captured_at'] += 100
    if mutation == 'dimensions': data['packet']['images'][0]['width'] += 1
    if mutation == 'task': data['task'] = 'a' * 2001
    if mutation == 'robot': data['robot']['busy'] = True
    if mutation == 'history': data['history'] = {'endpoint': 'https://unapproved.invalid'}
    if mutation == 'review': data['review'] = 1
    assert ipc.FileVisualBridge().observe(**data)['reason'] == 'invalid_input'
    store = ipc.VisualStore()
    try: assert store.jobs() == []
    finally: store.close()


def test_robot_whitelist_and_jev_gate(runtime):
    data = qwen_payload()
    data['robot']['hidden_object_xyz'] = [1, 2, 3]
    clean = ipc.normalize_payload('qwen', data)
    assert 'hidden_object_xyz' not in clean['robot']
    data = {'robot': robot(), 'observation': observation()}
    data['observation']['decision'] = 'completed'
    assert ipc.FileVisualBridge('jev').decide(**data)['reason'] == 'invalid_input'


def job_data(session='a' * 32):
    payload = ipc.normalize_payload('qwen', qwen_payload())
    return {'schema': ipc.REQUEST_SCHEMA, 'simulation_only': True, 'id': 'b' * 32,
            'session_id': session, 'client_id': 'c' * 32, 'generation': 0, 'component': 'qwen',
            'expires_at': time.time() + 1, 'payload_sha256': ipc.payload_hash(payload), 'payload': payload}


@pytest.mark.parametrize('mutation', ['session', 'hash', 'generation', 'endpoint', 'payload_extra', 'robot_extra'])
def test_job_binding_and_exact_schema(runtime, mutation):
    data = job_data()
    if mutation == 'session': data['session_id'] = 'd' * 32
    if mutation == 'hash': data['payload_sha256'] = '0' * 64
    if mutation == 'generation': data['generation'] = True
    if mutation == 'endpoint': data['endpoint'] = 'https://unapproved.invalid'
    if mutation == 'payload_extra': data['payload']['api_key'] = 'SECRET'
    if mutation == 'robot_extra': data['payload']['robot']['hidden_object'] = [1, 2, 3]
    with pytest.raises(ValueError): ipc.validate_job(data, 'b' * 32, 'a' * 32)


def test_forged_response_digest_rejected(runtime):
    _, start = runtime
    qwen, _, thread, _ = start(FakeQwen('slow'))
    bridge = ipc.FileVisualBridge()
    output = []
    caller = threading.Thread(target=lambda: output.append(bridge.observe(**qwen_payload())))
    caller.start()
    assert qwen.entered.wait(.8)
    store = ipc.VisualStore()
    try:
        name = store.jobs()[0]
        data = store.read(name)
        binding = {k: data[k] for k in ('id', 'session_id', 'client_id', 'generation', 'component', 'expires_at', 'payload_sha256')}
        binding['payload_sha256'] = '0' * 64
        store.write(data['id'] + '.result', {**binding, 'schema': ipc.RESULT_SCHEMA,
                    'result': {'accepted': True, 'reason': 'accepted', 'calls': 1, 'observation': observation()}})
    finally: store.close()
    caller.join(.7)
    thread.join(.7)
    assert output[0]['reason'] == 'invalid_response'


def test_store_rejects_links_fifo_permissions_bad_json(runtime):
    root, _ = runtime
    store = ipc.VisualStore()
    name = 'a' * 32 + '.job'
    try:
        with pytest.raises(ValueError): store.read('../secret')
        target = root.parent / 'other'
        target.write_text('{}')
        (root / name).symlink_to(target)
        with pytest.raises((ValueError, OSError)): store.read(name)
        with pytest.raises(ValueError): store.write(name, {})
        (root / name).unlink()
        os.mkfifo(root / name, 0o600)
        with pytest.raises(ValueError): store.read(name)
        (root / name).unlink()
        store.write(name, {})
        os.link(root / name, root / ('b' * 32 + '.job'))
        with pytest.raises(ValueError): store.read(name)
        (root / ('b' * 32 + '.job')).unlink()
        (root / name).chmod(0o644)
        with pytest.raises(ValueError): store.read(name)
        (root / name).chmod(0o600)
        (root / name).write_text('{"x":1,"x":2}')
        with pytest.raises(ValueError): store.read(name)
        (root / name).write_text('{"x":NaN}')
        with pytest.raises(ValueError): store.read(name)
    finally: store.close()


def test_store_bound_does_not_change_legacy(runtime, monkeypatch):
    from embodied_arm.runtime.ipc import MAX_BYTES as legacy
    assert legacy == 32767
    monkeypatch.setattr(ipc, 'MAX_BYTES', 20)
    store = ipc.VisualStore()
    try:
        with pytest.raises(ValueError): store.write('a' * 32 + '.job', {'x': 'a' * 30})
    finally: store.close()


@pytest.mark.parametrize('calls,seconds', [(0, 1), (201, 1), (True, 1), (1, 0), (1, 601), (1, float('nan'))])
def test_worker_budget_validation(calls, seconds):
    with pytest.raises(ValueError): worker_run(FakeQwen(), FakeJev(), calls, seconds)


def test_cli_loaders_only_explicit_mocked_and_close(runtime, capsys):
    qwen, jev = Mock(), Mock()
    with patch.object(QwenVisualBrain, 'from_config', return_value=qwen) as qload, \
         patch.object(VisualJev, 'from_config', return_value=jev) as jload, \
         patch('embodied_arm.runtime.visual_loop_worker.worker_run', return_value={'status': 'time_budget'}) as run:
        main(['--config', '/fake/outside.json', '--max-calls', '3', '--max-seconds', '9'])
    assert qload.call_args.kwargs['max_calls'] == 4
    assert jload.call_args.kwargs['max_calls'] == 3
    assert run.call_args.args == (qwen, jev, 3, 9.)
    qwen.close.assert_called_once()
    jev.close.assert_called_once()
    assert json.loads(capsys.readouterr().out) == {'status': 'time_budget'}


def test_ready_snapshot_reopens_once_after_atomic_replace(runtime, monkeypatch):
    store = ipc.VisualStore()
    store.write('ready.json', {'version': 1})
    real_fstat = os.fstat
    replaced = []
    def replace_after_open(fd):
        if not replaced:
            replaced.append(True)
            store.write('ready.json', {'version': 2})
        return real_fstat(fd)
    monkeypatch.setattr(ipc.os, 'fstat', replace_after_open)
    try: assert store.read('ready.json') == {'version': 2}
    finally: store.close()


@pytest.mark.parametrize('name', ['ready.json', 'a' * 32 + '.job'])
def test_reopen_never_accepts_repeated_unlink_or_non_snapshot(runtime, monkeypatch, name):
    store = ipc.VisualStore()
    store.write(name, {'version': 1})
    real_fstat = os.fstat
    calls = []
    def replace_every_open(fd):
        calls.append(1)
        store.write(name, {'version': len(calls) + 1})
        return real_fstat(fd)
    monkeypatch.setattr(ipc.os, 'fstat', replace_every_open)
    try:
        with pytest.raises(ValueError): store.read(name)
        assert len(calls) == (2 if name == 'ready.json' else 1)
    finally: store.close()


def test_worker_rejects_old_session_without_inference(runtime):
    _, start = runtime
    qwen, _, thread, done = start(seconds=.2)
    store = ipc.VisualStore()
    try:
        data = job_data()
        store.write(data['id'] + '.job', data)
    finally: store.close()
    thread.join(.7)
    assert done[0]['rejected'] == 1 and qwen.count.value == 0


def test_worker_exclusive_lock_and_startup_probe_retry(runtime, monkeypatch):
    import fcntl
    original = fcntl.flock
    attempts = []
    def initial_probe(fd, operation):
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB and not attempts:
            attempts.append(1)
            raise BlockingIOError()
        return original(fd, operation)
    monkeypatch.setattr('embodied_arm.runtime.visual_loop_worker.fcntl.flock', initial_probe)
    _, start = runtime
    _, _, thread, _ = start(seconds=.2)
    with pytest.raises(ValueError, match='already active'):
        worker_run(FakeQwen(), FakeJev(), 1, .1)
    assert ipc.worker_status()['available']
    thread.join(.7)


def test_large_full_jpeg_packet_exceeds_legacy_without_altering_images(runtime):
    import random
    raw = random.Random(14).randbytes(320 * 240 * 3)
    image = Image.frombytes('RGB', (320, 240), raw)
    out = BytesIO()
    image.save(out, 'JPEG', quality=85)
    data = qwen_payload()
    for item in data['packet']['images']:
        item['data_url'] = 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()
    assert len(ipc._json(data)) > 32767
    clean = ipc.normalize_payload('qwen', data)
    assert clean['packet'] == data['packet']
    store = ipc.VisualStore()
    try:
        store.write('a' * 32 + '.job', clean)
        assert store.read('a' * 32 + '.job') == clean
    finally: store.close()


def test_cli_sigterm_reaps_pending_child(runtime, monkeypatch):
    qwen, jev = FakeQwen('slow'), FakeJev()
    monkeypatch.setattr(QwenVisualBrain, 'from_config', lambda *a, **kw: qwen)
    monkeypatch.setattr(VisualJev, 'from_config', lambda *a, **kw: jev)
    worker = MP.Process(target=main, args=(['--config', '/fake/outside.json', '--max-calls', '1', '--max-seconds', '3'],))
    caller = None
    worker.start()
    try:
        until = time.monotonic() + 1
        while not ipc.worker_status()['available'] and time.monotonic() < until: time.sleep(.002)
        assert ipc.worker_status()['available']
        output = []
        caller = threading.Thread(target=lambda: output.append(ipc.FileVisualBridge().observe(**qwen_payload())))
        caller.start()
        assert qwen.entered.wait(1)
        pid = qwen.pid.value
        worker.terminate()
        worker.join(1)
        assert not worker.is_alive() and worker.exitcode == 0
        caller.join(1)
        assert not caller.is_alive() and not output[0]['accepted']
        with pytest.raises(ProcessLookupError): os.kill(pid, 0)
        assert not ipc.worker_status()['available']
    finally:
        if worker.is_alive(): worker.kill()
        worker.join(1)
        worker.close()
        if caller: caller.join(1)
