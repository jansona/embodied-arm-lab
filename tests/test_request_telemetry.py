"""Metadata-only observability with mock HTTP; never live requests or config."""
import base64
from io import BytesIO
import json
import time
from types import SimpleNamespace
import httpx
from PIL import Image
import pytest
from embodied_arm.providers.brain import QwenBrain, MODEL
from embodied_arm.providers.vision_adapter import QwenVision
from embodied_arm.providers.so101_jev import SO101JevAdapter, build_observation
from embodied_arm.providers.request_telemetry import RequestTelemetry, sanitize_request_telemetry

RID = '12345678-1234-1234-1234-123456789abc'
SCENE = {'blocks': {'red': [.2, 0, .01]}, 'ee': [.2, 0, .1], 'goal': [.3, 0, .01]}


def frame():
    out = BytesIO()
    Image.new('RGB', (64, 48), 'red').save(out, format='JPEG')
    url = 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()
    return {'frame_id': 'test-frame', 'captured_at': time.monotonic(), 'images': [
        {'camera_id': cid, 'width': 64, 'height': 48, 'data_url': url} for cid in ('left', 'right')]}


def observation():
    robot = {'name': 'SO-ARM101', 'arm_dof': 5, 'ee': [.2, 0, .06], 'joints': [0]*5,
        'joint_limits': [[-2, 2]]*5, 'gripper_openness': .4, 'gripper_angle': .2,
        'paused': False, 'busy': False, 'epoch': 1, 'actions': 0,
        'workspace': [[.06, .38], [-.25, .25], [.006, .36]], 'cube_edge_m': .02}
    return build_observation({'source': 'rgb_only', 'kind': 'gripper', 'target_openness': 1, 'tolerance': .02}, robot)


@pytest.mark.parametrize('component', ['brain', 'vision', 'cerebellum'])
def test_timeout_marks_ambiguous_processing_and_never_retries(component):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout('SECRET provider raw error', request=request)
    transport = httpx.MockTransport(handler)
    if component == 'brain':
        adapter = QwenBrain(api_key='SECRET', transport=transport)
        result = adapter.interpret('PRIVATE prompt', SCENE)
    elif component == 'vision':
        adapter = QwenVision(api_key='SECRET', transport=transport)
        result = adapter.perceive(frame())
    else:
        adapter = SO101JevAdapter(api_key='SECRET', transport=transport)
        result = adapter.decide(observation())
    assert result.reason == 'timeout' and count == 1
    meta = adapter.last_request
    assert meta['component'] == component and meta['processing_uncertain']
    assert meta['automatic_retry'] is False and meta['http_status'] is None
    assert meta['usage'] == {} and meta['request_id'] is None
    assert 0 <= meta['elapsed_seconds'] < 1 and meta['started_at_unix'] <= time.time()
    assert 'SECRET' not in json.dumps(meta) and 'PRIVATE' not in json.dumps(meta)


def test_brain_records_usage_and_request_id_without_changing_result_schema():
    plan = {'action': 'pick', 'object_id': 'red', 'target_id': None, 'message': '尝试'}
    payload = {'model': MODEL, 'id': 'chatcmpl-ignored', 'usage': {'prompt_tokens': 123,
        'completion_tokens': 10, 'total_tokens': 133, 'details': {'SECRET': 'raw'}},
        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(plan)}}]}
    adapter = QwenBrain(api_key='SECRET', transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload, headers={'x-request-id': RID, 'Authorization': 'SECRET'})))
    result = adapter.interpret('PRIVATE', SCENE)
    assert result.accepted and 'telemetry' not in result.to_dict()
    meta = adapter.last_request
    assert meta['http_status'] == 200 and meta['request_id'] == RID
    assert meta['usage'] == {'prompt_tokens': 123, 'completion_tokens': 10, 'total_tokens': 133}
    assert meta['outcome'] == 'accepted' and not meta['processing_uncertain']
    assert 'SECRET' not in json.dumps(meta) and 'PRIVATE' not in json.dumps(meta)


def test_jev_records_valid_metadata_even_on_rejected_model_answer():
    adapter = SO101JevAdapter(api_key='SECRET', transport=httpx.MockTransport(lambda _: httpx.Response(
        200, json={'request_id': RID, 'usage': {'input_tokens': 7}, 'answers': {}})))
    assert adapter.decide(observation()).reason == 'invalid_response'
    assert adapter.last_request['request_id'] == RID
    assert adapter.last_request['usage'] == {'input_tokens': 7}


def test_vision_records_http_outcome_with_no_body_or_error_capture():
    adapter = QwenVision(api_key='SECRET', transport=httpx.MockTransport(lambda _: httpx.Response(
        403, json={'message': 'SECRET'}, headers={'x-request-id': RID})))
    assert adapter.perceive(frame()).reason == 'http_error_403'
    assert adapter.last_request['http_status'] == 403
    assert adapter.last_request['outcome'] == 'http_error'
    assert adapter.last_request['request_id'] == RID
    assert 'SECRET' not in json.dumps(adapter.last_request)


def test_sanitizer_rebuilds_whitelist_and_drops_bad_ids_usage():
    owner = SimpleNamespace()
    trace = RequestTelemetry(owner, 'brain')
    trace.payload({'id': 'SECRET https://example.com', 'usage': {'prompt_tokens': True,
        'completion_tokens': -1, 'total_tokens': 1_000_000_001, 'output_tokens': 5, 'raw': 'SECRET'}})
    trace.finish('timeout')
    clean = sanitize_request_telemetry({**owner.last_request, 'raw_body': 'SECRET', 'automatic_retry': True})
    assert clean['request_id'] is None and clean['usage'] == {'output_tokens': 5}
    assert clean['automatic_retry'] is False and 'SECRET' not in json.dumps(clean)
    for value in (None, {}, {'component': []}, {**clean, 'elapsed_seconds': float('nan')}):
        assert sanitize_request_telemetry(value) is None


def test_invalid_input_never_creates_request_record():
    adapter = QwenBrain(api_key='SECRET', transport=httpx.MockTransport(lambda _: pytest.fail('No request')))
    assert adapter.interpret('', SCENE).reason == 'invalid_input'
    assert adapter.last_request is None


@pytest.mark.parametrize('component', ['brain', 'vision', 'cerebellum'])
@pytest.mark.parametrize('malformed', [False, True])
def test_worker_audit_persists_sanitized_metadata_without_public_result_expansion(tmp_path, monkeypatch, component, malformed):
    import multiprocessing
    import threading
    from embodied_arm.providers.brain import BrainResult
    from embodied_arm.providers.vision_adapter import VisionResult
    from embodied_arm.providers.so101_jev import SO101Decision
    from embodied_arm.runtime import brain_ipc, brain_worker, vision_ipc, vision_worker, so101_ipc, so101_worker

    calls = multiprocessing.get_context('fork').Value('i', 0)
    class FakeAdapter:
        def metadata(self):
            with calls.get_lock(): calls.value += 1
            trace = RequestTelemetry(self, component)
            trace.payload({'request_id': RID, 'usage': {'input_tokens': 12}})
            trace.finish('timeout')
            self.last_request.update(raw_prompt='SECRET', headers={'Authorization': 'SECRET'}, automatic_retry=True)
            if malformed: self.last_request['component'] = 'wrong_component'
        def interpret(self, *args):
            self.metadata()
            return BrainResult(reason='timeout', calls=1)
        def perceive(self, packet):
            self.metadata()
            return VisionResult(reason='timeout', frame_id=packet['frame_id'], captured_at=packet['captured_at'], calls=1)
        def decide(self, obs):
            self.metadata()
            return SO101Decision(reason='timeout', axis_index=obs['axis_index'])

    root = tmp_path / component
    if component == 'brain':
        module, worker = brain_ipc, brain_worker
        monkeypatch.setattr(module, 'BRAIN_IPC_ROOT', root)
        bridge = module.FileBrainBridge()
        invoke = lambda: bridge.interpret('PRIVATE', SCENE)
    elif component == 'vision':
        module, worker = vision_ipc, vision_worker
        monkeypatch.setattr(module, 'VISION_IPC_ROOT', root)
        bridge = module.FileVisionBridge()
        invoke = lambda: bridge.perceive(frame())
    else:
        module, worker = so101_ipc, so101_worker
        monkeypatch.setattr(module, 'SO101_IPC_ROOT', root)
        bridge = module.FileSO101Bridge()
        paused = observation()
        paused['robot']['paused'] = True
        invoke = lambda: bridge.decide(paused)
    results = []
    thread = threading.Thread(target=lambda: results.append(worker.worker_run(FakeAdapter(), max_calls=1, max_seconds=2)))
    thread.start()
    try:
        until = time.monotonic() + 1
        while not module.worker_status()['available'] and time.monotonic() < until: time.sleep(.002)
        assert module.worker_status()['available']
        result = invoke()
        assert result.reason == 'timeout'
        assert 'telemetry' not in result.to_dict() and 'request_telemetry' not in result.to_dict()
    finally:
        thread.join(3)
    assert not thread.is_alive() and calls.value == 1
    audit_text = (root / 'last-run-audit.json').read_text()
    assert 'SECRET' not in audit_text and 'PRIVATE' not in audit_text
    records = json.loads(audit_text)
    assert len(records) == 1
    entry = records[0]
    assert entry['automatic_retry'] is False and entry['processing_uncertain'] is True
    assert entry['request_metadata_available'] is not malformed
    if malformed:
        assert entry['request_telemetry'] is None
    else:
        meta = entry['request_telemetry']
        assert meta['request_id'] == RID and meta['usage'] == {'input_tokens': 12}
        assert meta['outcome'] == 'timeout' and meta['component'] == component
