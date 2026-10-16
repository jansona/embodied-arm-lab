"""Independent offline failure recovery: real orchestration, MOCK boundaries only."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from embodied_arm.services.sequence_execution import SequenceExecutionService
from embodied_arm.services.task_service import TaskService
from tests.e2e_conversation.test_dynamic_loop import LoopBrain, LoopSource, settle
from tests.test_sequence_execution import Primitive, outcome, sequence

FIRST = 'Moving RGB anchors unavailable'
RETRY = 'Fresh recovery frame has occluded red landmark'


class RecordingBrain(LoopBrain):
    """MOCK provider; records precisely what model history receives."""
    def __init__(self):
        super().__init__()
        self.available = True
        self.histories = []

    def status(self):
        return {'available': self.available}

    def interpret(self, text, scene, history=None):
        self.plans.append(deepcopy(scene))
        self.histories.append(deepcopy(history))
        plan = sequence(('pick', 'red', None))
        plan['message'] = 'MOCK proposal: pick red'
        return SimpleNamespace(accepted=True, provenance='MOCK-NOT-LIVE',
                               plan=SimpleNamespace(to_dict=lambda: plan))


def fixture():
    failure = outcome('failed', calls=4, actions=3, holding='empty', phase='approach')
    failure.update(reason='perception_rejected', perception_detail=FIRST,
                   recovery_detail=RETRY, driver={'reason': 'perception_rejected'})
    primitive = Primitive(failure)
    source = LoopSource()
    brain = RecordingBrain()
    service = TaskService(scene_source=source, brain=brain,
                          executor=SequenceExecutionService(primitive))
    return source, brain, primitive, service


def test_public_failure_preserves_first_error_retry_and_applied_counts():
    _, _, primitive, service = fixture()
    service.submit('MOCK pick red')
    status = settle(service)
    assert status['state'] == 'failed' and status['error'] == 'execution_unverified'
    result = status['outcome']
    assert result['failure'] == {
        'step_id': 's1', 'phase': 'approach', 'reason': 'perception_rejected',
        'first_perception_error': FIRST, 'retry_perception_error': RETRY,
        'driver_reason': 'perception_rejected'}
    assert result['calls'] == 4 and result['actions'] == 3
    assert status['motion_executed'] is True and not result['verified']
    assert len(primitive.requests) == 1


def test_new_command_after_failure_observes_and_replies_without_stop():
    source, brain, _, service = fixture()
    service.submit('MOCK pick red')
    failed = settle(service)
    previous_frame = source.version
    service.submit('MOCK inspect red again', execute=False)
    recovered = settle(service)
    assert recovered['state'] == 'plan_preview' and recovered['error'] is None
    assert recovered['id'] != failed['id'] and source.version > previous_frame
    assert len(brain.plans) == 2 and recovered['motion_executed'] is False


def test_unavailable_vision_calls_no_brain_and_reconnect_accepts_new_command():
    source, brain, primitive, service = fixture()
    source.error = 'vision_unavailable'
    service.submit('MOCK inspect red', execute=False)
    blocked = settle(service)
    assert blocked['state'] == 'blocked' and blocked['error'] == 'vision_unavailable'
    assert brain.plans == [] and primitive.requests == []
    source.error = None
    service.submit('MOCK inspect red after reconnect', execute=False)
    recovered = settle(service)
    assert recovered['state'] == 'plan_preview' and len(brain.plans) == 1
    assert source.version == 1 and not recovered['motion_executed']


def test_unavailable_brain_rejects_without_observation_then_reconnect_works():
    source, brain, primitive, service = fixture()
    brain.available = False
    with pytest.raises(ValueError, match='brain_unavailable'):
        service.submit('MOCK inspect red', execute=False)
    assert source.version == 0 and brain.plans == [] and primitive.requests == []
    brain.available = True
    service.submit('MOCK inspect red after reconnect', execute=False)
    assert settle(service)['state'] == 'plan_preview'
    assert len(brain.plans) == 1 and source.version == 1


def test_runtime_failure_notice_is_retained_outside_model_conversation():
    _, brain, _, service = fixture()
    request = 'MOCK pick red'
    service.submit(request)
    failed = settle(service)
    notices = failed['runtime_notices']
    assert len(notices) == 1
    notice = notices[0]
    assert notice['kind'] == 'task_failure'
    assert notice['request_id'] == failed['id'] and notice['request_text'] == request
    assert notice['state'] == 'failed' and notice['reason'] == 'execution_unverified'
    assert notice['motion_executed'] is True and isinstance(notice['at'], (float, int))
    history = deepcopy(failed['history'])
    assert history == [{'role': 'user', 'content': request},
                       {'role': 'assistant', 'content': 'MOCK proposal: pick red'}]
    service.submit('MOCK inspect after failure', execute=False)
    recovered = settle(service)
    assert recovered['runtime_notices'] == notices
    assert brain.histories[-1] == history
    assert all(item.get('kind') != 'task_failure' for item in recovered['history'])


def test_stop_keeps_pending_busy_then_new_command_uses_current_scene_and_robot():
    """MOCK blocked provider and encoders; real cancellation and wrapper forwarding."""
    import threading
    source, brain, primitive, service = fixture()
    entered, release = threading.Event(), threading.Event()
    interpret = brain.interpret
    def slow_first(text, scene, history=None):
        if text == 'MOCK first request':
            entered.set()
            assert release.wait(3), 'Test failed to release MOCK provider'
        return interpret(text, scene, history)
    brain.interpret = slow_first
    robot = {'joints': [0.] * 5, 'gripper_openness': .2, 'holding_status': 'unknown'}
    primitive.robot_state = lambda: deepcopy(robot)
    reconciled = []
    def reconcile():
        reconciled.append(source.version)
        return {'verified': False, 'reason': 'MOCK uncertain holding', 'holding_state': 'unknown',
                'certificate': {'private_fixture_evidence': 'MUST_NOT_APPEAR'}}
    primitive.reconcile_observation = reconcile
    try:
        service.submit('MOCK first request', execute=False)
        assert entered.wait(3)
        stopped = service.stop()
        assert stopped['state'] == 'cancelled' and stopped['pending'] is True
        with pytest.raises(ValueError, match='task_busy'):
            service.submit('MOCK premature retry', execute=False)
        assert source.version == 1
    finally:
        release.set()
    assert settle(service)['state'] == 'cancelled'
    robot.update(joints=[.1, .2, .3, .4, .5], gripper_openness=.7)
    service.submit('MOCK inspect current scene', execute=False)
    result = settle(service)
    assert result['state'] == 'plan_preview' and not result['pending']
    assert brain.plans[-1]['mock_scene_version'] == 2
    assert brain.plans[-1]['robot_state'] == robot
    assert reconciled == [1, 2] and primitive.resets == 0
    holding_events = [e['detail'] for e in result['events']
                      if isinstance(e.get('detail'), dict) and e['detail'].get('phase') == 'holding_reacquisition']
    assert len(holding_events) == 1
    assert set(holding_events[0]['holding']) == {'verified', 'reason', 'holding_state'}
    assert 'MUST_NOT_APPEAR' not in str(result) and 'certificate' not in str(holding_events)


def test_measured_cycle_progress_reaches_review_without_resetting_global_budget():
    """MOCK measurements/verdicts cross real sequence and task orchestration."""
    from tests.e2e_conversation.test_resumable_camera_loop import sequence_task_fixture
    source, brain, primitive, service = sequence_task_fixture(final_verified=False)
    progress = {
        'phase': 'retreat', 'goal_start_xyz_m': [.2, .1, .05],
        'goal_end_xyz_m': [.21, .11, .06], 'error_start_xyz_m': [.04, .02, .01],
        'error_end_xyz_m': [.02, .01, .005], 'object_start_xyz_m': [.1, .1, .02],
        'object_end_xyz_m': [.11, .1, .03], 'observations': 3,
    }
    primitive.results[1]['progress'] = [deepcopy(progress)]
    service.submit('MOCK move red with changed goal')
    status = settle(service)
    review = brain.repairs[0]
    context = review['feedback']['cycle_context']
    assert context['progress'] == [progress]
    assert context['residual_xyz_m'] == progress['error_end_xyz_m']
    assert context['remaining_global_calls'] == 4
    assert review['scene']['mock_scene_version'] > brain.plans[0]['mock_scene_version']
    assert [options['max_calls'] for _, options in primitive.requests] == [20, 19, 4]
    assert len({options['deadline'] for _, options in primitive.requests}) == 1
    assert status['state'] == 'failed' and status['outcome']['verified'] is False
    assert status['outcome']['calls'] == 17 and status['cycle_reviews'] == 1
