"""Offline orchestration fixtures only: no perception, physics, model, or keys."""
from copy import deepcopy
from collections import deque
import threading
import pytest
from embodied_arm.services.sequence_execution import SequenceExecutionService


def single(action='pick', obj='red', target=None):
    return {'action': action, 'object_id': obj, 'target_id': target, 'message': 'test'}


def sequence(*intents):
    if not intents:
        intents = (('pick', 'red', None), ('place', 'red', 'goal'),
                   ('pick', 'blue', None), ('place', 'blue', 'on:red'))
    return {'action': 'sequence', 'object_id': None, 'target_id': None, 'message': 'test',
            'steps': [{'step_id': f's{i + 1}', 'action': a, 'object_id': o, 'target_id': t}
                      for i, (a, o, t) in enumerate(intents)]}


def outcome(status='completed', *, calls=1, actions=1, holding='held_verified',
            phase='verify_lift', token=None, verified=None):
    return {'status': status, 'verified': status == 'completed' if verified is None else verified,
            'calls': calls, 'actions': actions, 'phase': phase, 'holding_state': holding,
            'resume_token': token, 'reason': 'cycle_budget' if status == 'yielded' else None}


def placed(**kwargs):
    return outcome(holding='empty', phase='verify_release', **kwargs)


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class Primitive:
    validated = False
    experimental_simulation = True
    held_certificate = None
    def __init__(self, *results):
        self.results = deque(results)
        self.requests = []
        self.stops = self.resets = 0
    def readiness(self): return {'available': True, 'reason': None}
    def bounded_execute(self, plan, **options):
        self.requests.append((deepcopy(plan), options))
        result = self.results.popleft()
        return result(plan, options) if callable(result) else deepcopy(result)
    def stop(self): self.stops += 1
    def reset(self): self.resets += 1


def run(service, plan=None, **options):
    return service.bounded_execute(plan or single(), max_calls=options.pop('max_calls', 20),
                                   deadline=options.pop('deadline', 100),
                                   cycle_max_calls=options.pop('cycle_max_calls', 16), **options)


def test_four_steps_preserve_model_order_and_shared_budgets():
    primitive = Primitive(outcome(calls=2), placed(calls=3), outcome(calls=4), placed(calls=2))
    executor = SequenceExecutionService(primitive, clock=Clock())
    result = run(executor, sequence(), max_calls=15, cycle_max_calls=15)
    assert result['verified'] and result['status'] == 'completed'
    assert result['calls'] == result['cumulative_calls'] == 11
    assert result['completed_step_ids'] == ['s1', 's2', 's3', 's4']
    assert [p['object_id'] for p, _ in primitive.requests] == ['red', 'red', 'blue', 'blue']
    assert [o['max_calls'] for _, o in primitive.requests] == [15, 13, 10, 6]
    assert [o['cycle_max_calls'] for _, o in primitive.requests] == [15, 13, 10, 6]
    assert all(o['deadline'] == 100 for _, o in primitive.requests)
    assert primitive.stops == 0 and result['resume_token'] is None


def test_partial_primitive_resume_preserves_completed_prefix_and_owned_tail():
    primitive = Primitive(outcome(calls=2), outcome('yielded', holding='release_pending',
                         phase='retreat', token='primitive-release'), placed(), outcome(), placed())
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor, sequence(), cycle_max_calls=3)
    assert first['status'] == 'yielded' and first['completed_step_ids'] == ['s1']
    assert first['holding_state'] == 'release_pending'
    assert first['resume_token'] != 'primitive-release'
    second = run(executor, single('place', 'red', 'goal'), max_calls=17,
                 resume_token=first['resume_token'])
    assert second['verified'] and second['calls'] == 3 and second['cumulative_calls'] == 6
    assert second['completed_step_ids'] == ['s1', 's2', 's3', 's4']
    assert primitive.requests[2][1]['resume_token'] == 'primitive-release'
    assert [p['action'] for p, _ in primitive.requests].count('pick') == 2
    assert 'primitive-release' not in str(first)
    assert primitive.stops == 0


def test_outer_tokens_rotate_and_old_token_cannot_replay():
    primitive = Primitive(outcome('yielded', phase='close', token='p1'),
                          outcome('yielded', phase='lift', token='p2'), outcome())
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor)
    second = run(executor, resume_token=first['resume_token'])
    assert first['resume_token'] != second['resume_token']
    with pytest.raises(ValueError, match='invalid_resume_token'):
        run(executor, resume_token=first['resume_token'])
    last = run(executor, resume_token=second['resume_token'])
    assert last['verified'] and last['cumulative_calls'] == 3
    assert [o['resume_token'] for _, o in primitive.requests] == [None, 'p1', 'p2']


def test_changed_active_intent_is_rejected_without_consuming_valid_token():
    primitive = Primitive(outcome('yielded', phase='lift', token='p1'), outcome())
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor)
    with pytest.raises(ValueError, match='active_intent_changed'):
        run(executor, single('pick', 'blue'), resume_token=first['resume_token'])
    assert run(executor, resume_token=first['resume_token'])['verified']


def test_active_primitive_rejects_changed_tail_even_with_matching_first_intent():
    primitive = Primitive(outcome('yielded', token='p1'))
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor, sequence())
    revised = sequence(('pick', 'red', None), ('place', 'red', 'on:blue'))
    with pytest.raises(ValueError, match='active_intent_changed'):
        run(executor, revised, resume_token=first['resume_token'])
    assert len(primitive.requests) == 1


def test_boundary_singleton_preserves_pending_tail_without_primitive_token():
    primitive = Primitive(outcome(), placed(), outcome(), placed())
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor, sequence(), cycle_max_calls=1)
    assert first['phase'] == 'step_boundary' and first['completed_step_ids'] == ['s1']
    last = run(executor, single('place', 'red', 'goal'), resume_token=first['resume_token'])
    assert last['verified'] and last['completed_step_ids'] == ['s1', 's2', 's3', 's4']
    assert primitive.requests[1][1]['resume_token'] is None


def test_boundary_revision_requires_explicit_sequence_and_cannot_replay_prefix():
    primitive = Primitive(outcome(), placed())
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor, sequence(), cycle_max_calls=1)
    token = first['resume_token']
    with pytest.raises(ValueError, match='tail_revision_requires_sequence'):
        run(executor, single('place', 'red', 'on:blue'), resume_token=token)
    with pytest.raises(ValueError, match='completed_step_replay'):
        run(executor, sequence(), resume_token=token)
    revised = sequence(('place', 'red', 'on:blue'))
    last = run(executor, revised, resume_token=token)
    assert last['verified'] and last['revision'] == 1
    assert last['completed_step_ids'] == ['s1', 's2']
    assert primitive.requests[-1][0]['target_id'] == 'on:blue'


def test_resume_cannot_extend_original_deadline_or_reset_global_budget():
    clock = Clock()
    primitive = Primitive(outcome('yielded', calls=2, token='p1'), outcome('yielded', token='p2'))
    executor = SequenceExecutionService(primitive, clock=clock)
    first = run(executor, max_calls=3, deadline=10)
    last = run(executor, max_calls=200, deadline=1000, resume_token=first['resume_token'])
    assert primitive.requests[-1][1]['max_calls'] == 1
    assert primitive.requests[-1][1]['deadline'] == 10
    assert last['status'] == 'failed' and last['reason'] == 'call_budget_exhausted'
    assert last['resume_token'] is None and last['cumulative_calls'] == 3


def test_smaller_remaining_budget_stays_smaller_across_further_resumes():
    primitive = Primitive(outcome('yielded', token='p1'), outcome('yielded', token='p2'),
                          outcome('yielded', token='p3'))
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor)
    second = run(executor, max_calls=2, resume_token=first['resume_token'])
    last = run(executor, max_calls=200, resume_token=second['resume_token'])
    assert primitive.requests[-1][1]['max_calls'] == 1
    assert last['reason'] == 'call_budget_exhausted'


def test_deadline_blocks_next_step_after_completed_primitive():
    clock = Clock()
    def expired(plan, options):
        clock.now = 10
        return outcome()
    primitive = Primitive(expired, placed())
    executor = SequenceExecutionService(primitive, clock=clock)
    result = run(executor, sequence(), deadline=10)
    assert result['reason'] == 'time_budget' and result['calls'] == 1
    assert len(primitive.requests) == 1 and primitive.stops == 1


@pytest.mark.parametrize('bad', [
    {}, {'action': 'execute'}, single('pick', target='goal'), single('place', target='on:red'),
    single('place', target='on:green'), {**single(), 'xyz': [0, 0, 0]},
    {**single(), 'object_id': []}, {**single(), 'target_id': []},
    {**sequence(), 'steps': []}, {**sequence(), 'steps': [{'step_id': 's1', 'action': 'open'}]},
])
def test_invalid_plans_never_reach_primitive(bad):
    primitive = Primitive()
    executor = SequenceExecutionService(primitive, clock=Clock())
    with pytest.raises(ValueError):
        executor.bounded_execute(bad, max_calls=20, deadline=100, cycle_max_calls=16)
    assert not primitive.requests


@pytest.mark.parametrize('bad', [
    outcome('completed', verified=False), outcome('completed', holding='unknown'),
    outcome('yielded', token=None), outcome('yielded', calls=0, actions=0, token='no-progress'),
    {**outcome(), 'calls': True}, {**outcome(), 'actions': -1},
    {**outcome(), 'holding_state': 'maybe'}, {**outcome(), 'holding_state': []},
    {**outcome(), 'phase': None},
    outcome(calls=17),
])
def test_invalid_or_unverified_outcomes_fail_closed(bad):
    primitive = Primitive(bad)
    executor = SequenceExecutionService(primitive, clock=Clock())
    result = run(executor)
    assert not result['verified'] and result['status'] == 'failed'
    assert result['resume_token'] is None and primitive.stops == 1


def test_perception_failure_is_not_turned_into_stack_success():
    primitive = Primitive(outcome('failed', holding='release_pending', phase='verify_release'))
    executor = SequenceExecutionService(primitive, clock=Clock())
    result = run(executor, single('place', 'blue', 'on:red'))
    assert result['status'] == 'failed' and result['completed_step_ids'] == []
    assert result['failed_step_id'] == 's1'


def test_stop_invalidates_checkpoint_and_new_token_is_required():
    primitive = Primitive(outcome('yielded', token='p1'))
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor)
    with pytest.raises(ValueError, match='pending_checkpoint_requires_resume'):
        run(executor)
    executor.stop()
    with pytest.raises(ValueError, match='invalid_resume_token'):
        run(executor, resume_token=first['resume_token'])
    assert executor.status()['status'] == 'cancelled'


def test_cancel_during_inflight_primitive_discards_late_success():
    entered, release = threading.Event(), threading.Event()
    def pending(plan, options):
        entered.set()
        assert release.wait(2)
        return outcome()
    primitive = Primitive(pending)
    executor = SequenceExecutionService(primitive, clock=Clock())
    collected = []
    thread = threading.Thread(target=lambda: collected.append(run(executor)))
    thread.start()
    assert entered.wait(1)
    with pytest.raises(ValueError, match='sequence_busy'):
        run(executor)
    executor.stop()
    release.set()
    thread.join(2)
    assert collected[0]['status'] == 'cancelled' and not collected[0]['verified']
    assert collected[0]['completed_step_ids'] == []


def test_telemetry_aggregates_invocation_counts_and_strips_primitive_tokens():
    def report(plan, options):
        options['event_callback']({'phase': 'motion_completed', 'calls': 2, 'actions': 1,
                                   'resume_token': 'private', 'checkpoint': {'private': True}})
        return placed(calls=2)
    primitive = Primitive(outcome(calls=3), report)
    executor = SequenceExecutionService(primitive, clock=Clock())
    events = []
    result = run(executor, sequence(('pick', 'red', None), ('place', 'red', 'goal')),
                 event_callback=events.append)
    event = next(x for x in events if x['phase'] == 'motion_completed')
    assert event['calls'] == 5 and event['actions'] == 2 and event['step_id'] == 's2'
    assert 'private' not in str(events) and result['calls'] == 5


def test_exception_preserves_reported_budget_and_does_not_leak_exception():
    def broken(plan, options):
        options['event_callback']({'phase': 'jev_decision', 'calls': 2, 'actions': 0})
        raise ValueError('private provider contents')
    primitive = Primitive(broken)
    executor = SequenceExecutionService(primitive, clock=Clock())
    result = run(executor)
    assert result['calls'] == 2 and result['reason'] == 'primitive_execution_failed'
    assert 'private' not in str(result)


def test_readiness_and_reset_are_forwarded_without_claiming_validation():
    primitive = Primitive()
    executor = SequenceExecutionService(primitive, clock=Clock())
    assert not executor.validated and executor.experimental_simulation
    assert executor.readiness() == primitive.readiness()
    executor.reset()
    assert primitive.stops == 1 and primitive.resets == 1


def test_total_step_limit_includes_completed_prefix_after_tail_revision():
    primitive = Primitive(outcome())
    executor = SequenceExecutionService(primitive, clock=Clock(), max_steps=2)
    initial = sequence(('pick', 'red', None), ('place', 'red', 'goal'))
    first = run(executor, initial, cycle_max_calls=1)
    revised = sequence(('place', 'red', 'on:blue'), ('pick', 'blue', None))
    with pytest.raises(ValueError, match='step_budget_exhausted'):
        run(executor, revised, resume_token=first['resume_token'])


def test_event_failure_after_completed_step_stops_and_preserves_accounting():
    primitive = Primitive(outcome())
    executor = SequenceExecutionService(primitive, clock=Clock())
    def broken(detail):
        raise RuntimeError('private callback contents')
    result = run(executor, event_callback=broken)
    assert result['status'] == 'failed' and not result['verified']
    assert result['reason'] == 'sequence_execution_failed'
    assert result['calls'] == 1 and result['completed_step_ids'] == ['s1']
    assert primitive.stops == 1 and result['resume_token'] is None
    assert 'private' not in str(result)


def test_stop_signals_shared_cancellation_event():
    entered, release = threading.Event(), threading.Event()
    cancel = threading.Event()
    def pending(plan, options):
        entered.set()
        assert release.wait(2)
        assert options['cancel_event'].is_set()
        return outcome()
    primitive = Primitive(pending)
    executor = SequenceExecutionService(primitive, clock=Clock())
    results = []
    worker = threading.Thread(target=lambda: results.append(run(executor, cancel_event=cancel)))
    worker.start()
    assert entered.wait(1)
    executor.stop()
    assert cancel.is_set()
    release.set()
    worker.join(2)
    assert results[0]['status'] == 'cancelled'


def test_cancelled_request_does_not_invoke_primitive():
    primitive = Primitive()
    executor = SequenceExecutionService(primitive, clock=Clock())
    cancel = threading.Event()
    cancel.set()
    result = run(executor, cancel_event=cancel)
    assert result['status'] == 'cancelled' and not primitive.requests


def test_expired_original_deadline_invalidates_resume_without_new_motion():
    clock = Clock()
    primitive = Primitive(outcome('yielded', token='p1'))
    executor = SequenceExecutionService(primitive, clock=clock)
    first = run(executor, deadline=10)
    clock.now = 11
    result = run(executor, deadline=100, resume_token=first['resume_token'])
    assert result['status'] == 'failed' and result['reason'] == 'time_budget'
    assert len(primitive.requests) == 1 and result['resume_token'] is None


def test_capabilities_and_held_proof_are_forwarded_without_inference():
    primitive = Primitive()
    primitive.held = 'red'
    primitive.supported_targets = ['goal']
    primitive.held_evidence = {'epoch': 7, 'object_id': 'red'}
    executor = SequenceExecutionService(primitive, clock=Clock())
    assert executor.supports_sequences and executor.supports_cycles
    assert executor.held == 'red'
    assert executor.planning_held_object() is None
    primitive.planning_held_object = lambda: 'red'
    assert executor.planning_held_object() == 'red'
    assert executor.held_certificate == primitive.held_evidence
    executor.held_certificate['epoch'] = 8
    assert primitive.held_evidence['epoch'] == 7
    executor.supported_targets.append('on:red')
    assert primitive.supported_targets == ['goal']
    primitive.supported_targets = ['goal', 'on:red', 'on:blue']
    assert executor.supported_targets == primitive.supported_targets


def test_pause_forwards_without_invalidating_checkpoint():
    primitive = Primitive(outcome('yielded', token='p1'), outcome())
    paused = []
    primitive.pause = lambda: paused.append(True)
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor)
    executor.pause()
    assert paused == [True] and primitive.stops == 0
    assert run(executor, resume_token=first['resume_token'])['verified']


def test_recent_micro_actions_are_preserved_without_duplicating_resume_history():
    opened = {'action': 'open', 'applied': True}
    closed = {'action': 'close', 'applied': True}
    lifted = {'action': 'z_plus', 'applied': False}
    released = {'action': 'open', 'applied': True}
    primitive = Primitive(
        {**outcome('yielded', token='p1'), 'recent_actions': [opened, closed]},
        {**outcome(), 'recent_actions': [opened, closed, lifted]},
        {**placed(), 'recent_actions': [released]})
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor, sequence(('pick', 'red', None), ('place', 'red', 'goal')))
    assert first['recent_actions'] == [opened, closed]
    last = run(executor, resume_token=first['resume_token'])
    assert last['recent_actions'] == [opened, closed, lifted, released]
    last['recent_actions'][0]['applied'] = False
    assert executor.status()['recent_actions'][0]['applied'] is True


def test_recent_actions_remain_bounded_across_step_boundaries():
    actions = [{'action': 'hold', 'applied': True}] * 8
    released = {'action': 'open', 'applied': True}
    primitive = Primitive({**outcome(), 'recent_actions': actions},
                          {**placed(), 'recent_actions': [released]})
    executor = SequenceExecutionService(primitive, clock=Clock())
    result = run(executor, sequence(('pick', 'red', None), ('place', 'red', 'goal')))
    assert result['recent_actions'] == actions[1:] + [released]


@pytest.mark.parametrize('actions', [None, [{'action': 'teleport', 'applied': True}],
                                   [{'action': 'hold', 'applied': 'true'}],
                                   [{'action': 'hold', 'applied': True, 'resume_token': 'private'}]])
def test_invalid_micro_action_evidence_fails_closed(actions):
    primitive = Primitive({**outcome(), 'recent_actions': actions})
    executor = SequenceExecutionService(primitive, clock=Clock())
    result = run(executor)
    assert not result['verified'] and result['reason'] == 'invalid_primitive_outcome'


def test_omitted_local_cycle_limit_defaults_to_global_remaining_budget():
    primitive = Primitive(outcome())
    executor = SequenceExecutionService(primitive, clock=Clock())
    result = executor.bounded_execute(single(), max_calls=3, deadline=100)
    assert result['verified'] and primitive.requests[0][1]['cycle_max_calls'] == 3


def test_current_plan_tracks_owned_tail_revision_and_stable_completed_ids():
    primitive = Primitive(outcome(), placed(status='yielded', token='p1'), placed())
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor, sequence(), cycle_max_calls=1)
    revised = sequence(('place', 'red', 'on:blue'))
    second = run(executor, revised, resume_token=first['resume_token'])
    current = second['current_plan']
    assert set(current) == {'action', 'object_id', 'target_id', 'message', 'steps'}
    assert current['action'] == 'sequence' and current['object_id'] is current['target_id'] is None
    assert current['steps'] == [
        {'step_id': 's1', 'action': 'pick', 'object_id': 'red', 'target_id': None},
        {'step_id': 's2', 'action': 'place', 'object_id': 'red', 'target_id': 'on:blue'}]
    assert second['completed_step_ids'] == ['s1'] and second['active_step_id'] == 's2'
    assert len(current['steps']) <= 6 and len(current['message']) <= 300
    last = run(executor, single('place', 'red', 'on:blue'), resume_token=second['resume_token'])
    assert last['current_plan'] == current and last['completed_step_ids'] == ['s1', 's2']
    assert first['current_plan']['steps'] == sequence()['steps']
    current['steps'][1]['target_id'] = 'goal'
    assert executor.status()['current_plan']['steps'][1]['target_id'] == 'on:blue'


def test_singleton_current_plan_uses_canonical_server_owned_step_ids():
    primitive = Primitive(outcome('yielded', token='p1'), outcome())
    executor = SequenceExecutionService(primitive, clock=Clock())
    first = run(executor, {**single(), 'step_id': 's6'})
    assert first['active_step_id'] == 's1'
    assert first['current_plan']['steps'] == [
        {'step_id': 's1', 'action': 'pick', 'object_id': 'red', 'target_id': None}]
    last = run(executor, {**single(), 'step_id': 's3'}, resume_token=first['resume_token'])
    assert last['current_plan'] == first['current_plan']
    assert last['completed_step_ids'] == ['s1']
