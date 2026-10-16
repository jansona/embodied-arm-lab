"""Server-owned, bounded continuation of image-verified symbolic primitives.

The primitive executor owns perception, motor phases and held-object evidence.
This wrapper owns order, cumulative budgets and opaque continuation tokens. It
never reads simulator state, calls a model, or substitutes success for evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import threading
import time
import uuid


HOLDING_STATES = frozenset(('empty', 'held_verified', 'release_pending', 'unknown'))
MICRO_ACTIONS = frozenset(('x_plus', 'x_minus', 'y_plus', 'y_minus', 'z_plus',
                           'z_minus', 'open', 'close', 'hold'))


def _valid_recent_actions(actions):
    return (isinstance(actions, list) and len(actions) <= 8
            and all(isinstance(item, dict) and set(item) == {'action', 'applied'}
                    and isinstance(item['action'], str) and item['action'] in MICRO_ACTIONS
                    and type(item['applied']) is bool for item in actions))


def _intent(step):
    return (step['action'], step['object_id'], step['target_id'])


def _steps(plan):
    """Structural validation only; do not manufacture a scene for brain validation."""
    if not isinstance(plan, dict):
        raise ValueError('invalid_sequence_plan')
    if plan.get('action') == 'sequence':
        if (set(plan) != {'action', 'object_id', 'target_id', 'message', 'steps'}
                or plan['object_id'] is not None or plan['target_id'] is not None):
            raise ValueError('invalid_sequence_plan')
        raw = plan['steps']
        if not isinstance(raw, list) or not 1 <= len(raw) <= 6:
            raise ValueError('invalid_sequence_plan')
    else:
        if (set(plan) - {'step_id', 'action', 'object_id', 'target_id', 'message'}
                or not {'action', 'object_id', 'target_id'} <= set(plan)):
            raise ValueError('invalid_sequence_plan')
        raw = [{k: v for k, v in plan.items() if k != 'message'}]
        raw[0].setdefault('step_id', 's1')
    if 'message' in plan and (not isinstance(plan['message'], str)
                              or len(plan['message']) > 300):
        raise ValueError('invalid_sequence_plan')
    out = []
    for index, step in enumerate(raw):
        if not isinstance(step, dict) or set(step) != {'step_id', 'action', 'object_id', 'target_id'}:
            raise ValueError('invalid_sequence_step')
        action, obj, target = step['action'], step['object_id'], step['target_id']
        sid = step['step_id']
        if not isinstance(sid, str) or sid not in ('s1', 's2', 's3', 's4', 's5', 's6'):
            raise ValueError('invalid_sequence_step')
        if plan.get('action') == 'sequence' and sid != f's{index + 1}':
            raise ValueError('invalid_sequence_step')
        if (not isinstance(action, str) or action not in ('pick', 'place')
                or not isinstance(obj, str) or obj not in ('red', 'blue')):
            raise ValueError('invalid_sequence_step')
        if action == 'pick' and target is not None:
            raise ValueError('unsupported_pick_target')
        if action == 'place' and (not isinstance(target, str)
                                 or target not in ('goal', 'on:red', 'on:blue')
                                 or target == 'on:' + obj):
            raise ValueError('invalid_sequence_target')
        out.append(deepcopy(step))
    return out


@dataclass
class _Execution:
    steps: list
    deadline: float
    call_cap: int
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    index: int = 0
    started: int = 0
    calls: int = 0
    actions: int = 0
    revision: int = 0
    active_started: bool = False
    primitive_token: str | None = None
    outer_token: str | None = None
    phase: str | None = None
    holding_state: str = 'unknown'
    records: list = field(default_factory=list)
    completed: list = field(default_factory=list)
    recent_actions: list = field(default_factory=list)
    primitive_actions_base: list = field(default_factory=list)


class SequenceExecutionService:
    supports_sequences = True
    supports_cycles = True

    def __init__(self, primitive_executor, *, clock=time.monotonic, max_steps=6):
        if type(max_steps) is not int or not 1 <= max_steps <= 6:
            raise ValueError('invalid_budget')
        self.primitive_executor = primitive_executor
        self.clock = clock
        self.max_steps = max_steps
        self._lock = threading.RLock()
        self._running = threading.Lock()
        self._generation = 0
        self._execution = None
        self._cancel_event = None
        self._last = {'status': 'idle', 'verified': False, 'calls': 0, 'actions': 0}

    @property
    def validated(self):
        return bool(getattr(self.primitive_executor, 'validated', False))

    @property
    def experimental_simulation(self):
        return bool(getattr(self.primitive_executor, 'experimental_simulation', False))

    @property
    def supported_targets(self):
        return deepcopy(getattr(self.primitive_executor, 'supported_targets', []))

    @property
    def held(self):
        return getattr(self.primitive_executor, 'held', None)

    @property
    def held_certificate(self):
        evidence = getattr(self.primitive_executor, 'held_certificate', None)
        if evidence is None:
            evidence = getattr(self.primitive_executor, 'held_evidence', None)
        return deepcopy(evidence)

    def planning_held_object(self):
        method = getattr(self.primitive_executor, 'planning_held_object', None)
        return method() if method is not None else None

    def pause(self):
        return self.primitive_executor.pause()

    def readiness(self):
        method = getattr(self.primitive_executor, 'readiness', None)
        if method is not None:
            return deepcopy(method())
        return {'available': False, 'reason': 'controller_readiness_unknown'}

    def status(self):
        with self._lock:
            return deepcopy(self._last)

    def stop(self):
        with self._lock:
            self._generation += 1
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._execution = None
            self._last.update(status='cancelled', verified=False, resume_token=None)
        self.primitive_executor.stop()

    cancel = stop

    def reset(self):
        self.stop()
        method = getattr(self.primitive_executor, 'reset', None)
        if method is not None:
            method()

    def _accept(self, plan, max_calls, deadline, resume_token):
        incoming = _steps(plan)
        with self._lock:
            current = self._execution
            if resume_token is None:
                if current is not None and current.outer_token is not None:
                    raise ValueError('pending_checkpoint_requires_resume')
                if len(incoming) > self.max_steps:
                    raise ValueError('step_budget_exhausted')
                # A new task owns its IDs even when a singleton arrives with a
                # step_id from another context. Resumes preserve existing IDs.
                incoming = [{**step, 'step_id': f's{i + 1}'}
                            for i, step in enumerate(incoming)]
                current = _Execution(incoming, deadline, max_calls)
                self._execution = current
            else:
                if (not isinstance(resume_token, str) or current is None
                        or not current.outer_token or resume_token != current.outer_token):
                    raise ValueError('invalid_resume_token')
                remaining = current.steps[current.index:]
                incoming_intents = [_intent(x) for x in incoming]
                remaining_intents = [_intent(x) for x in remaining]
                if current.active_started:
                    # A singleton continuation preserves the entire owned tail.
                    if (incoming_intents != remaining_intents
                            and incoming_intents != remaining_intents[:1]):
                        raise ValueError('active_intent_changed')
                elif incoming_intents != remaining_intents and incoming_intents != remaining_intents[:1]:
                    if plan.get('action') != 'sequence':
                        raise ValueError('tail_revision_requires_sequence')
                    if any(_intent(x) in [_intent(y) for y in current.completed] for x in incoming):
                        raise ValueError('completed_step_replay')
                    if current.started + len(incoming) > self.max_steps:
                        raise ValueError('step_budget_exhausted')
                    current.revision += 1
                    # Stable internal IDs are never replaced with Qwen's new s1.
                    revised = [{**x, 'step_id': f's{current.started + i + 1}'}
                               for i, x in enumerate(incoming)]
                    current.steps = current.steps[:current.index] + revised
                current.outer_token = None  # Consume only after validating the continuation.
                current.deadline = min(current.deadline, deadline)
                current.call_cap = min(current.call_cap, current.calls + max_calls)
            return current, self._generation

    def bounded_execute(self, plan, *, max_calls, deadline, cycle_max_calls=None,
                        cancel_event=None, event_callback=None, resume_token=None):
        if cycle_max_calls is None:
            cycle_max_calls = max_calls
        if (type(max_calls) is not int or not 1 <= max_calls <= 200
                or type(cycle_max_calls) is not int or not 1 <= cycle_max_calls <= 200
                or type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError('invalid_budget')
        if not self._running.acquire(blocking=False):
            raise ValueError('sequence_busy')
        cancel = cancel_event or threading.Event()
        invocation_calls = invocation_actions = 0
        current = None
        generation = None
        with self._lock:
            self._cancel_event = cancel
        try:
            current, generation = self._accept(plan, max_calls, deadline, resume_token)
            remaining_global = min(max_calls, current.call_cap - current.calls)
            cycle_limit = min(cycle_max_calls, remaining_global)

            def emit(detail):
                if event_callback is not None:
                    event_callback(detail)

            def result(status, reason=None, *, extra=None):
                token = uuid.uuid4().hex if status == 'yielded' else None
                active = (deepcopy(current.steps[current.index])
                          if current.index < len(current.steps) else None)
                outcome = {
                    'status': status, 'verified': status == 'completed', 'reason': reason,
                    'calls': invocation_calls, 'actions': invocation_actions,
                    'cumulative_calls': current.calls, 'cumulative_actions': current.actions,
                    'completed_step_ids': [x['step_id'] for x in current.completed],
                    'completed_steps': deepcopy(current.completed), 'active_step': active,
                    'current_plan': {'action': 'sequence', 'object_id': None, 'target_id': None,
                                     'message': 'Use fresh visual evidence for the symbolic task.',
                                     'steps': deepcopy(current.steps)},
                    'active_step_id': active['step_id'] if active else None,
                    'failed_step_id': active['step_id'] if active and status == 'failed' else None,
                    'phase': current.phase, 'holding_state': current.holding_state,
                    'recent_actions': deepcopy(current.recent_actions),
                    'resume_token': token, 'steps': deepcopy(current.records),
                    'revision': current.revision, 'source': 'primitive_visual_verdicts',
                }
                if extra:
                    outcome.update(extra)
                with self._lock:
                    if generation == self._generation:
                        current.outer_token = token
                        self._last = deepcopy(outcome)
                        if status != 'yielded':
                            self._execution = None
                return outcome

            def cancelled():
                return cancel.is_set() or generation != self._generation

            while current.index < len(current.steps):
                if cancelled():
                    self.primitive_executor.stop()
                    return result('cancelled', 'cancelled')
                if self.clock() >= current.deadline:
                    self.primitive_executor.stop()
                    return result('failed', 'time_budget')
                available = min(remaining_global - invocation_calls,
                                current.call_cap - current.calls)
                local = cycle_limit - invocation_calls
                if available <= 0:
                    self.primitive_executor.stop()
                    return result('failed', 'call_budget_exhausted')
                if local <= 0:
                    current.phase = 'step_boundary' if not current.active_started else current.phase
                    return result('yielded', 'cycle_budget')
                step = current.steps[current.index]
                if not current.active_started:
                    if current.started >= self.max_steps:
                        self.primitive_executor.stop()
                        return result('failed', 'step_budget_exhausted')
                    current.started += 1
                    current.active_started = True
                    current.primitive_actions_base = deepcopy(current.recent_actions)
                child_calls = child_actions = 0

                def on_event(detail):
                    nonlocal child_calls, child_actions
                    if cancelled() or self.clock() >= current.deadline:
                        return False
                    if not isinstance(detail, dict):
                        return False
                    forwarded = deepcopy(detail)
                    calls, actions = detail.get('calls'), detail.get('actions')
                    if type(calls) is int and 0 <= calls <= min(available, local):
                        child_calls = max(child_calls, calls)
                        forwarded['calls'] = invocation_calls + calls
                    if type(actions) is int and actions >= 0:
                        child_actions = max(child_actions, actions)
                        forwarded['actions'] = invocation_actions + actions
                    # Primitive tokens/checkpoints are never model-facing telemetry.
                    forwarded.pop('resume_token', None)
                    forwarded.pop('checkpoint', None)
                    forwarded.update(step_id=step['step_id'], step_index=current.index,
                                     revision=current.revision)
                    emit(forwarded)
                    return True

                try:
                    primitive = self.primitive_executor.bounded_execute(
                        deepcopy(step), max_calls=available, deadline=current.deadline,
                        cycle_max_calls=min(local, available), cancel_event=cancel,
                        event_callback=on_event, resume_token=current.primitive_token)
                except Exception:
                    invocation_calls += child_calls
                    invocation_actions += child_actions
                    current.calls += child_calls
                    current.actions += child_actions
                    self.primitive_executor.stop()
                    return result('cancelled' if cancelled() else 'failed',
                                  'cancelled' if cancelled() else 'primitive_execution_failed')
                if (not isinstance(primitive, dict)
                        or type(primitive.get('calls')) is not int
                        or not child_calls <= primitive['calls'] <= min(available, local)
                        or type(primitive.get('actions')) is not int
                        or primitive['actions'] < child_actions
                        or primitive.get('status') not in ('completed', 'yielded', 'failed', 'cancelled')
                        or type(primitive.get('verified')) is not bool
                        or not isinstance(primitive.get('holding_state'), str)
                        or primitive['holding_state'] not in HOLDING_STATES
                        or not isinstance(primitive.get('phase'), str)
                        or not _valid_recent_actions(primitive.get('recent_actions', []))):
                    invocation_calls += child_calls
                    invocation_actions += child_actions
                    current.calls += child_calls
                    current.actions += child_actions
                    self.primitive_executor.stop()
                    return result('failed', 'invalid_primitive_outcome')
                used, applied = primitive['calls'], primitive['actions']
                invocation_calls += used
                invocation_actions += applied
                current.calls += used
                current.actions += applied
                current.phase = primitive['phase']
                current.holding_state = primitive['holding_state']
                if 'recent_actions' in primitive:
                    # Each primitive owns a cumulative, bounded history across
                    # resumes. Replace its segment instead of duplicating it.
                    current.recent_actions = deepcopy(
                        (current.primitive_actions_base + primitive['recent_actions'])[-8:])
                public = {k: deepcopy(v) for k, v in primitive.items()
                          if k not in ('resume_token', 'checkpoint')}
                current.records.append({'step_id': step['step_id'], 'revision': current.revision,
                                        'outcome': public})
                if cancelled():
                    self.primitive_executor.stop()
                    return result('cancelled', 'cancelled')
                if self.clock() >= current.deadline:
                    self.primitive_executor.stop()
                    return result('failed', 'time_budget')
                status = primitive['status']
                if (status == 'completed' and primitive['verified']
                        and primitive['holding_state'] != ('held_verified' if step['action'] == 'pick' else 'empty')):
                    self.primitive_executor.stop()
                    return result('failed', 'inconsistent_holding_verdict')
                if status == 'completed' and primitive['verified']:
                    current.completed.append(deepcopy(step))
                    current.index += 1
                    current.active_started = False
                    current.primitive_token = None
                    emit({'phase': 'symbolic_step_verified', 'step_id': step['step_id'],
                          'calls': invocation_calls, 'actions': invocation_actions})
                    continue
                if status == 'yielded' and not primitive['verified']:
                    token = primitive.get('resume_token')
                    if not isinstance(token, str) or not token or used == 0:
                        self.primitive_executor.stop()
                        return result('failed', 'invalid_primitive_checkpoint')
                    if current.calls >= current.call_cap or invocation_calls >= remaining_global:
                        self.primitive_executor.stop()
                        return result('failed', 'call_budget_exhausted')
                    current.primitive_token = token
                    return result('yielded', 'cycle_budget')
                self.primitive_executor.stop()
                return result('cancelled' if status == 'cancelled' else 'failed',
                              primitive.get('reason') or 'execution_unverified')
            if cancelled():
                self.primitive_executor.stop()
                return result('cancelled', 'cancelled')
            return result('completed', 'all_steps_visually_verified')
        except Exception:
            # Invalid input/token attempts leave a legitimate paused checkpoint
            # intact. Failures after accepting a continuation must stop it.
            if current is None:
                raise
            self.primitive_executor.stop()
            return result('cancelled' if cancelled() else 'failed',
                          'cancelled' if cancelled() else 'sequence_execution_failed')
        finally:
            with self._lock:
                if self._cancel_event is cancel:
                    self._cancel_event = None
            self._running.release()
