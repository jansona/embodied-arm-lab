"""One bounded conversation/task lifecycle, independent of HTTP and robot physics.

Providers are injected. This module never reads credentials, simulator object
poses or images, and never substitutes rules for model-selected actions.
"""
from copy import deepcopy
import threading
import time
import uuid
UNKNOWN_SCENE = {'observation_status': 'unavailable', 'blocks': {}}
TERMINAL = {'completed', 'blocked', 'failed', 'cancelled'}

class TaskService:

    def __init__(self, *, scene_source, brain, executor, clock=time.monotonic):
        self.source = scene_source
        self.brain = brain
        self.executor = executor
        self.clock = clock
        self.lock = threading.RLock()
        self.generation = 0
        self.thread = None
        self.cancel_event = threading.Event()
        self.history = []
        self.record = {'id': None, 'state': 'idle', 'events': [], 'plan': None, 'error': None, 'execute': False}

    def execution_readiness(self):
        if not self.executor or not getattr(self.executor,'validated',False):
            return {'available':False,'reason':'visual_execution_not_validated'}
        readiness=getattr(self.executor,'readiness',None)
        return readiness() if readiness else {'available':True,'reason':None}

    def status(self):
        with self.lock:
            ready=self.execution_readiness()
            return deepcopy({**self.record, 'history': self.history, 'pending': bool(self.thread and self.thread.is_alive()), 'execution_available': ready['available'], 'unavailable_reason': ready['reason']})

    def _event(self, generation, state, detail=None):
        with self.lock:
            if generation != self.generation or self.cancel_event.is_set():
                return False
            self.record['state'] = state
            self.record['events'].append({'state': state, 'at': self.clock(), 'detail': detail})
            self.record['events'] = self.record['events'][-100:]
            return True

    def submit(self, text, *, execute=True):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError('invalid_text')
        if type(execute) is not bool:
            raise ValueError('invalid_mode')
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('task_busy')
            ready = self.brain.status()
            if not ready.get('available', ready.get('configured', False)):
                raise ValueError('brain_unavailable')
            if execute and not self.execution_readiness()['available']:
                raise ValueError('execution_unavailable')
            self.generation += 1
            generation = self.generation
            self.cancel_event = threading.Event()
            self.record = {'id': uuid.uuid4().hex, 'state': 'queued', 'events': [], 'plan': None, 'error': None, 'execute': execute}
            history = deepcopy(self.history[-6:])
            self.history.append({'role': 'user', 'content': text.strip()})
            self.history = self.history[-20:]
            self.thread = threading.Thread(target=self._run, args=(generation, text.strip(), execute, history, self.cancel_event), daemon=True)
            self.thread.start()
            return self.status()

    def _run(self, generation, text, execute, history, cancel):
        phase = 'planning'
        try:
            scene = UNKNOWN_SCENE
            if execute:
                phase = 'observing'
                if not self._event(generation, phase):
                    return
                self.source.refresh()
                if cancel.is_set():
                    return
                scene = self.source.brain_scene()
            phase = 'planning'
            if not self._event(generation, phase):
                return
            result = self.brain.interpret(text, scene, history)
            if cancel.is_set():
                return
            if not result.accepted or result.plan is None:
                raise ValueError('brain_rejected')
            plan = result.plan.to_dict()
            if not execute and plan['action'] not in ('status', 'clarify', 'stop'):
                raise ValueError('unexpected_motion_plan')
            with self.lock:
                if generation != self.generation:
                    return
                self.record['plan'] = plan
                self.record['provenance'] = getattr(result, 'provenance', 'unknown')
                self.history.append({'role': 'assistant', 'content': plan['message']})
            if plan['action'] in ('status', 'clarify', 'stop'):
                self._event(generation, 'completed', 'conversation_only')
                return
            if not execute:
                raise ValueError('unexpected_motion_plan')
            if plan['action'] not in ('pick', 'place'):
                raise ValueError('unsupported_task')
            self.source.snapshot()
            phase = 'executing'
            if not self._event(generation, phase):
                return
            outcome = self.executor.execute(plan, cancel_event=cancel, event_callback=lambda detail: self._event(generation, 'executing', detail))
            if cancel.is_set():
                return
            if isinstance(outcome,dict):
                with self.lock:
                    if generation==self.generation:self.record['outcome']=deepcopy(outcome)
            if not isinstance(outcome, dict) or outcome.get('verified') is not True:
                raise ValueError('execution_unverified')
            with self.lock:
                if generation == self.generation:
                    self.record['outcome'] = deepcopy(outcome)
                    self.history.append({'role': 'assistant', 'content': f"执行反馈：{plan['action']} {plan.get('object_id')} 已通过图像验证。"})
                    self.history = self.history[-20:]
            self._event(generation, 'completed', 'visually_verified')
        except Exception as exc:
            if cancel.is_set():
                return
            allowed = {'brain_rejected', 'unexpected_motion_plan', 'unsupported_task', 'execution_unverified'}
            reason = str(exc) if str(exc) in allowed else {'observing': 'observation_unavailable', 'planning': 'planning_failed', 'executing': 'execution_failed'}[phase]
            with self.lock:
                if generation == self.generation:
                    self.record['error'] = reason
            self._event(generation, 'blocked' if phase == 'observing' or reason == 'unsupported_task' else 'failed', reason)
        finally:
            if execute and self.executor is not None:
                self.executor.stop()

    def reset(self):
        self.stop()
        with self.lock:self.history.clear()
        if self.executor is not None and hasattr(self.executor,'reset'):self.executor.reset()
        return self.status()

    def stop(self):
        with self.lock:
            self.cancel_event.set()
            self.generation += 1
            self.record['state'] = 'cancelled'
            self.record['error'] = None
        for dependency in (self.brain, self.source, self.executor):
            method = getattr(dependency, 'cancel', None) or getattr(dependency, 'stop', None)
            if method:
                method()
        return self.status()
