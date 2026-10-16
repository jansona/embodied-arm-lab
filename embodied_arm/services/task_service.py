"""One bounded conversation/task lifecycle, independent of HTTP and robot physics.

Providers are injected. This module never reads credentials, simulator object
poses or images, and never substitutes rules for model-selected actions.
"""
from copy import deepcopy
import threading
import time
import uuid
UNKNOWN_SCENE = {'observation_status': 'unavailable', 'blocks': {}}
TERMINAL = {'completed', 'plan_preview', 'blocked', 'failed', 'cancelled'}
SMALL_TALK = frozenset(('你好','您好','嗨','谢谢','感谢','hello','hi','thanks','thank you'))
OBSERVATION_ERRORS = {'vision_unavailable','network_error','timeout','expired','call_budget_exhausted',
    'landmark_uncertain','occluded','occluded_or_missing_feature','uncertain_detection','low_confidence',
    'unavailable','scene_stale','scene_changed_during_vision','invalid_or_stale_frame','stale_frame','observation_unavailable'}

class TaskService:

    def __init__(self, *, scene_source, brain, executor, clock=time.monotonic, max_replans=1, max_controller_calls=120, max_seconds=300):
        if type(max_replans) is not int or not 0 <= max_replans <= 1 or type(max_controller_calls) is not int or not 1 <= max_controller_calls <= 200 or not 1 <= max_seconds <= 420:
            raise ValueError("invalid_budget")
        self.max_replans=max_replans
        self.max_controller_calls=max_controller_calls
        self.max_seconds=max_seconds
        self.source = scene_source
        self.brain = brain
        self.executor = executor
        self.clock = clock
        self.lock = threading.RLock()
        self.generation = 0
        self.thread = None
        self.cancel_event = threading.Event()
        self.history = []
        self.plan_binding=None
        self.record = {'id': None, 'state': 'idle', 'events': [], 'plan': None, 'error': None, 'execute': False}

    def execution_readiness(self):
        if not self.executor or not (getattr(self.executor,'validated',False) or getattr(self.executor,'experimental_simulation',False)):
            return {'available':False,'reason':'visual_execution_not_validated'}
        readiness=getattr(self.executor,'readiness',None)
        return readiness() if readiness else {'available':True,'reason':None}

    def status(self):
        with self.lock:
            ready=self.execution_readiness()
            plan_fresh=False
            if self.plan_binding is not None:
                try:plan_fresh=self._binding(self.source.snapshot())==self.plan_binding
                except ValueError:pass
            return deepcopy({**self.record, 'plan_fresh':plan_fresh, 'history': self.history, 'pending': bool(self.thread and self.thread.is_alive()), 'execution_validation':'experimental_simulation' if getattr(self.executor,'experimental_simulation',False) else 'validated' if getattr(self.executor,'validated',False) else 'unvalidated', 'execution_available': ready['available'], 'unavailable_reason': ready['reason']})

    @staticmethod
    def _binding(snapshot):
        scene=snapshot.get('scene',{});robot=snapshot.get('robot',{})
        return {'frame_ids':scene.get('frame_ids'),'epoch':robot.get('epoch'),'actions':robot.get('actions')}

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
            self.plan_binding=None
            generation = self.generation
            self.cancel_event = threading.Event()
            self.record = {'id': uuid.uuid4().hex, 'state': 'queued', 'events': [], 'plan': None, 'error': None, 'execute': execute, 'motion_executed':False, 'replans':0, 'controller_calls':0}
            history = deepcopy(self.history[-6:])
            self.history.append({'role': 'user', 'content': text.strip()})
            self.history = self.history[-20:]
            self.thread = threading.Thread(target=self._run, args=(generation, text.strip(), execute, history, self.cancel_event), daemon=True)
            self.thread.start()
            return self.status()

    def _run(self, generation, text, execute, history, cancel):
        phase = 'planning'
        deadline=self.clock()+self.max_seconds
        try:
            scene = UNKNOWN_SCENE
            ensure=getattr(self.source,'ensure_brain_scene',None)
            smalltalk=text.strip().lower().strip('!！。.?？') in SMALL_TALK
            if execute or ensure is not None and not smalltalk:
                phase = 'observing'
                if not self._event(generation, phase):return
                if ensure is not None:scene=ensure()
                else:
                    self.source.refresh()
                    scene=self.source.brain_scene()
                if cancel.is_set():return
            elif ensure is not None:
                try:scene=self.source.brain_scene()
                except ValueError:pass
            phase = 'planning'
            if not self._event(generation, phase):
                return
            if self.clock() >= deadline:raise ValueError('time_budget')
            input_binding=self._binding(self.source.snapshot()) if scene!=UNKNOWN_SCENE else None
            result = self.brain.interpret(text, scene, history)
            if cancel.is_set():
                return
            if self.clock() >= deadline:raise ValueError('time_budget')
            if not result.accepted or result.plan is None:
                raise ValueError('brain_rejected')
            if scene!=UNKNOWN_SCENE:
                try:binding=self._binding(self.source.snapshot())
                except ValueError:raise ValueError('scene_stale') from None
                if binding!=input_binding:raise ValueError('scene_stale')
            plan = result.plan.to_dict()
            if not execute and scene==UNKNOWN_SCENE and plan['action'] not in ('status', 'clarify', 'stop'):
                raise ValueError('unexpected_motion_plan')
            with self.lock:
                if generation != self.generation:
                    return
                self.record['plan'] = plan
                self.plan_binding=binding if scene!=UNKNOWN_SCENE else None
                self.record['provenance'] = getattr(result, 'provenance', 'unknown')
                self.history.append({'role': 'assistant', 'content': plan['message']})
            if plan['action'] in ('status', 'clarify', 'stop'):
                self._event(generation, 'completed', 'conversation_only')
                return
            if not execute:
                self._event(generation,'plan_preview','execution_not_requested')
                return
            if plan['action'] not in ('pick', 'place'):
                raise ValueError('unsupported_task')
            self.source.snapshot()
            phase = 'executing'
            if not self._event(generation, phase):
                return
            total_calls=0
            total_actions=0
            repairs=0
            bounded=getattr(self.executor,'bounded_execute',None)
            while True:
                if cancel.is_set():return
                if self.clock() >= deadline:raise ValueError('time_budget')
                remaining=self.max_controller_calls-total_calls
                if remaining <= 0:raise ValueError('call_budget_exhausted')
                def execution_event(detail, base_calls=total_calls, base_actions=total_actions):
                    forwarded=deepcopy(detail)
                    if isinstance(forwarded,dict):
                        observed_calls=forwarded.get('calls');observed_actions=forwarded.get('actions')
                        if type(observed_calls) is int and 0 <= observed_calls <= remaining:
                            forwarded['calls']=base_calls+observed_calls
                            with self.lock:
                                if generation==self.generation and not cancel.is_set():
                                    self.record['controller_calls']=forwarded['calls']
                        if type(observed_actions) is int and observed_actions >= 0:
                            forwarded['actions']=base_actions+observed_actions
                    return self._event(generation,'executing',forwarded)
                kwargs={'cancel_event':cancel, 'event_callback':execution_event}
                outcome=bounded(plan,max_calls=remaining,deadline=deadline,**kwargs) if bounded else self.executor.execute(plan,**kwargs)
                if cancel.is_set():return
                if not isinstance(outcome,dict):raise ValueError('execution_unverified')
                if bounded and not {'calls','actions'} <= set(outcome):raise ValueError('invalid_execution_budget')
                used=outcome.get('calls',0); applied=outcome.get('actions',0)
                if type(used) is not int or not 0 <= used <= remaining or type(applied) is not int or applied < 0:
                    raise ValueError('invalid_execution_budget')
                total_calls+=used;total_actions+=applied
                with self.lock:
                    if generation != self.generation:return
                    self.record['controller_calls']=total_calls
                    self.record['outcome']=deepcopy(outcome)
                    self.record['motion_executed']=bool(total_actions)
                if self.clock() >= deadline:raise ValueError('time_budget')
                if outcome.get('verified') is True:break
                recoverable=(bounded and outcome.get('recoverable') is True and outcome.get('safe_to_replan') is True
                             and outcome.get('reason')=='scene_changed' and repairs < self.max_replans
                             and total_calls < self.max_controller_calls and self.clock() < deadline)
                if not recoverable:break
                phase='replanning'
                if not self._event(generation,phase,{'attempt':repairs+1,'reason':'scene_changed'}):return
                ensure=getattr(self.source,'ensure_brain_scene',None)
                if ensure:scene=ensure()
                else:self.source.refresh();scene=self.source.brain_scene()
                if cancel.is_set():return
                if self.clock() >= deadline:raise ValueError('time_budget')
                repair_binding=self._binding(self.source.snapshot())
                result=self.brain.replan(text,scene,plan,{'completed_step_ids':[],'failed_step_id':'s1','reason':'scene_changed'},history)
                repairs+=1
                if cancel.is_set():return
                if self.clock() >= deadline:raise ValueError('time_budget')
                if not result.accepted or result.plan is None:raise ValueError('brain_rejected')
                if self._binding(self.source.snapshot()) != repair_binding:raise ValueError('scene_stale')
                plan=result.plan.to_dict()
                if plan['action'] not in ('pick','place'):raise ValueError('unsupported_task')
                with self.lock:
                    if generation != self.generation:return
                    self.record['replans']=repairs
                    self.record['plan']=plan
                    self.plan_binding=repair_binding
                phase='executing'
                if not self._event(generation,phase,{'attempt':repairs+1}):return
            outcome={**outcome,'calls':total_calls,'actions':total_actions,'replans':repairs}
            if isinstance(outcome,dict):
                with self.lock:
                    if generation==self.generation:
                        self.record['outcome']=deepcopy(outcome)
                        self.record['motion_executed']=bool(outcome.get('actions',0))
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
            allowed = {'brain_rejected', 'unexpected_motion_plan', 'unsupported_task', 'execution_unverified','time_budget','invalid_execution_budget'} | OBSERVATION_ERRORS
            reason = str(exc) if str(exc) in allowed else {'observing': 'observation_unavailable', 'planning': 'planning_failed', 'executing': 'execution_failed', 'replanning':'planning_failed'}[phase]
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
