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
    'rgb_face_refinement_rejected','triangulation_rejected','frame_mismatch','invalid_response','vision_rejected','rgb_tracking_rejected',
    'unavailable','scene_stale','scene_changed_during_vision','invalid_or_stale_frame','stale_frame','observation_unavailable'}

class TaskService:

    def __init__(self, *, scene_source, brain, executor, clock=time.monotonic, max_replans=1, max_controller_calls=120, max_seconds=300, cycle_max_calls=16, max_cycle_reviews=8):
        if type(max_replans) is not int or not 0 <= max_replans <= 1 or type(max_controller_calls) is not int or not 1 <= max_controller_calls <= 200 or not 1 <= max_seconds <= 420:
            raise ValueError("invalid_budget")
        if type(cycle_max_calls) is not int or not 1<=cycle_max_calls<=200 or type(max_cycle_reviews) is not int or not 0<=max_cycle_reviews<=8:raise ValueError("invalid_budget")
        self.cycle_max_calls=cycle_max_calls
        self.max_cycle_reviews=max_cycle_reviews
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
        self.runtime_notices = []
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
            return deepcopy({**self.record, 'plan_fresh':plan_fresh, 'history': self.history, 'runtime_notices':self.runtime_notices, 'pending': bool(self.thread and self.thread.is_alive()), 'execution_validation':'experimental_simulation' if getattr(self.executor,'experimental_simulation',False) else 'validated' if getattr(self.executor,'validated',False) else 'unvalidated', 'execution_available': ready['available'], 'unavailable_reason': ready['reason']})

    def _runtime_scene(self, scene, execute):
        if not getattr(self.executor,'experimental_simulation',False):return scene
        scene=deepcopy(scene)
        robot_state=getattr(self.executor,'robot_state',None)
        if robot_state:
            value=robot_state()
            if value is not None:
                from embodied_arm.providers.brain import validate_robot_state
                scene['robot_state']=validate_robot_state(value)
        scene['runtime_capabilities']={'robot_name':'SO-ARM101','execution_requested':execute,'experimental':True,
            'supported_actions':['pick','place']+(['sequence'] if getattr(self.executor,'supports_sequences',False) else []),
            'supported_targets':getattr(self.executor,'supported_targets',['goal']),
            'controller':'jev','completion_verification':'temporal_rgb'}
        if scene.get('observation_status')!='unavailable':
            scene['cube_edge_m']=.02;scene['perception_source']='rgb_only'
            held=self.executor.planning_held_object() if hasattr(self.executor,'planning_held_object') else getattr(self.executor,'held',None)
            scene.pop('held_object',None)
            if held in ('red','blue'):scene['held_object']=held
        return scene

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
            self.record = {'id': uuid.uuid4().hex, 'state': 'queued', 'events': [], 'plan': None, 'error': None, 'execute': execute, 'motion_executed':False, 'replans':0, 'controller_calls':0,'controller_actions':0,'cycle_reviews':0,'cycle_index':0,'cycle_max_calls':self.cycle_max_calls,'max_controller_calls':self.max_controller_calls}
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
            if scene.get('observation_status')!='unavailable':
                reconcile=getattr(self.executor,'reconcile_observation',None)
                if reconcile:
                    if cancel.is_set():return
                    holding=reconcile()
                    if cancel.is_set():return
                    if isinstance(holding,dict):
                        public_holding={key:holding.get(key) for key in ('verified','reason','holding_state')}
                        if not self._event(generation,'observing',{'phase':'holding_reacquisition','holding':public_holding}):return
            scene=self._runtime_scene(scene,execute)
            known_scene=scene.get('observation_status')!='unavailable'
            phase = 'planning'
            if not self._event(generation, phase):
                return
            if self.clock() >= deadline:raise ValueError('time_budget')
            input_binding=self._binding(self.source.snapshot()) if known_scene else None
            brain_started=self.clock()
            result = self.brain.interpret(text, scene, history)
            with self.lock:
                if generation==self.generation:self.record['brain_roundtrip_seconds']=max(0.,self.clock()-brain_started)
            if cancel.is_set():
                return
            if self.clock() >= deadline:raise ValueError('time_budget')
            if not result.accepted or result.plan is None:
                raise ValueError('brain_rejected')
            if known_scene:
                try:binding=self._binding(self.source.snapshot())
                except ValueError:raise ValueError('scene_stale') from None
                if binding!=input_binding:raise ValueError('scene_stale')
            plan = result.plan.to_dict()
            if not execute and not known_scene and plan['action'] not in ('status', 'clarify', 'stop'):
                raise ValueError('unexpected_motion_plan')
            with self.lock:
                if generation != self.generation:
                    return
                self.record['plan'] = plan
                self.plan_binding=binding if known_scene else None
                self.record['provenance'] = getattr(result, 'provenance', 'unknown')
                self.history.append({'role': 'assistant', 'content': plan['message']})
            if plan['action'] in ('status', 'clarify', 'stop'):
                self._event(generation, 'completed', 'conversation_only')
                return
            if not execute:
                self._event(generation,'plan_preview','execution_not_requested')
                return
            if plan['action'] not in (('pick','place','sequence') if getattr(self.executor,'supports_sequences',False) else ('pick','place')):
                raise ValueError('unsupported_task')
            self.source.snapshot()
            phase = 'executing'
            if not self._event(generation, phase):
                return
            total_calls=0
            total_actions=0
            repairs=0
            cycle_reviews=0
            resume_token=None
            original_plan=deepcopy(plan)
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
                                    forwarded['calls']=max(self.record.get('controller_calls',0),forwarded['calls'])
                                    self.record['controller_calls']=forwarded['calls']
                        if type(observed_actions) is int and observed_actions >= 0:
                            forwarded['actions']=base_actions+observed_actions
                            with self.lock:
                                if generation==self.generation and not cancel.is_set():
                                    forwarded['actions']=max(self.record.get('controller_actions',0),forwarded['actions'])
                                    self.record['controller_actions']=forwarded['actions']
                                    self.record['motion_executed']=bool(forwarded['actions'])
                    return self._event(generation,'executing',forwarded)
                kwargs={'cancel_event':cancel, 'event_callback':execution_event}
                if bounded and getattr(self.executor,'supports_cycles',False):
                    kwargs.update(cycle_max_calls=self.cycle_max_calls,resume_token=resume_token)
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
                    self.record['controller_actions']=total_actions
                    self.record['outcome']=deepcopy(outcome)
                    self.record['motion_executed']=bool(total_actions)
                if self.clock() >= deadline:raise ValueError('time_budget')
                if outcome.get('verified') is True:break
                yielded=(bounded and outcome.get('status')=='yielded' and outcome.get('reason')=='cycle_budget')
                recoverable=(bounded and outcome.get('recoverable') is True and outcome.get('safe_to_replan') is True
                             and outcome.get('reason')=='scene_changed' and repairs < self.max_replans)
                if not yielded and not recoverable:break
                if yielded:
                    if cycle_reviews>=self.max_cycle_reviews or cycle_reviews+repairs>=8:raise ValueError('cycle_review_budget')
                    if not isinstance(outcome.get('resume_token'),str) or used<=0:raise ValueError('invalid_cycle_yield')
                    resume_token=outcome['resume_token']
                if cycle_reviews+repairs>=8:raise ValueError('cycle_review_budget')
                if total_calls>=self.max_controller_calls:raise ValueError('call_budget_exhausted')
                phase='replanning';review_reason='cycle_budget' if yielded else 'scene_changed'
                if not self._event(generation,phase,{'attempt':cycle_reviews+repairs+1,'reason':review_reason}):return
                if yielded:
                    self.source.refresh();scene=self.source.brain_scene()
                else:
                    ensure=getattr(self.source,'ensure_brain_scene',None)
                    if ensure:scene=ensure()
                    else:self.source.refresh();scene=self.source.brain_scene()
                scene=self._runtime_scene(scene,True)
                if cancel.is_set():return
                if self.clock()>=deadline:raise ValueError('time_budget')
                repair_binding=self._binding(self.source.snapshot())
                completed=outcome.get('completed_step_ids',[])
                active_id=outcome.get('active_step_id') or outcome.get('failed_step_id') or 's1'
                feedback={'completed_step_ids':completed,'failed_step_id':active_id,'reason':review_reason}
                previous=outcome.get('current_plan') or (original_plan if original_plan.get('action')=='sequence' else plan)
                if yielded:
                    holding=outcome.get('holding_state','unknown');held=self.executor.planning_held_object() if hasattr(self.executor,'planning_held_object') else getattr(self.executor,'held',None)
                    phase_name=outcome.get('phase','approach')
                    phase_name={'released':'release','closing':'close','step_boundary':'approach'}.get(phase_name,phase_name)
                    feedback['cycle_context']={'cycle_index':cycle_reviews+1,'completed_micro_actions':total_actions,
                        'remaining_global_calls':self.max_controller_calls-total_calls,
                        'held_object':held if holding=='held_verified' else None,
                        'held_verified':holding in ('held_verified','empty'),'phase':phase_name,
                        'residual_xyz_m':None,'stop_reason':'cycle_budget','observation_status':'fresh',
                        'recent_actions':outcome.get('recent_actions',[])[-8:]}
                    progress=outcome.get('progress',[])
                    if progress:
                        feedback['cycle_context']['progress']=progress[-8:]
                        if progress[-1].get('phase')==phase_name:feedback['cycle_context']['residual_xyz_m']=progress[-1]['error_end_xyz_m']
                brain_started=self.clock()
                result=self.brain.replan(text,scene,previous,feedback,history)
                with self.lock:
                    if generation==self.generation:self.record['brain_roundtrip_seconds']=max(0.,self.clock()-brain_started)
                if yielded:cycle_reviews+=1
                else:repairs+=1
                if cancel.is_set():return
                if self.clock()>=deadline:raise ValueError('time_budget')
                if not result.accepted or result.plan is None:raise ValueError('brain_rejected')
                if self._binding(self.source.snapshot())!=repair_binding:raise ValueError('scene_stale')
                plan=result.plan.to_dict()
                if plan['action'] in ('clarify','stop','status'):
                    raise ValueError('planner_stopped_cycle')
                if plan['action'] not in (('pick','place','sequence') if getattr(self.executor,'supports_sequences',False) else ('pick','place')):raise ValueError('unsupported_task')
                with self.lock:
                    if generation!=self.generation:return
                    self.record['replans']=repairs;self.record['cycle_reviews']=cycle_reviews
                    self.record['cycle_index']=cycle_reviews
                    self.record['plan']=plan;self.plan_binding=repair_binding
                phase='executing'
                if not self._event(generation,phase,{'attempt':cycle_reviews+repairs+1}):return
            outcome={**outcome,'calls':total_calls,'actions':total_actions,'replans':repairs,'cycle_reviews':cycle_reviews}
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
            allowed = {'brain_rejected', 'unexpected_motion_plan', 'unsupported_task', 'execution_unverified','time_budget','invalid_execution_budget','cycle_review_budget','invalid_cycle_yield','planner_stopped_cycle','call_budget_exhausted'} | OBSERVATION_ERRORS
            reason = str(exc) if str(exc) in allowed else {'observing': 'observation_unavailable', 'planning': 'planning_failed', 'executing': 'execution_failed', 'replanning':'planning_failed'}[phase]
            with self.lock:
                if generation == self.generation:
                    self.record['error'] = reason
                    failure_state='blocked' if phase=='observing' or reason=='unsupported_task' else 'failed'
                    self.runtime_notices.append({'kind':'task_failure','request_id':self.record['id'],
                        'request_text':text[:200],'state':failure_state,'reason':reason,
                        'motion_executed':bool(self.record.get('motion_executed')),'at':self.clock()})
                    self.runtime_notices=self.runtime_notices[-20:]
            self._event(generation, 'blocked' if phase == 'observing' or reason == 'unsupported_task' else 'failed', reason)
        finally:
            if execute and self.executor is not None:
                if self.record.get('state')=='completed' and not cancel.is_set() and hasattr(self.executor,'pause'):
                    self.executor.pause()
                else:self.executor.stop()

    def reset(self):
        self.stop()
        with self.lock:self.history.clear();self.runtime_notices.clear()
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
