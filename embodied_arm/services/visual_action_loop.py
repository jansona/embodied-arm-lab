"""Simulation-only Qwen visual decisions followed by one Jev actuator choice.

No image interpretation, object geometry or task recipes are implemented here.
"""
from copy import deepcopy
import math
import threading
import time
import uuid
from embodied_arm.providers.visual_brain import sanitize_robot, frame_snapshot, validate_visual_result


class VisualActionLoop:
    def __init__(self, sim, qwen, jev, *, cycle_max_calls=32, max_calls=200,
                 max_seconds=420, clock=time.monotonic, sleep=time.sleep):
        if type(cycle_max_calls) is not int or not 1<=cycle_max_calls<=200: raise ValueError('invalid_cycle_budget')
        if type(max_calls) is not int or not 1<=max_calls<=200: raise ValueError('invalid_call_budget')
        if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 1<=max_seconds<=420: raise ValueError('invalid_deadline')
        self.sim,self.qwen,self.jev=sim,qwen,jev
        self.cycle_max_calls,self.max_calls,self.max_seconds=cycle_max_calls,max_calls,max_seconds
        self.clock,self.sleep=clock,sleep
        self.lock=threading.RLock();self.cancelled=threading.Event();self.thread=None
        self.record=self._empty();self.history=[];self.notices=[]

    def _empty(self):
        return dict(id=None,state='idle',pending=False,events=[],observation=None,controller_calls=0,
                    controller_actions=0,qwen_calls=0,cycle_index=0,outcome=None)

    def _event(self,state,**detail):
        with self.lock:
            self.record['state']=state
            self.record['events'].append(dict(state=state,at=self.clock(),detail=detail))
            self.record['events']=self.record['events'][-200:]

    def status(self):
        with self.lock:
            out=deepcopy(self.record);out['history']=deepcopy(self.history);out['runtime_notices']=deepcopy(self.notices)
        try: ready=bool(self.qwen.status().get('available')) and bool(self.jev.status().get('available'))
        except Exception: ready=False
        out.update(execution_available=ready,unavailable_reason=None if ready else 'provider_unavailable',
                   execution_validation='experimental_simulation',cycle_max_calls=self.cycle_max_calls,
                   max_calls=self.max_calls,max_seconds=self.max_seconds)
        return out

    def submit(self,text,execute=True):
        if not isinstance(text,str) or not 0<len(text.strip())<=2000 or type(execute) is not bool: raise ValueError('invalid_request')
        with self.lock:
            if self.record['pending']: raise ValueError('task_busy')
            self.cancelled=threading.Event();self.record=self._empty()
            self.record.update(id=uuid.uuid4().hex,pending=True,state='queued')
            self.history.append(dict(role='user',text=text));self.history=self.history[-40:]
            self.thread=threading.Thread(target=self._run,args=(text,execute),daemon=True);self.thread.start()
            return self.record['id']

    def _check(self,deadline):
        if self.cancelled.is_set(): raise RuntimeError('cancelled')
        if self.clock()>deadline: raise RuntimeError('deadline')

    def _robot(self): return sanitize_robot(self.sim.proprioception())

    def _unchanged(self,old):
        new=self._robot()
        if new['epoch']!=old['epoch'] or new['actions']!=old['actions'] or new['busy'] or not new['paused']:
            raise RuntimeError('robot_changed_during_decision')
        if max(abs(a-b) for a,b in zip(new['joints'],old['joints']))>.005:
            raise RuntimeError('robot_changed_during_decision')
        if math.dist(new['ee'],old['ee'])>.002 or abs(new['gripper_openness']-old['gripper_openness'])>.02:
            raise RuntimeError('robot_changed_during_decision')
        return new

    def _run(self,text,execute):
        deadline=self.clock()+self.max_seconds;previous=None;recent=[]
        try:
            self.sim.pause_idle()
            while True:
                self._check(deadline)
                if not self.qwen.status().get('available'): raise RuntimeError('qwen_unavailable')
                if execute and self.record['controller_calls']==0 and not self.jev.status().get('available'): raise RuntimeError('jev_unavailable')
                self._event('observing',phase='observing')
                robot=self._robot();captured=self.sim.capture_frames();packet=frame_snapshot(captured)
                # These labels are fixed camera mount metadata, not image inference.
                if captured.get('camera_profile')=='observer_wrist':
                    for image in packet['images']:
                        image['camera_id']={'left':'observer','right':'wrist'}.get(image['camera_id'],image['camera_id'])
                self._check(deadline);self._unchanged(robot)
                review=bool(self.record['controller_calls'] and self.record['controller_calls']%self.cycle_max_calls==0)
                if review:
                    self.record['cycle_index']+=1
                    self._event('reviewing',phase='cycle_review',calls=self.record['controller_calls'],actions=self.record['controller_actions'])
                self.record['qwen_calls']+=1
                result=self.qwen.observe(text,packet,robot,history={'previous_observation':previous,'applied_actions':recent[-8:]},review=review)
                self._check(deadline);self._unchanged(robot);frame_snapshot(packet)
                result=validate_visual_result(result,frame_id=packet['frame_id'])
                if not result['accepted']: raise RuntimeError('qwen_'+result['reason'])
                observation=result['observation'];previous=observation
                self.record['observation']=deepcopy(observation)
                self._event('planning',phase='visual_decision',observation=observation)
                with self.lock:
                    self.history.append(dict(role='assistant',text=observation['local_objective'],provenance='qwen_visual_assessment'))
                    self.history=self.history[-40:]
                decision=observation['decision']
                if not execute or decision!='continue':
                    state='preview' if not execute else {'completed':'completed','clarify':'clarification','stop':'stopped'}[decision]
                    with self.lock:
                        self._check(deadline)
                        self.record['outcome']={'completion_basis':'qwen_visual_assessment','model_assessed_complete':bool(execute and decision=='completed'),
                            'reason':decision,'calls':self.record['controller_calls'],'actions':self.record['controller_actions']}
                        self._event(state,phase='visual_judgment',decision=decision)
                    break
                if self.record['controller_calls']>=self.max_calls: raise RuntimeError('call_budget')
                if not self.jev.status().get('available'): raise RuntimeError('jev_unavailable')
                self.record['controller_calls']+=1
                answer=self.jev.decide(observation,robot)
                from embodied_arm.providers.visual_cerebellum import validate_jev_result
                answer=validate_jev_result(answer)
                self._check(deadline);self._unchanged(robot);frame_snapshot(packet)
                self._event('executing',phase='decision_received',action=answer['proposed_action'],accepted=answer['accepted'],
                            confidence=answer['confidence'],calls=self.record['controller_calls'],actions=self.record['controller_actions'])
                if not answer['accepted']: raise RuntimeError('jev_'+answer['reason'])
                action=answer['action'];applied=False
                if action!='hold':
                    # The lock makes Stop and command submission mutually exclusive.
                    with self.lock:
                        self._check(deadline);self._unchanged(robot)
                        self.sim.resume()
                        if action in ('gripper_open_10pct','gripper_close_10pct'):
                            target=robot['gripper_openness']+(.1 if action=='gripper_open_10pct' else -.1)
                            if not 0<=target<=1:raise RuntimeError('mechanical_limit')
                            self.sim.manual('gripper',target,origin='jev')
                        else:
                            axis,sign,size=action.split('_');target=list(robot['ee'])
                            target['xyz'.index(axis)]+=(float(size[:-2])/1000)*(1 if sign=='plus' else -1)
                            self.sim.manual('xyz',target,origin='jev')
                        applied=True;self.record['controller_actions']+=1
                        recent.append(dict(action=action,applied=True))
                        self._event('executing',phase='motion_applied',action=action,calls=self.record['controller_calls'],actions=self.record['controller_actions'])
                    motion_deadline=min(deadline,self.clock()+15)
                    while self._robot()['busy']:
                        self._check(motion_deadline);self.sleep(.01)
                    current=self._robot()
                    if current['epoch']!=robot['epoch'] or current['actions']!=robot['actions']+1:
                        raise RuntimeError('robot_changed_during_decision')
                    self.sim.pause_idle()
                else: recent.append(dict(action=action,applied=False))
                self._check(deadline)
                self._event('executing',phase='motion_completed',action=action,applied=applied,
                            calls=self.record['controller_calls'],actions=self.record['controller_actions'])
        except Exception as exc:
            reason=str(exc) if isinstance(exc,(RuntimeError,ValueError)) else 'execution_error'
            allowed={'cancelled','deadline','call_budget','robot_changed_during_decision','qwen_unavailable','jev_unavailable','stale_frame','mechanical_limit'}
            if reason not in allowed and not reason.startswith(('qwen_','jev_')): reason='execution_error'
            self.sim.stop()
            self.record['outcome']={'model_assessed_complete':False,'reason':reason,'calls':self.record['controller_calls'],'actions':self.record['controller_actions']}
            self._event('cancelled' if self.cancelled.is_set() else 'failed',phase='stopped',reason=reason)
            self.notices.append(dict(request_id=self.record['id'],reason=reason,text='Task stopped; completion has not been established.'))
            self.notices=self.notices[-40:]
        finally:
            with self.lock:self.record['pending']=False

    def stop(self):
        with self.lock:
            self.cancelled.set();self.sim.stop()
            if self.record['pending']:self._event('stopping',phase='stopping')
        for client in (self.qwen,self.jev):
            if hasattr(client,'cancel'):client.cancel()

    def reset(self):
        with self.lock:
            if self.record['pending']:raise ValueError('task_busy')
            self.sim.reset();self.record=self._empty();self.history=[];self.notices=[]

    def close(self):
        self.stop()
        if self.thread:self.thread.join(timeout=2)
        return not (self.thread and self.thread.is_alive())

    def wait_closed(self,timeout=None):
        if self.thread:self.thread.join(timeout=timeout)
        return not (self.thread and self.thread.is_alive())
