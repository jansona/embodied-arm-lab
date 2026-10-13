"""Bounded symbolic sequence runner; model order, measured feedback, no fallback."""
from __future__ import annotations
import copy
import threading
import time
from brain import validate_plan
from stack_plans import plan_for_symbolic_step, symbolic_step_result
from task_executor import TaskExecutor

OUTCOME_REASONS = {'grasp_not_verified':'grasp_failed', 'placement_not_verified':'placement_failed',
                   'support_moved':'support_moved', 'support_unstable':'scene_changed',
                   'invalid_observation':'scene_changed'}


class _ControllerBudget:
    """One Jev request budget across all primitives and an optional repair."""
    def __init__(self, controller, limit):self.controller=controller;self.limit=limit;self.calls=0
    def decide(self, obs):
        if self.calls>=self.limit:raise ValueError('Shared controller budget exhausted')
        self.calls+=1
        return self.controller.decide(obs)
    def cancel(self):
        if hasattr(self.controller,'cancel'):self.controller.cancel()


class SequenceExecutor:
    def __init__(self, sim, controller=None, *, brain=None, text='', history=None,
                 allow_replan=False, executor_factory=TaskExecutor,
                 max_controller_calls=200, max_seconds=420,
                 ground=plan_for_symbolic_step, measure=symbolic_step_result):
        if (max_controller_calls,max_seconds) not in ((120,300),(200,420)):raise ValueError('Unsupported sequence budget')
        self.max_controller_calls=max_controller_calls;self.max_seconds=max_seconds
        self.sim=sim;self.brain=brain;self.text=text;self.history=history or []
        self.allow_replan=allow_replan;self.factory=executor_factory;self.ground=ground;self.measure=measure
        self.controller=_ControllerBudget(controller,max_controller_calls) if controller is not None else None
        self.lock=threading.RLock();self.epoch=0;self.thread=None;self.active=None
        self.records=[];self.done_actions=0;self.done_calls=0
        self.result={'status':'idle','controller':'jev' if controller is not None else 'deterministic servo',
                     'actions':0,'controller_calls':0,'reason':None,'replans':0,'steps':[],
                     'max_steps':6,'max_actions':200,'max_controller_calls':max_controller_calls,'max_seconds':max_seconds}

    def status(self):
        with self.lock:
            state=copy.deepcopy(self.result);state['steps']=copy.deepcopy(self.records)
            active=self.active.status() if self.active else {}
            state['actions']=self.done_actions+active.get('actions',0)
            state['controller_calls']=self.controller.calls if self.controller else 0
            if active.get('last_decision'):state['last_decision']=active['last_decision']
            return state

    def stop(self):
        with self.lock:
            self.epoch+=1
            if self.active:self.active.stop()
            if self.brain and hasattr(self.brain,'cancel'):self.brain.cancel()
            self.sim.stop();self.result.update(status='stopped',reason='cancelled')
            for record in self.records:
                if record['status']=='running':record['status']='stopped'

    def start(self, plan):
        validated=validate_plan(plan,self.sim.state()).to_dict()
        if validated['action']!='sequence':raise ValueError('A symbolic sequence is required')
        with self.lock:
            if self.thread and self.thread.is_alive():raise ValueError('Sequence still pending')
            if self.sim.state()['busy']:raise ValueError('Robot busy')
            self.epoch+=1;epoch=self.epoch
            self.result.update(status='running',reason=None)
            self.records=[{**step,'status':'pending','outcome':None,'revision':0} for step in validated['steps']]
            self.thread=threading.Thread(target=self._run,args=(epoch,validated,time.monotonic()+self.max_seconds),daemon=True)
            self.thread.start()
        return self.status()

    def _valid(self,epoch):return epoch==self.epoch

    def _finish(self,epoch,status,reason=None):
        with self.lock:
            if not self._valid(epoch):return
            self.result.update(status=status,reason=reason)
            self.sim.stop()

    def _run(self,epoch,plan,deadline):
        completed=[];revision=0;attempts=0;offset=0
        try:
            while True:
                repair=None
                for index,step in enumerate(plan['steps']):
                    with self.lock:
                        if not self._valid(epoch):return
                        record=self.records[offset+index]
                        if time.monotonic()>=deadline:self._finish(epoch,'failed','time_budget');return
                        if attempts>=6 or self.done_actions>=200:self._finish(epoch,'failed','action_budget');return
                        if self.controller and self.controller.calls>=self.max_controller_calls:self._finish(epoch,'failed','controller_budget');return
                        attempts+=1;record['status']='running';self.result['step']=offset+index
                        before=self.sim.state()
                        try:primitive=self.ground(step,before)
                        except (ValueError,TypeError,KeyError):
                            outcome={'success':False,'verified':False,'reason':'scene_changed'}
                            leaf=None
                        else:
                            leaf=self.factory(self.sim,controller=self.controller);self.active=leaf
                            leaf.start(primitive,max_seconds=min(120,max(.01,deadline-time.monotonic())),
                                       max_actions=min(100,200-self.done_actions))
                    if leaf:
                        while leaf.status()['status']=='running':
                            with self.lock:
                                if not self._valid(epoch):return
                                if time.monotonic()>=deadline:
                                    leaf.stop();self._finish(epoch,'failed','time_budget');return
                            time.sleep(.01)
                        with self.lock:
                            if not self._valid(epoch):return
                            leaf_state=leaf.status();self.done_actions+=leaf_state.get('actions',0)
                            self.active=None
                            if leaf_state['status']!='completed':
                                record.update(status='failed',outcome={'success':False,'verified':False,
                                              'reason':leaf_state.get('reason') or 'motion_failed'})
                                # Controller rejection never triggers a model retry or another controller.
                                self._finish(epoch,'failed',leaf_state.get('reason') or 'motion_failed');return
                            outcome=self.measure(step,before,self.sim.state())
                    with self.lock:
                        if not self._valid(epoch):return
                        record['outcome']=outcome
                        record['status']='completed' if outcome.get('verified') else 'failed'
                        if outcome.get('verified'):
                            completed.append(step['step_id']);continue
                        reason=OUTCOME_REASONS.get(outcome.get('reason'), 'scene_changed')
                        if not self.allow_replan or revision>=1 or not self.brain or not hasattr(self.brain,'replan'):
                            self._finish(epoch,'failed',reason);return
                        self.result.update(status='replanning',reason=reason,replans=1)
                        feedback={'completed_step_ids':completed,'failed_step_id':step['step_id'],'reason':reason}
                        scene=self.sim.state()
                    response=self.brain.replan(self.text,scene,plan,feedback,history=self.history)
                    with self.lock:
                        if not self._valid(epoch):return
                        self.result['brain_calls']=response.calls
                        if not response.accepted or response.plan is None:
                            self._finish(epoch,'failed','replan_rejected');return
                        repair=response.plan.to_dict()
                        if repair['action'] in ('clarify','stop','status'):
                            self.result.update(message=repair['message'],repair_plan=repair)
                            self._finish(epoch,'needs_input' if repair['action']=='clarify' else 'stopped','repair_requires_user');return
                        repair=validate_plan(repair,self.sim.state()).to_dict()
                        if repair['action']!='sequence' or len(repair['steps'])>6-attempts:
                            self._finish(epoch,'failed','repair_exceeds_step_budget');return
                        for pending in self.records[offset+index+1:]:pending['status']='superseded'
                        offset=len(self.records);revision+=1
                        self.records.extend({**s,'status':'pending','outcome':None,'revision':revision} for s in repair['steps'])
                        self.result.update(status='running',reason=None,repair_plan=repair,message=repair['message'])
                    break
                if repair is not None:
                    plan=repair;completed=[];continue
                with self.lock:
                    if not self._valid(epoch):return
                    self.result['outcome']={'verified':True,'description':'全部符号步骤的物理结果已验证'}
                    self._finish(epoch,'completed')
                return
        except Exception:
            with self.lock:
                if self._valid(epoch) and self.active:self.active.stop()
            self._finish(epoch,'failed','sequence_error')
