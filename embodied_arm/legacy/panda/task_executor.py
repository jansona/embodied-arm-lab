"""Validated brain-plan executor with explicit, replaceable low-level controller.

Deterministic control is never attributed to Jev. No networking or credentials.
"""
import math
import threading
import time
from embodied_arm.legacy.panda.control import apply_action, observation
from embodied_arm.simulation.simulator import WORKSPACE


def validate_plan(plan):
    if not isinstance(plan,dict) or set(plan)!={'steps'} or not isinstance(plan['steps'],list) or not 1<=len(plan['steps'])<=12:
        raise ValueError('Plan requires 1..12 steps')
    out=[]
    for step in plan['steps']:
        if not isinstance(step,dict):raise ValueError('Invalid step')
        action=step.get('action')
        if action=='move':
            if set(step)!={'action','target_xyz'}:raise ValueError('Invalid move fields')
            xyz=step['target_xyz']
            if not isinstance(xyz,list) or len(xyz)!=3 or not all(type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi for v,(lo,hi) in zip(xyz,WORKSPACE)):
                raise ValueError('Move outside workspace')
            out.append({'action':'move','target_xyz':list(xyz)})
        elif action=='lift':
            if set(step)!={'action','height_m'} or type(step['height_m']) not in (int,float) or not .05<=step['height_m']<=.45:raise ValueError('Invalid lift height')
            out.append(dict(step))
        elif action in ('open','close','hold') and set(step)=={'action'}:out.append(dict(step))
        else:raise ValueError('Unsupported plan action')
    return {'steps':out}


class TaskExecutor:
    def __init__(self,sim,controller=None):
        self.sim=sim;self.controller=controller
        self.label='jev' if controller is not None else 'deterministic servo'
        self.lock=threading.RLock();self.epoch=0;self.thread=None
        self.result={'status':'idle','controller':self.label,'actions':0,'controller_calls':0,'step':0,'reason':None}
    def status(self):
        with self.lock:return dict(self.result)
    def stop(self):
        with self.lock:
            self.epoch+=1;self.result.update(status='stopped',reason='cancelled');self.sim.stop()
            if self.controller is not None and hasattr(self.controller,'cancel'):
                try:self.controller.cancel()
                except Exception:pass
    def start(self,plan,max_seconds=120,max_actions=100):
        plan=validate_plan(plan)
        if type(max_actions) is not int or not 1<=max_actions<=100 or not isinstance(max_seconds,(int,float)) or not 0<max_seconds<=120:raise ValueError('Invalid task budget')
        with self.lock:
            if self.thread and self.thread.is_alive():raise ValueError('Previous task still pending')
            if self.sim.state()['busy']:raise ValueError('Robot busy')
            self.epoch+=1;epoch=self.epoch
            with self.sim.lock:self.sim.paused=False
            self.result={'status':'running','controller':self.label,'actions':0,'controller_calls':0,'step':0,'reason':None}
            self.thread=threading.Thread(target=self._run,args=(epoch,plan,time.monotonic()+max_seconds,max_actions),daemon=True);self.thread.start()
        return self.status()
    def _finish(self,epoch,status,reason=None):
        with self.lock:
            if epoch!=self.epoch:return
            self.result.update(status=status,reason=reason);self.sim.stop()
    def _run(self,epoch,plan,deadline,max_actions):
        try:
            for index,step in enumerate(plan['steps']):
                target=None;grip_sent=False
                while True:
                    with self.lock:
                        if epoch!=self.epoch:return
                        if time.monotonic()>deadline:self._finish(epoch,'failed','time_budget');return
                        state=self.sim.state();self.result['step']=index
                        if state['paused']:self._finish(epoch,'failed','simulation_paused');return
                        busy=state['busy']
                    if busy:time.sleep(.003);continue
                    action=step['action']
                    if action=='hold':break
                    if action in ('open','close'):
                        if grip_sent:break
                        if self.controller is None:command=(action,None)
                        else:
                            obs=observation(self.sim);obs['objective']={'id':'gripper','target_state':'open' if action=='open' else 'closed'};obs['task']='Set the gripper to the supplied target state.'
                            with self.lock:
                                if self.result['controller_calls']>=80:raise ValueError('Controller call budget')
                                self.result['controller_calls']+=1
                            decision=self.controller.decide(obs)
                            with self.lock:
                                if epoch!=self.epoch:return
                                self.result['last_decision']=decision.to_dict()
                            if not decision.accepted:
                                self._finish(epoch,'failed','controller_'+decision.reason if decision.reason in ('low_confidence','timeout','network_error','invalid_response') else 'controller_error');return
                            if decision.action=='hold':
                                already=(state['gripper_width']>=.075) if action=='open' else (state['gripper_width']<.075)
                                if not already:raise ValueError('Gripper held in wrong state')
                                break
                            if decision.action!=action:raise ValueError('Wrong gripper action')
                            command=('controller',decision.action)
                        grip_sent=True
                    else:
                        if target is None:
                            target=list(step['target_xyz']) if action=='move' else [*state['ee'][:2],step['height_m']]
                            if action=='lift' and target[2]<state['ee'][2]:raise ValueError('Lift cannot descend')
                        error=math.dist(state['ee'],target)
                        if self.controller is not None and max(abs(a-b) for a,b in zip(state['ee'],target))<=.01:
                            # Independent terminal safety predicate; never claimed as a model hold.
                            with self.lock:self.result['terminal_safety_stop']=True
                            break
                        if error<.005:break
                        if self.controller is None:
                            scale=min(1,.02/error)
                            command=('xyz',[a+(b-a)*scale for a,b in zip(state['ee'],target)])
                        else:
                            obs=observation(self.sim);obs['objective']={'id':'local_target','target_xyz':target,'tolerance_m':.01};obs['task']='Move to the supplied local Cartesian target, then hold.'
                            with self.lock:
                                if self.result['controller_calls']>=80:raise ValueError('Controller call budget')
                                self.result['controller_calls']+=1
                            decision=self.controller.decide(obs)
                            with self.lock:
                                if epoch!=self.epoch:return
                            with self.lock:self.result['last_decision']=decision.to_dict()
                            if not decision.accepted:
                                self._finish(epoch,'failed','controller_'+decision.reason if decision.reason in ('low_confidence','timeout','network_error','invalid_response') else 'controller_error');return
                            if decision.action=='hold':
                                if max(abs(a-b) for a,b in zip(state['ee'],target))<=.01:break
                                raise ValueError('Controller held away from target')
                            if decision.action not in ('x_plus','x_minus','y_plus','y_minus','z_plus','z_minus'):raise ValueError('Unexpected local movement action')
                            axis='xyz'.index(decision.action[0])
                            step_size=min(.02,abs(target[axis]-state['ee'][axis]))
                            if step_size<1e-6:raise ValueError('Selected axis already aligned')
                            predicted=list(state['ee']);predicted[axis]+=step_size if decision.action.endswith('plus') else -step_size
                            if math.dist(predicted,target)>=error:raise ValueError('Model move would not progress')
                            command=('controller',decision.action,step_size)
                    with self.lock:
                        if epoch!=self.epoch:return
                        if time.monotonic()>deadline:raise ValueError('Time budget')
                        if self.result['actions']>=max_actions:raise ValueError('Action budget')
                        with self.sim.lock:
                            if command[0]=='controller':apply_action(self.sim,command[1],origin=self.label,step_m=command[2] if len(command)>2 else .02)
                            else:
                                self.sim.manual(*command)
                                origin='brain plan / deterministic gripper' if command[0] in ('open','close') else self.label
                                self.sim.mode=origin
                                if self.sim.queue:self.sim.queue[-1]['origin']=origin
                        self.result['actions']+=1
            self._finish(epoch,'completed')
        except Exception:
            self._finish(epoch,'failed','task_or_controller_error')


def plan_from_intent(intent,state):
    """Ground a brain intent in current simulator measurements, never model XYZ."""
    if not isinstance(intent,dict) or set(intent)-{'action','object_id','target_id','message'}:raise ValueError('Invalid intent fields')
    action=intent.get('action');object_id=intent.get('object_id');target_id=intent.get('target_id')
    if action not in ('pick','place') or object_id not in ('red','blue') or target_id not in (None,'goal'):raise ValueError('Intent is not an executable object task')
    if action=='place' and target_id!='goal':raise ValueError('Place needs known goal')
    block=state['blocks'][object_id];goal=state['goal'];steps=[]
    # Standard geometry-specific pick primitive; Qwen selects intent/object,
    # this deterministic decomposition is not claimed as Jev planning.
    if action=='pick':
        steps=[{'action':'open'},{'action':'move','target_xyz':[block[0],block[1],.25]},
               {'action':'move','target_xyz':[block[0],block[1],.029]},
               {'action':'close'},{'action':'lift','height_m':.27}]
    elif state['blocks'][object_id][2]<.10:
        raise ValueError('Place requires an already lifted object; request pick with goal instead')
    if target_id=='goal':
        steps.extend([{'action':'move','target_xyz':[goal[0],goal[1],.27]},
                      {'action':'move','target_xyz':[goal[0],goal[1],.031]},
                      {'action':'open'},{'action':'lift','height_m':.28}])
    return validate_plan({'steps':steps})
