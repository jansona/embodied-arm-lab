"""Offline sequence orchestration: order, fresh observations, feedback and bounds."""
import copy
import threading
import time
from brain import BrainPlan, BrainResult, BrainStep
from sequence_executor import SequenceExecutor


def plan():
    return {'action':'sequence','object_id':None,'target_id':None,'message':'先夹起蓝色，再放到红色上',
            'steps':[{'step_id':'s1','action':'pick','object_id':'blue','target_id':None},
                     {'step_id':'s2','action':'place','object_id':'blue','target_id':'on:red'}]}

class Sim:
    def __init__(self):
        self.data={'blocks':{'red':[.50,0,.025],'blue':[.58,-.19,.025]},'ee':[.5,0,.27],
                   'goal':[.48,.21,.025],'gripper_width':.08,'busy':False,'version':0}
        self.stops=0
    def state(self):return copy.deepcopy(self.data)
    def stop(self):self.stops+=1

class Leaf:
    def __init__(self,sim,controller=None):self.sim=sim;self.result={'status':'idle','actions':0,'controller_calls':0}
    def start(self,primitive,**kwargs):
        self.sim.data['version']+=1
        step=primitive['intent']
        if step['action']=='pick':self.sim.data['held_object']=step['object_id']
        else:self.sim.data.pop('held_object',None)
        self.result.update(status='completed',actions=1)
    def status(self):return dict(self.result)
    def stop(self):self.result['status']='stopped'

def wait(runner):
    runner.thread.join(3)
    assert not runner.thread.is_alive()
    return runner.status()


def test_order_fresh_grounding_and_measured_outcomes():
    sim=Sim();seen=[]
    def ground(step,state):seen.append((step['step_id'],state['version']));return {'intent':step}
    runner=SequenceExecutor(sim,executor_factory=Leaf,ground=ground,
        measure=lambda step,before,after:{'verified':True,'success':True,'reason':'completed'})
    runner.start(plan());s=wait(runner)
    assert seen==[('s1',0),('s2',1)]
    assert s['status']=='completed' and s['outcome']['verified']
    assert [x['status'] for x in s['steps']]==['completed','completed']
    assert s['actions']==2 and s['controller_calls']==0


def test_failed_physical_measurement_stops_next_step_without_implicit_repair():
    class Brain:
        def replan(self,*args,**kwargs):raise AssertionError('No implicit replan')
    runner=SequenceExecutor(Sim(),brain=Brain(),executor_factory=Leaf,ground=lambda s,_:{'intent':s},
        measure=lambda *args:{'verified':False,'reason':'grasp_not_verified'})
    runner.start(plan());s=wait(runner)
    assert s['status']=='failed' and s['reason']=='grasp_failed'
    assert s['actions']==1 and s['steps'][1]['status']=='pending'
    assert s['replans']==0


def test_explicit_repair_uses_factual_feedback_and_one_shared_attempt():
    class Brain:
        def __init__(self):self.calls=[]
        def replan(self,text,scene,previous,feedback,history=None):
            self.calls.append(feedback)
            return BrainResult(accepted=True,calls=2,plan=BrainPlan(action='sequence',message='放下蓝色',
                steps=(BrainStep('s1','place','blue','on:red'),)))
    brain=Brain();measurements=iter([False,True])
    runner=SequenceExecutor(Sim(),brain=brain,allow_replan=True,executor_factory=Leaf,ground=lambda s,_:{'intent':s},
        measure=lambda *args:{'verified':next(measurements),'reason':'grasp_not_verified'})
    runner.start(plan());s=wait(runner)
    assert brain.calls==[{'completed_step_ids':[],'failed_step_id':'s1','reason':'grasp_failed'}]
    assert s['status']=='completed' and s['replans']==1 and s['brain_calls']==2
    assert [x['status'] for x in s['steps']]==['failed','superseded','completed']


def test_second_failure_never_repairs_again():
    class Brain:
        calls=0
        def replan(self,*args,**kwargs):
            self.calls+=1
            return BrainResult(accepted=True,plan=BrainPlan(action='sequence',message='放下',
                steps=(BrainStep('s1','place','blue','on:red'),)))
    brain=Brain();runner=SequenceExecutor(Sim(),brain=brain,allow_replan=True,executor_factory=Leaf,
        ground=lambda s,_:{'intent':s},measure=lambda *args:{'verified':False,'reason':'placement_not_verified'})
    runner.start(plan());s=wait(runner)
    assert s['status']=='failed' and brain.calls==1 and s['replans']==1


def test_stop_discards_pending_repair_and_never_starts_late_motion():
    entered=threading.Event();release=threading.Event()
    class Brain:
        def replan(self,*args,**kwargs):
            entered.set();release.wait(2)
            return BrainResult(accepted=True,plan=BrainPlan(action='sequence',message='放下',
                steps=(BrainStep('s1','place','blue','on:red'),)))
    sim=Sim();runner=SequenceExecutor(sim,brain=Brain(),allow_replan=True,executor_factory=Leaf,
        ground=lambda s,_:{'intent':s},measure=lambda *args:{'verified':False,'reason':'placement_not_verified'})
    runner.start(plan());assert entered.wait(2)
    assert runner.status()['status']=='replanning'
    runner.stop();release.set();s=wait(runner)
    assert s['status']=='stopped' and sim.data['version']==1


def test_controller_rejection_does_not_replan_or_fallback():
    class FailedLeaf(Leaf):
        def start(self,*args,**kwargs):self.result.update(status='failed',reason='controller_low_confidence')
    class Brain:
        def replan(self,*args,**kwargs):raise AssertionError('No retry of low-confidence controller')
    runner=SequenceExecutor(Sim(),brain=Brain(),allow_replan=True,executor_factory=FailedLeaf,
        ground=lambda s,_:{'intent':s})
    runner.start(plan());s=wait(runner)
    assert s['reason']=='controller_low_confidence' and s['actions']==0


def test_real_contact_physics_sequence_stacks_blue_on_red():
    from simulator import Simulation
    sim=Simulation();runner=SequenceExecutor(sim)
    try:
        runner.start(plan());end=time.monotonic()+20
        while runner.thread.is_alive() and time.monotonic()<end:
            sim.tick(24);time.sleep(.001)
        s=wait(runner)
        assert s['status']=='completed',s
        assert s['outcome']['verified']
        assert s['steps'][-1]['outcome']['support_stable']
        assert abs(sim.state()['blocks']['blue'][2]-sim.state()['blocks']['red'][2]-.05)<.015
    finally:runner.stop();sim.close()


def test_shared_controller_budget_cannot_reset_between_steps():
    class Controller:
        calls=0
        def decide(self,obs):self.calls+=1
    class CostlyLeaf(Leaf):
        def __init__(self,sim,controller=None):super().__init__(sim,controller);self.controller=controller
        def start(self,*args,**kwargs):
            for _ in range(110):self.controller.decide({})
            super().start(*args,**kwargs)
    controller=Controller();runner=SequenceExecutor(Sim(),controller=controller,executor_factory=CostlyLeaf,
        ground=lambda s,_:{'intent':s},measure=lambda *args:{'verified':True})
    runner.start(plan());s=wait(runner)
    assert controller.calls==200 and s['controller_calls']==200
    assert s['status']=='failed'


def test_normal_and_recovery_budgets_are_explicitly_bounded():
    import pytest
    normal=SequenceExecutor(Sim())
    recovery=SequenceExecutor(Sim(),max_controller_calls=200,max_seconds=420)
    assert normal.status()['max_controller_calls']==200 and normal.status()['max_seconds']==420
    assert recovery.status()['max_controller_calls']==200 and recovery.status()['max_seconds']==420
    with pytest.raises(ValueError):SequenceExecutor(Sim(),max_controller_calls=201,max_seconds=420)
