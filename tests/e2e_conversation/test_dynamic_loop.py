"""Independent dynamic-feedback acceptance.

MOCK observations/FakeJev unless explicitly noted; real-servo test is physics
with a synthetic target, NOT image-recognition or live-provider validation.
"""
import copy
import math
import threading

import pytest
from embodied_arm.control.jev_motion import JevMotionDriver
from tests.test_jev_motion import FakeSim, FakeJev, ZERO


def harness():
    sim=FakeSim();jev=FakeJev()
    return sim,jev,JevMotionDriver(sim,jev,sleep=sim.external_tick)


def target_update(snapshot):
    assert snapshot['provenance']=='MOCK-RGB-CONTRACT'
    return {'target_xyz':snapshot['target'],'covariance_m2':ZERO}


def test_disturbance_after_first_action_changes_next_jev_target_and_direction():
    sim,jev,driver=harness();target=[.224,0,.06];observations=[]
    def observe():
        value={'provenance':'MOCK-RGB-CONTRACT','frame':len(observations)+1,'target':list(target)}
        observations.append(value);return value
    def after():
        if len(sim.commands)==1:target[0]=.192
        return observe()
    result=driver.move_to([.224,0,.06],ZERO,before_decision=observe,
                          after_motion=after,target_update=target_update,max_calls=20)
    assert result['status']=='completed',result
    assert len(jev.requests)>1 and len(observations)>len(jev.requests)
    assert jev.requests[0]['target']['target_xyz'][0]==.224
    assert jev.requests[1]['target']['target_xyz'][0]==.192
    assert sim.commands[0][1][0]>.2
    assert sim.commands[1][1][0]<sim.commands[0][1][0]
    prior=[.2,0,.06]
    for action,xyz in sim.commands:
        assert action=='xyz' and math.dist(prior,xyz)<=.008000001
        prior=xyz
    assert abs(sim.state['ee'][0]-.192)<=.003
    assert result['grasp_verified'] is False


def test_new_observation_precedes_apparent_target_completion():
    sim,jev,driver=harness();observed=[]
    def observe():
        observed.append(1)
        return {'provenance':'MOCK-RGB-CONTRACT','target':[.217,0,.06]}
    result=driver.move_to([.2,0,.06],ZERO,before_decision=observe,
                          target_update=target_update,after_motion=observe)
    assert observed and jev.requests and result['actions']>0
    assert result['status']=='completed'


@pytest.mark.parametrize('reason',['stale_rgb','occluded'])
def test_invalid_observation_cannot_fake_completion_even_at_old_target(reason):
    sim,jev,driver=harness()
    def invalid():raise ValueError(reason)
    result=driver.move_to([.2,0,.06],ZERO,before_decision=invalid,target_update=target_update)
    assert result['status']=='stopped' and result['reason']=='perception_rejected'
    assert not jev.requests and not sim.commands and not result['grasp_verified']


def test_occlusion_after_first_action_prevents_future_calls():
    sim,jev,driver=harness()
    def observe():return {'provenance':'MOCK-RGB-CONTRACT','target':[.224,0,.06]}
    def occluded():raise ValueError('occluded')
    result=driver.move_to([.224,0,.06],ZERO,before_decision=observe,
                          target_update=target_update,after_motion=occluded)
    assert result['reason']=='perception_rejected'
    assert len(jev.requests)==len(sim.commands)==1
    assert not result['grasp_verified']


def test_stop_after_motion_cancels_future_observation_and_model_calls():
    sim,jev,driver=harness();cancel=threading.Event();before=[]
    def observe():
        before.append(1);return {'provenance':'MOCK-RGB-CONTRACT','target':[.224,0,.06]}
    def after():cancel.set();return {'provenance':'MOCK-RGB-CONTRACT','target':[.192,0,.06]}
    result=driver.move_to([.224,0,.06],ZERO,before_decision=observe,target_update=target_update,
                          after_motion=after,cancel_event=cancel)
    assert result['reason']=='cancelled'
    assert len(before)==len(jev.requests)==len(sim.commands)==1


def test_repeated_target_drift_cannot_reset_call_budget():
    sim,jev,driver=harness();counter=[0]
    def observe():
        counter[0]+=1
        return {'provenance':'MOCK-RGB-CONTRACT','target':[.25 if counter[0]%2 else .15,0,.06]}
    result=driver.move_to([.25,0,.06],ZERO,before_decision=observe,
                          target_update=target_update,max_calls=2)
    assert result['reason']=='call_budget_exhausted'
    assert len(jev.requests)==len(sim.commands)==2


def test_real_servo_corrects_synthetic_target_disturbance_with_fake_jev(monkeypatch):
    """Real SO101 servo physics + MOCK target feedback; no image-recognition claim."""
    from embodied_arm.simulation.so101_simulation import SO101Simulation
    import pybullet as p
    sim=SO101Simulation()
    try:
        def forbidden(*args,**kwargs):raise AssertionError('Oracle/reset/attachment forbidden')
        monkeypatch.setattr(sim,'evaluation_state',forbidden)
        for name in ('resetJointState','resetBasePositionAndOrientation','createConstraint'):
            monkeypatch.setattr(p,name,forbidden)
        initial=list(sim.proprioception()['ee']);target=list(initial);target[0]+=.024
        jev=FakeJev();observations=[];moves=[]
        real_manual=sim.manual
        def manual(action,values=None,**kwargs):
            moves.append((copy.deepcopy(sim.proprioception()['ee']),action,copy.deepcopy(values)))
            return real_manual(action,values,**kwargs)
        monkeypatch.setattr(sim,'manual',manual)
        def observe():
            observations.append(1)
            return {'provenance':'MOCK-RGB-CONTRACT','target':list(target)}
        def after():
            if len(moves)==1:target[0]=initial[0]-.009
            return observe()
        driver=JevMotionDriver(sim,jev,sleep=lambda _:sim.tick(24))
        result=driver.move_to(target,ZERO,before_decision=observe,target_update=target_update,
                              after_motion=after,max_calls=20)
        assert result['status']=='completed',result
        assert len(jev.requests)>1 and len(observations)>len(jev.requests)
        assert jev.requests[0]['target']['target_xyz'][0]>initial[0]
        assert jev.requests[1]['target']['target_xyz'][0]<initial[0]
        for before,action,values in moves:
            assert action=='xyz' and math.dist(before,values)<=.008000001
        assert abs(sim.proprioception()['ee'][0]-target[0])<.003
        assert result['provenance']=='offline_fake_jev' and result['grasp_verified'] is False
    finally:sim.close()


@pytest.mark.parametrize('replacement,reason',[
    ({'target_xyz':[float('nan'),0,.06],'covariance_m2':ZERO},'invalid_target_update'),
    ({'target_xyz':[.9,0,.06],'covariance_m2':ZERO},'target_outside_workspace'),
    ({'target_xyz':[.22,0,.06],'covariance_m2':[[1,2,0],[2,1,0],[0,0,1]]},'invalid_target_update'),
])
def test_dynamic_target_update_preserves_numeric_and_workspace_guards(replacement,reason):
    sim,jev,driver=harness()
    result=driver.move_to([.22,0,.06],ZERO,before_decision=lambda:{'mock':True},
                          target_update=lambda snapshot:replacement)
    assert result['status']=='stopped' and result['reason']==reason
    assert not jev.requests and not sim.commands


def test_slow_fresh_observation_cannot_extend_original_deadline():
    sim=FakeSim();jev=FakeJev();now=[0.]
    driver=JevMotionDriver(sim,jev,clock=lambda:now[0],sleep=sim.external_tick)
    def slow_observe():
        now[0]=11
        return {'provenance':'MOCK-RGB-CONTRACT','target':[.22,0,.06]}
    result=driver.move_to([.22,0,.06],ZERO,before_decision=slow_observe,
                          target_update=target_update,deadline=10)
    assert result['reason']=='deadline' and not jev.requests and not sim.commands


def test_explicit_old_rgb_timestamp_cannot_verify_old_target():
    import time
    sim,jev,driver=harness()
    def observe():
        return {'provenance':'MOCK-RGB-CONTRACT','target':[.2,0,.06],
                'scene':{'captured_at':time.monotonic()-10,
                         'frame_ids':{'left':'stale','right':'stale'}}}
    result=driver.move_to([.2,0,.06],ZERO,before_decision=observe,target_update=target_update)
    assert result['status']=='stopped' and result['reason']=='perception_rejected'
    assert not jev.requests and not sim.commands


class LoopSource:
    """MOCK scene versions; no simulation truth or image recognition."""
    def __init__(self):self.version=0;self.error=None
    def refresh(self):
        if self.error:raise ValueError(self.error)
        self.version+=1
    def ensure_brain_scene(self):
        self.refresh();return self.brain_scene()
    def brain_scene(self):return {'blocks':{},'mock_scene_version':self.version,'perception_source':'rgb_only'}
    def snapshot(self):
        if self.error:raise ValueError(self.error)
        return {'scene':{'frame_ids':{'left':str(self.version),'right':str(self.version)}},
                'robot':{'epoch':0,'actions':0}}
    def cancel(self):pass


class LoopBrain:
    def __init__(self):self.plans=[];self.repairs=[];self.on_repair=None
    def status(self):return {'available':True}
    @staticmethod
    def response(message):
        from types import SimpleNamespace
        plan={'action':'pick','object_id':'red','target_id':None,'message':message}
        return SimpleNamespace(accepted=True,provenance='MOCK-BRAIN-NOT-LIVE',
                               plan=SimpleNamespace(to_dict=lambda:plan))
    def interpret(self,text,scene,history=None):
        self.plans.append(copy.deepcopy(scene));return self.response('MOCK initial plan')
    def replan(self,text,scene,previous_plan,feedback,history):
        self.repairs.append(copy.deepcopy({'scene':scene,'previous_plan':previous_plan,'feedback':feedback}))
        if self.on_repair:self.on_repair()
        return self.response('MOCK repaired plan')
    def cancel(self):pass


class LoopExecutor:
    """Synthetic bounded-executor contract; outcomes are not physical evidence."""
    validated=True
    def __init__(self,outcomes):self.outcomes=outcomes;self.attempts=[];self.on_attempt=None;self.stops=0
    def readiness(self):return {'available':True,'reason':None}
    def bounded_execute(self,plan,*,cancel_event,event_callback,max_calls,deadline):
        self.attempts.append({'plan':copy.deepcopy(plan),'max_calls':max_calls,'deadline':deadline})
        for iteration in range(4):
            event_callback({'phase':'observation','iteration':iteration,
                            'frame_ids':{'left':f'MOCK-{iteration}','right':f'MOCK-{iteration}'},
                            'observation_age_seconds':0,'goal':[.2,0,.06]})
        if self.on_attempt:self.on_attempt()
        return copy.deepcopy(self.outcomes[min(len(self.attempts)-1,len(self.outcomes)-1)])
    def execute(self,*args,**kwargs):raise AssertionError('Bounded executor must receive shared budgets')
    def stop(self):self.stops+=1


def repairable(calls=2):
    return {'verified':False,'recoverable':True,'safe_to_replan':True,
            'reason':'scene_changed','calls':calls,'actions':1,'provenance':'MOCK-NOT-PHYSICAL'}


def make_loop(outcomes,**kwargs):
    from embodied_arm.services.task_service import TaskService
    source=LoopSource();brain=LoopBrain();executor=LoopExecutor(outcomes)
    return source,brain,executor,TaskService(scene_source=source,brain=brain,executor=executor,**kwargs)


def settle(service):
    import time
    for _ in range(300):
        result=service.status()
        if not result['pending']:return result
        time.sleep(.01)
    raise AssertionError('MOCK loop did not terminate')


def test_scene_change_repairs_once_with_fresh_scene_and_shared_budget():
    source,brain,executor,service=make_loop([repairable(),{'verified':True,'calls':3,'actions':2}],
                                           max_replans=1,max_controller_calls=7,max_seconds=30)
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='completed',result
    assert len(brain.plans)==len(brain.repairs)==1
    assert source.version>=2
    assert brain.repairs[0]['scene']['mock_scene_version']>brain.plans[0]['mock_scene_version']
    assert brain.repairs[0]['feedback']['reason']=='scene_changed'
    assert [attempt['max_calls'] for attempt in executor.attempts]==[7,5]
    assert executor.attempts[0]['deadline']==executor.attempts[1]['deadline']
    assert any(event['state']=='replanning' for event in result['events'])


def test_many_observations_do_not_call_qwen_per_frame():
    source,brain,executor,service=make_loop([{'verified':True,'calls':4,'actions':4}])
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='completed'
    observations=[event for event in result['events'] if isinstance(event.get('detail'),dict)
                  and event['detail'].get('phase')=='observation']
    assert len(observations)==4 and len(brain.plans)==1 and not brain.repairs


def test_accepted_unverified_actions_cannot_be_task_completion():
    _,brain,executor,service=make_loop([{'verified':False,'calls':3,'actions':3,'reason':'grasp_unverified'}])
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed' and result['outcome']['verified'] is False
    assert len(executor.attempts)==1 and not brain.repairs
    assert not any(event.get('detail')=='visually_verified' for event in result['events'])


def test_recovery_requires_explicit_safe_to_replan_flag():
    unsafe=repairable();unsafe['safe_to_replan']=False
    _,brain,executor,service=make_loop([unsafe])
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed' and len(executor.attempts)==1 and not brain.repairs


def test_repeated_disturbance_cannot_exceed_one_replan():
    _,brain,executor,service=make_loop([repairable()],max_replans=1,max_controller_calls=10)
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed'
    assert len(brain.repairs)==1 and len(executor.attempts)==2


def test_shared_budget_exhaustion_prevents_paid_replan_and_second_execute():
    _,brain,executor,service=make_loop([repairable(calls=3)],max_controller_calls=3)
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed'
    assert len(executor.attempts)==1 and not brain.repairs


def test_occlusion_during_repair_blocks_new_qwen_and_execution():
    source,brain,executor,service=make_loop([repairable()])
    executor.on_attempt=lambda:setattr(source,'error','occluded')
    service.submit('MOCK pick');result=settle(service)
    assert result['state'] in ('blocked','failed')
    assert len(executor.attempts)==1 and not brain.repairs
    assert not any(event.get('detail')=='visually_verified' for event in result['events'])


def test_stop_during_qwen_repair_cancels_future_execute():
    _,brain,executor,service=make_loop([repairable(),{'verified':True,'calls':1,'actions':1}])
    brain.on_repair=service.stop
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='cancelled'
    assert len(brain.repairs)==1 and len(executor.attempts)==1


def test_original_task_deadline_does_not_reset_for_repair():
    now=[0.]
    _,brain,executor,service=make_loop([repairable()],clock=lambda:now[0],max_seconds=10)
    executor.on_attempt=lambda:now.__setitem__(0,11.)
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed'
    assert len(executor.attempts)==1 and not brain.repairs


def test_stop_during_repair_observation_blocks_replan_and_motion():
    source,brain,executor,service=make_loop([repairable()])
    refresh=source.refresh
    def cancel_second_refresh():
        refresh()
        if source.version==2:service.stop()
    source.refresh=cancel_second_refresh
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='cancelled' and not brain.repairs and len(executor.attempts)==1


def test_changed_frame_during_replan_rejects_new_plan():
    source,brain,executor,service=make_loop([repairable(),{'verified':True,'calls':1,'actions':1}])
    brain.on_repair=lambda:setattr(source,'version',source.version+1)
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed' and result['error']=='scene_stale'
    assert len(brain.repairs)==1 and len(executor.attempts)==1


def test_late_verified_executor_result_cannot_bypass_task_deadline():
    now=[0.]
    _,brain,executor,service=make_loop([{'verified':True,'calls':1,'actions':1}],
                                      clock=lambda:now[0],max_seconds=10)
    executor.on_attempt=lambda:now.__setitem__(0,11.)
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed' and result['error']=='time_budget'
    assert not any(event.get('detail')=='visually_verified' for event in result['events'])


@pytest.mark.parametrize('outcome',[
    {'verified':True,'calls':-1,'actions':0},
    {'verified':True,'calls':4,'actions':1},
    {'verified':True,'calls':1,'actions':-1},
    {'verified':True,'calls':True,'actions':1},
    {'verified':True,'actions':1},
    {'verified':True,'calls':1},
])
def test_invalid_executor_accounting_cannot_claim_completion(outcome):
    _,brain,executor,service=make_loop([outcome],max_controller_calls=3)
    service.submit('MOCK pick');result=settle(service)
    assert result['state']=='failed' and result['error']=='invalid_execution_budget'
    assert not brain.repairs


def test_fresh_preclose_displacement_blocks_gripper_alignment():
    """Synthetic visual estimate plus encoder pose; no simulator object oracle."""
    from dataclasses import replace
    from embodied_arm.control.manipulation import VisualManipulationExecutor
    from tests.test_phase_visual_feedback import observation
    sim=FakeSim();sim.state['ee']=[.1,.1,.013]
    executor=VisualManipulationExecutor(simulation=sim,observer=object(),driver=object(),clock=lambda:1.)
    scene=observation().scene
    snapshot={'source':'rgb_only','schema':'so101-camera-observation-v1',
              'scene':scene.to_dict(),'robot':sim.proprioception()}
    geometry={'approach_offset_world':[0.,0.,0.],'tcp_below_top_m':.007}
    executor._check_grasp_alignment(snapshot,'red',geometry)
    objects=dict(scene.objects)
    objects['red']=replace(objects['red'],xyz=(.112,.1,.02))
    snapshot['scene']=replace(scene,objects=objects).to_dict()
    with pytest.raises(ValueError,match='preclose_alignment_unverified'):
        executor._check_grasp_alignment(snapshot,'red',geometry)
    assert not sim.commands
