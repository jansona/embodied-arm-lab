"""Offline FakeJev contracts; these are not live-model validation."""
from copy import deepcopy
import threading
import pytest
from embodied_arm.control.jev_motion import JevMotionDriver
from embodied_arm.providers.so101_jev import parse_response, ALIYUN_MODEL


class FakeSim:
    def __init__(self):
        self.state={'name':'SO-ARM101','arm_dof':5,'ee':[.2,0,.06],
                    'joints':[0]*5,'joint_limits':[[-3,3]]*5,
                    'gripper_openness':1.,'gripper_angle':1.74533,'paused':True,
                    'busy':False,'epoch':1,'actions':0,
                    'workspace':[[.07,.33],[-.19,.19],[.008,.33]],'cube_edge_m':.02}
        self.commands=[];self.stopped=False;self.pending=None
    def proprioception(self):return deepcopy(self.state)
    def evaluation_state(self):raise AssertionError('Oracle forbidden')
    def resume(self):self.state['paused']=False
    def manual(self,action,values=None,*,origin):
        assert origin=='offline_fixture'
        self.commands.append((action,values));self.pending=(action,values)
        self.state['busy']=True;self.state['actions']+=1
    def external_tick(self,_):
        if self.pending:
            action,values=self.pending
            if action=='xyz':self.state['ee']=list(values)
            else:self.state['gripper_openness']=1. if action=='open' else .15
            self.pending=None;self.state['busy']=False
    def pause_idle(self):
        assert not self.state['busy'];self.state['paused']=True
    def stop(self):self.stopped=True;self.pending=None;self.state['busy']=False;self.state['paused']=True;self.state['epoch']+=1


class FakeJev:
    provenance='offline_fake_jev'
    def status(self):return {'available':True,'provenance':self.provenance}
    def __init__(self,confidence=.9,choice=None,on_call=None):
        self.requests=[];self.confidence=confidence;self.choice=choice;self.on_call=on_call
    def decide(self,obs):
        self.requests.append(deepcopy(obs))
        if self.on_call:self.on_call()
        if obs['target']['kind']=='cartesian':
            axis=obs['axis_index'];error=obs['target']['target_xyz'][axis]-obs['robot']['ee'][axis]
            options=('positive','negative','hold')
            choice='positive' if error>.003 else 'negative' if error<-.003 else 'hold'
        else:
            error=obs['target']['target_openness']-obs['robot']['gripper_openness']
            options=('open','close','hold');choice='open' if error>.02 else 'close' if error<-.02 else 'hold'
        choice=self.choice or choice
        return parse_response({'model':ALIYUN_MODEL,'answers':{'action':{
            'type':'choice','choice':choice,'confidence':self.confidence,
            'probabilities':{k:.9 if k==choice else .05 for k in options}}}},obs)


def driver(sim=None,controller=None,**kwargs):
    sim=sim or FakeSim();controller=controller or FakeJev()
    return sim,controller,JevMotionDriver(sim,controller,sleep=sim.external_tick,**kwargs)


ZERO=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]


def test_model_selected_bounded_xyz_and_provenance():
    sim,jev,motion=driver();events=[]
    result=motion.move_to([.217,-.009,.051],ZERO,event_callback=events.append)
    assert result['status']=='completed' and result['provenance']=='offline_fake_jev'
    assert result['calls']==result['actions']==len(jev.requests)==len(sim.commands)
    assert [o['axis_index'] for o in jev.requests]==sorted(o['axis_index'] for o in jev.requests)
    prior=[.2,0,.06]
    for action,xyz in sim.commands:
        assert action=='xyz'
        assert sum(abs(a-b)>1e-9 for a,b in zip(xyz,prior))==1
        assert max(abs(a-b) for a,b in zip(xyz,prior))<=.008000001
        prior=xyz
    assert all(b['status']=='inside' for b in result['axis_uncertainty'])
    assert not sim.stopped


def test_full_uncertainty_envelope_is_not_loosened():
    sim,jev,motion=driver()
    cov=[[.0008**2 if i==j else 0 for j in range(3)] for i in range(3)]
    result=motion.move_to([.201,0,.06],cov)
    assert result['reason']=='uncertainty_unresolved'
    assert not jev.requests and not sim.commands and sim.stopped


@pytest.mark.parametrize('confidence,choice',[(.64,None),(.9,'negative')])
def test_rejection_has_no_fallback(confidence,choice):
    sim,jev,motion=driver(controller=FakeJev(confidence,choice))
    result=motion.move_to([.22,0,.06],ZERO)
    assert result['reason']=='controller_rejected' and not sim.commands


def test_gripper_blocked_by_object_does_not_require_encoder_zero():
    sim,jev,motion=driver();result=motion.gripper(0)
    assert result['status']=='completed' and result['reason']=='actuation_completed'
    assert result['grasp_verified'] is False and result['calls']==1
    assert sim.state['gripper_openness']==.15 and sim.commands==[('close',None)]


def test_gripper_hold_is_model_selected_and_not_grasp_proof():
    sim,jev,motion=driver();result=motion.gripper(1)
    assert result['calls']==1 and result['reason']=='encoder_within_tolerance'
    assert not sim.commands and not result['grasp_verified']


def test_robot_changed_during_model_is_rejected():
    sim=FakeSim();jev=FakeJev(on_call=lambda:sim.state.update(actions=3))
    _,_,motion=driver(sim,jev)
    assert motion.gripper(0)['reason']=='robot_changed_during_decision'
    assert not sim.commands


def test_pre_cancel_and_model_latency_cancel():
    cancel=threading.Event();cancel.set()
    sim,jev,motion=driver()
    assert motion.gripper(0,cancel_event=cancel)['reason']=='cancelled'
    assert not jev.requests
    cancel.clear();sim,jev,motion=driver(controller=FakeJev(on_call=cancel.set))
    assert motion.gripper(0,cancel_event=cancel)['reason']=='cancelled'
    assert not sim.commands


def test_callbacks_have_required_timing_and_fail_closed():
    sim,jev,motion=driver();order=[]
    jev.on_call=lambda:order.append('decide')
    def before():
        assert sim.state['paused'] and not sim.state['busy'];order.append('before')
    def after():
        assert sim.state['paused'] and not sim.state['busy'];order.append('after')
    result=motion.move_to([.217,0,.06],ZERO,before_decision=before,after_motion=after)
    assert result['status']=='completed'
    assert order==['before','decide','after']*result['calls']
    sim,jev,motion=driver()
    result=motion.gripper(0,before_decision=lambda:False)
    assert result['reason']=='perception_rejected' and result['calls']==0 and not sim.commands
    sim,jev,motion=driver()
    def fail():raise ValueError('occluded')
    result=motion.gripper(0,after_motion=fail)
    assert result['reason']=='perception_rejected' and result['actions']==1 and sim.stopped


def test_deadline_budget_workspace_and_covariance_guards():
    sim,jev,motion=driver(clock=lambda:5)
    assert motion.gripper(0,deadline=5)['reason']=='deadline'
    sim,jev,motion=driver()
    assert motion.move_to([.4,0,.06],ZERO)['reason']=='target_outside_workspace'
    assert not jev.requests
    sim,jev,motion=driver()
    assert motion.move_to([.23,0,.06],ZERO,max_calls=1)['reason']=='call_budget_exhausted'
    assert len(jev.requests)==1
    sim,jev,motion=driver()
    bad=[[1e-6,2e-6,0],[2e-6,1e-6,0],[0,0,1e-6]]
    assert motion.move_to([.22,0,.06],bad)['reason']=='invalid_covariance'
    assert not jev.requests


def test_real_simulation_servo_motion_with_fake_jev_and_external_ticks(monkeypatch):
    from embodied_arm.simulation.so101_simulation import SO101Simulation
    import pybullet as p
    sim=SO101Simulation()
    try:
        def forbidden(*args,**kwargs):raise AssertionError('No oracle, reset, or attachment')
        monkeypatch.setattr(sim,'evaluation_state',forbidden)
        for name in ('resetJointState','resetBasePositionAndOrientation','createConstraint'):
            monkeypatch.setattr(p,name,forbidden)
        target=list(sim.proprioception()['ee']);target[0]+=.009
        jev=FakeJev()
        # Test harness emulates the app's physics loop; driver never ticks.
        motion=JevMotionDriver(sim,jev,sleep=lambda _:sim.tick(24))
        result=motion.move_to(target,ZERO)
        assert result['status']=='completed' and result['actions']==1
        assert result['provenance']=='offline_fake_jev'
        result=motion.gripper(0)
        assert result['status']=='completed' and result['grasp_verified'] is False
    finally:
        sim.close()


def test_motion_timeout_stops_without_driver_ticking():
    sim=FakeSim();now=[0.]
    def sleep(_):now[0]+=.1
    motion=JevMotionDriver(sim,FakeJev(),clock=lambda:now[0],sleep=sleep)
    result=motion.gripper(0,deadline=20.)
    assert result['reason']=='motion_timeout' and sim.stopped


def test_model_callback_cannot_bypass_perception_failure():
    sim,jev,motion=driver()
    def change():sim.state['ee'][0]+=.005
    result=motion.gripper(0,before_decision=change)
    assert result['reason']=='robot_changed_during_decision'
    assert result['calls']==0 and not sim.commands


def test_amplitude_schedule_avoids_preventable_indeterminate_landing():
    sim,jev,motion=driver()
    result=motion.move_to([.22,0,.06],ZERO)
    assert result['status']=='completed' and len(sim.commands)==3
    assert result['amplitude_schedule']=='up_to_8mm_avoid_indeterminate_landing_no_direction_selection'
    prior=.2
    for _,xyz in sim.commands:
        assert 0 < xyz[0]-prior <= .008000001
        residual=abs(.22-xyz[0])
        assert residual <= .002 or residual > .004
        prior=xyz[0]
    assert all(request['axis_index']==0 for request in jev.requests)


def test_precision_floor_never_loosens_tolerance():
    sim,jev,motion=driver()
    covariance=[[.0011**2 if i==j else 0 for j in range(3)] for i in range(3)]
    result=motion.move_to([.22,0,.06],covariance)
    assert result['reason']=='precision_floor' and not sim.commands and not jev.requests
