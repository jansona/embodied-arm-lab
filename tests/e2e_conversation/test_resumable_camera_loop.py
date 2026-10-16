"""Independent continuation/camera acceptance; no live provider or object oracle."""
from dataclasses import replace
import numpy as np


def test_wrist_capture_binds_fresh_extrinsics_to_actual_servo_pose(monkeypatch):
    from embodied_arm.simulation.so101_simulation import SO101Simulation
    from embodied_arm.perception.camera_perception import CameraCalibration
    from embodied_arm.perception.moving_camera_geometry import compatible_calibrations
    sim=SO101Simulation(camera_profile='observer_wrist')
    try:
        monkeypatch.setattr(sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('Object oracle forbidden')))
        first=sim.capture_frames()
        joints=list(sim.proprioception()['joints']);joints[0]+=.05
        sim.resume();sim.manual('joints',joints,origin='offline_fixture')
        for _ in range(100):
            sim.tick(24)
            if not sim.proprioception()['busy']:break
        sim.pause_idle()
        second=sim.capture_frames()
        assert first['frame_id']!=second['frame_id'] and first['captured_at']<second['captured_at']
        before={c['camera_id']:CameraCalibration.from_dict(c) for c in first['calibrations']}
        after={c['camera_id']:CameraCalibration.from_dict(c) for c in second['calibrations']}
        assert before['left']==after['left']
        assert not np.allclose(before['right'].world_from_camera,after['right'].world_from_camera)
        assert compatible_calibrations(before,after,moving_wrist=True)
        assert not compatible_calibrations(before,after,moving_wrist=False)
        for meta in second['view_metadata']:
            assert meta['frame_id']==second['frame_id'] and meta['captured_at']==second['captured_at']
            assert meta['calibration']==after[meta['camera_id']].to_dict()
        assert {m['role'] for m in second['view_metadata']}=={'observer','wrist'}
        from copy import deepcopy
        import pytest
        from embodied_arm.perception.moving_camera_geometry import validate_capture_geometry
        validate_capture_geometry(second)
        mismatched=deepcopy(second);mismatched['view_metadata'][1]['frame_id']=first['frame_id']
        with pytest.raises(ValueError,match='Stale or mismatched'):
            validate_capture_geometry(mismatched)
        invalid=dict(after);invalid['right']=replace(after['right'],fx=after['right'].fx+1.)
        assert not compatible_calibrations(before,invalid,moving_wrist=True)
        invalid=dict(after);pose=np.array(after['left'].world_from_camera);pose[0,3]+=.01
        invalid['left']=replace(after['left'],world_from_camera=tuple(map(tuple,pose.tolist())))
        assert not compatible_calibrations(before,invalid,moving_wrist=True)
    finally:sim.close()


class ScriptedCycleDriver:
    """MOCK driver used only to isolate the real continuation state machine."""
    validated=True
    def __init__(self):self.operations=[];self.move_count=0
    def gripper(self,target,**kwargs):
        self.operations.append(('gripper',target,kwargs['max_calls']))
        return {'status':'actuation_completed','calls':1,'actions':1,'provenance':'MOCK'}
    def move_to(self,target,covariance,**kwargs):
        self.move_count+=1;self.operations.append(('move',tuple(target),kwargs['max_calls']))
        return {'status':'yielded' if self.move_count==1 else 'completed',
                'calls':kwargs['max_calls'] if self.move_count==1 else 1,'actions':1,'provenance':'MOCK'}


def resumable_fixture(monkeypatch):
    from types import SimpleNamespace
    from tests.test_jev_motion import FakeSim,ZERO
    from embodied_arm.control.manipulation import VisualManipulationExecutor
    sim=FakeSim();driver=ScriptedCycleDriver();now=[10.]
    executor=VisualManipulationExecutor(simulation=sim,observer=SimpleNamespace(validated=True),
        driver=driver,clock=lambda:now[0])
    def program(plan):
        yield {'kind':'gripper','phase':'open','target':1}
        yield {'kind':'move','phase':'approach','target':[.22,0,.06],'covariance':ZERO}
        yield {'kind':'gripper','phase':'close','target':0,'geometry':{}}
        return {'verified':True,'provenance':'MOCK-STATE-MACHINE-ONLY'}
    monkeypatch.setattr(executor,'_program',program)
    return sim,driver,executor,now


def test_sixteen_call_yield_resumes_pending_phase_without_replaying_gripper(monkeypatch):
    sim,driver,executor,_=resumable_fixture(monkeypatch)
    plan={'action':'pick','object_id':'red','target_id':None}
    first=executor.bounded_execute(plan,max_calls=20,deadline=100.,cycle_max_calls=16)
    assert first['status']=='yielded' and first['calls']==16 and not first['verified']
    assert sim.proprioception()['paused'] and not sim.stopped
    second=executor.bounded_execute(plan,max_calls=4,deadline=100.,cycle_max_calls=16,resume_token=first['resume_token'])
    assert second['status']=='completed' and second['calls']==2
    assert [op[:2] for op in driver.operations if op[0]=='gripper']==[('gripper',1),('gripper',0)]
    assert len([op for op in driver.operations if op[0]=='move'])==2


def test_resumed_primitive_cannot_reset_total_call_budget(monkeypatch):
    _,driver,executor,_=resumable_fixture(monkeypatch)
    plan={'action':'pick','object_id':'red','target_id':None}
    first=executor.bounded_execute(plan,max_calls=17,deadline=100.,cycle_max_calls=16)
    second=executor.bounded_execute(plan,max_calls=17,deadline=100.,cycle_max_calls=16,resume_token=first['resume_token'])
    assert second['status']=='failed' and second['reason']=='call_budget_exhausted'
    assert second['calls']==1 and not second['verified']
    assert not any(op[:2]==('gripper',0) for op in driver.operations)


def test_stop_invalidates_continuation_and_prevents_future_driver_calls(monkeypatch):
    import pytest
    _,driver,executor,_=resumable_fixture(monkeypatch)
    plan={'action':'pick','object_id':'red','target_id':None}
    first=executor.bounded_execute(plan,max_calls=20,deadline=100.,cycle_max_calls=16)
    count=len(driver.operations);executor.stop()
    with pytest.raises(ValueError,match='invalid_resume'):
        executor.bounded_execute(plan,max_calls=4,deadline=100.,cycle_max_calls=16,resume_token=first['resume_token'])
    assert len(driver.operations)==count


def test_resume_rejects_new_intent_or_extended_deadline(monkeypatch):
    import pytest
    _,driver,executor,_=resumable_fixture(monkeypatch)
    plan={'action':'pick','object_id':'red','target_id':None}
    first=executor.bounded_execute(plan,max_calls=20,deadline=100.,cycle_max_calls=16)
    count=len(driver.operations)
    with pytest.raises(ValueError,match='invalid_resume'):
        executor.bounded_execute(dict(plan,object_id='blue'),max_calls=4,deadline=100.,cycle_max_calls=16,resume_token=first['resume_token'])
    with pytest.raises(ValueError,match='resume_deadline_extended'):
        executor.bounded_execute(plan,max_calls=4,deadline=101.,cycle_max_calls=16,resume_token=first['resume_token'])
    assert len(driver.operations)==count


def test_real_driver_yields_at_sixteen_decisions_without_invalidating_scene():
    from tests.test_jev_motion import FakeSim,FakeJev,ZERO
    from embodied_arm.control.jev_motion import JevMotionDriver
    sim=FakeSim();controller=FakeJev();driver=JevMotionDriver(sim,controller,sleep=sim.external_tick)
    initial_epoch=sim.proprioception()['epoch'];captures=[]
    def observe():
        captures.append(1)
        return {'target_xyz':[.29 if len(controller.requests)%2==0 else .11,0,.06],
                'covariance_m2':ZERO}
    outcome=driver.move_to([.29,0,.06],ZERO,max_calls=16,yield_on_budget=True,
        before_decision=observe,after_motion=observe,target_update=lambda snapshot:snapshot)
    assert outcome['status']=='yielded' and outcome['reason']=='cycle_budget'
    assert outcome['calls']==len(controller.requests)==16 and outcome['actions']==16
    assert sim.proprioception()['epoch']==initial_epoch and sim.proprioception()['paused']
    assert not sim.stopped and len(captures)>16 and not outcome['grasp_verified']


def sequence_task_fixture(final_verified=True):
    """Real TaskService and sequence wrapper, MOCK primitive verdicts/providers."""
    from types import SimpleNamespace
    from tests.e2e_conversation.test_dynamic_loop import LoopSource,LoopBrain
    from tests.test_sequence_execution import Primitive,outcome,placed,sequence,single
    from embodied_arm.services.sequence_execution import SequenceExecutionService
    from embodied_arm.services.task_service import TaskService
    class Brain(LoopBrain):
        def interpret(self,text,scene,history=None):
            self.plans.append(scene)
            return SimpleNamespace(accepted=True,provenance='MOCK',plan=SimpleNamespace(to_dict=lambda:sequence(('pick','red',None),('place','red','goal'))))
        def replan(self,text,scene,previous_plan,feedback,history):
            super().replan(text,scene,previous_plan,feedback,history)
            return SimpleNamespace(accepted=True,provenance='MOCK',plan=SimpleNamespace(to_dict=lambda:single('place','red','goal')))
    final=placed() if final_verified else dict(placed(status='failed',verified=False),reason='unknown_support')
    primitive=Primitive(outcome(calls=1),outcome('yielded',calls=15,phase='retreat',
        holding='release_pending',token='MOCK-CHILD-CHECKPOINT'),final)
    primitive.pause=lambda:None  # MOCK primitive implements production pause contract.
    source=LoopSource();brain=Brain();wrapper=SequenceExecutionService(primitive)
    service=TaskService(scene_source=source,brain=brain,executor=wrapper,max_controller_calls=20)
    return source,brain,primitive,service


def test_task_cycle_review_keeps_completed_steps_and_resumes_release_only():
    from tests.e2e_conversation.test_dynamic_loop import settle
    source,brain,primitive,service=sequence_task_fixture()
    service.submit('MOCK move red');status=settle(service)
    assert status['state']=='completed',status
    assert len(brain.plans)==1 and len(brain.repairs)==1
    review=brain.repairs[0];feedback=review['feedback']
    assert feedback['completed_step_ids']==['s1'] and feedback['failed_step_id']=='s2'
    assert feedback['cycle_context']['remaining_global_calls']==4
    assert feedback['cycle_context']['held_verified'] is False
    assert review['scene']['mock_scene_version']>brain.plans[0]['mock_scene_version']
    assert [plan['action'] for plan,_ in primitive.requests]==['pick','place','place']
    assert primitive.requests[-1][1]['resume_token']=='MOCK-CHILD-CHECKPOINT'
    assert len({options['deadline'] for _,options in primitive.requests})==1
    assert status['outcome']['calls']==17 and status['cycle_reviews']==1


def test_hidden_support_after_resume_cannot_complete_sequence():
    from tests.e2e_conversation.test_dynamic_loop import settle
    _,brain,primitive,service=sequence_task_fixture(final_verified=False)
    service.submit('MOCK move red');status=settle(service)
    assert status['state']=='failed' and status['outcome']['verified'] is False,status
    assert status['outcome']['reason']=='unknown_support'
    assert status['outcome']['completed_step_ids']==['s1']
    assert not any(event.get('detail')=='visually_verified' for event in status['events'])


def test_stop_during_cycle_review_blocks_resumed_release():
    from tests.e2e_conversation.test_dynamic_loop import settle
    _,brain,primitive,service=sequence_task_fixture()
    brain.on_repair=service.stop
    service.submit('MOCK move red');status=settle(service)
    assert status['state']=='cancelled' and len(primitive.requests)==2


def test_stack_verifier_never_reuses_hidden_support_from_initial_frame():
    """Synthetic image-estimate contract tests the real final stack verifier."""
    from tests.test_phase_visual_feedback import observation
    from embodied_arm.control.stack_verification import verify_stack_release
    initial=observation().scene
    objects=dict(initial.objects)
    objects['blue']=replace(objects['red'],object_id='blue')
    objects['red']=replace(objects['red'],xyz=(.1,.1,.04))
    initial=replace(initial,objects=objects)
    hidden=observation(t=2.,z=.04).scene  # blue is explicitly unknown in this fixture
    result=verify_stack_release({'source':'rgb_only','scene':initial.to_dict()},
        {'source':'rgb_only','scene':initial.to_dict()},
        {'source':'rgb_only','scene':hidden.to_dict()},'red','blue',now=2.)
    assert result['verified'] is False and result['physical_contact_verified'] is False
    assert result['reason']=='support_visual_evidence_unavailable'


def test_stack_primitive_normalizes_symbolic_target_for_phase_observer():
    """Catch the on:blue planner token versus blue observer ID boundary."""
    from types import SimpleNamespace
    from tests.test_jev_motion import FakeSim
    from embodied_arm.control.manipulation import VisualManipulationExecutor
    seen=[]
    def begin_step(action,object_id,target_id=None,held_evidence=None):
        seen.append(target_id)
        assert target_id=='blue','Observer requires support object ID, not planner on: prefix'
    observer=SimpleNamespace(configure_task=lambda action:None,begin_step=begin_step)
    executor=VisualManipulationExecutor(simulation=FakeSim(),observer=observer,driver=object())
    program=executor._program({'action':'place','object_id':'red','target_id':'on:blue'})
    assert next(program)=={'kind':'observe','phase':'approach'}
    assert seen==['blue']


def test_resume_rejects_joint_change_even_when_tcp_and_epoch_match(monkeypatch):
    import pytest
    sim,driver,executor,_=resumable_fixture(monkeypatch)
    plan={'action':'pick','object_id':'red','target_id':None}
    first=executor.bounded_execute(plan,max_calls=20,deadline=100.,cycle_max_calls=16)
    count=len(driver.operations)
    sim.state['joints'][0]+=.01  # MOCK encoder drift with unchanged TCP/epoch/action count
    with pytest.raises(ValueError,match='resume_scene_changed'):
        executor.bounded_execute(plan,max_calls=4,deadline=100.,cycle_max_calls=16,resume_token=first['resume_token'])
    assert len(driver.operations)==count


def test_second_review_uses_revised_server_owned_plan_not_original_tail():
    from collections import deque
    from types import SimpleNamespace
    from tests.test_sequence_execution import outcome,placed,sequence,single
    from tests.e2e_conversation.test_dynamic_loop import LoopBrain,settle
    _,brain,primitive,service=sequence_task_fixture()
    service.cycle_max_calls=1
    primitive.results=deque([outcome(),outcome('yielded',phase='approach',token='MOCK-PLACE'),placed()])
    def replan(text,scene,previous_plan,feedback,history):
        LoopBrain.replan(brain,text,scene,previous_plan,feedback,history)
        plan=sequence(('place','red','on:blue')) if len(brain.repairs)==1 else single('place','red','on:blue')
        return SimpleNamespace(accepted=True,provenance='MOCK',plan=SimpleNamespace(to_dict=lambda:plan))
    brain.replan=replan
    service.submit('MOCK revise placement');status=settle(service)
    assert status['state']=='completed',status
    assert len(brain.repairs)==2
    second=brain.repairs[1]
    assert second['previous_plan']['steps'][1]['target_id']=='on:blue'
    assert second['feedback']['completed_step_ids']==['s1'] and second['feedback']['failed_step_id']=='s2'
    assert [plan['action'] for plan,_ in primitive.requests]==['pick','place','place']


def test_release_pending_cannot_reuse_old_lift_certificate_for_planning(monkeypatch):
    sim,_,executor,_=resumable_fixture(monkeypatch)
    executor.held='red';executor.held_evidence={'epoch':sim.proprioception()['epoch']}
    assert executor.planning_held_object()=='red'
    executor._run_state={'holding_state':'release_pending'}
    assert executor.planning_held_object() is None
    executor._run_state={'holding_state':'unknown'}
    assert executor.planning_held_object() is None


def release_guard_fixture():
    from types import SimpleNamespace
    from tests.test_jev_motion import FakeSim
    from tests.test_phase_visual_feedback import observation
    from embodied_arm.control.manipulation import VisualManipulationExecutor
    from embodied_arm.perception.camera_perception import EstimatedScene
    sim=FakeSim();sim.state['ee']=[.1,.1,.052]
    executor=VisualManipulationExecutor(simulation=sim,observer=SimpleNamespace(),driver=object(),clock=lambda:1.)
    executor._support_policy={'support_id':'blue','grip_offset':[0.,0.,.022]}
    scene=observation().scene;objects=dict(scene.objects)
    objects['blue']=replace(objects['red'],object_id='blue')
    objects['red']=replace(objects['red'],xyz=(.1,.1,.042))
    scene=replace(scene,objects=objects)
    snapshot={'source':'rgb_only','schema':'so101-camera-observation-v1',
        'scene':scene.to_dict(),'robot':sim.proprioception()}
    def resolver(current,carried):
        support=EstimatedScene.from_dict(current['scene']).objects['blue']
        return {'target_xyz':[support.xyz[0],support.xyz[1],support.xyz[2]+.032],
                'expected_object_top_xyz':[support.xyz[0],support.xyz[1],support.xyz[2]+.022],
                'covariance_m2':[[support.covariance_m2[i][j]+carried.covariance_m2[i][j] for j in range(3)] for i in range(3)]}
    return sim,executor,scene,snapshot,resolver


def test_shifted_support_blocks_release_at_old_tcp():
    import pytest
    sim,executor,scene,snapshot,resolver=release_guard_fixture()
    executor._check_release_alignment(snapshot,'red',resolver)
    objects=dict(scene.objects);objects['blue']=replace(objects['blue'],xyz=(.112,.1,.02))
    snapshot['scene']=replace(scene,objects=objects).to_dict()
    with pytest.raises(ValueError):executor._check_release_alignment(snapshot,'red',resolver)
    from tests.test_jev_motion import FakeJev
    from embodied_arm.control.jev_motion import JevMotionDriver
    controller=FakeJev();sim.state['gripper_openness']=.15
    driver=JevMotionDriver(sim,controller,clock=lambda:1.,sleep=sim.external_tick)
    def before_open():
        executor._check_release_alignment(snapshot,'red',resolver)
        return snapshot
    verdict=driver.gripper(1,before_decision=before_open)
    assert verdict['status']=='stopped' and verdict['reason']=='perception_rejected'
    assert not sim.commands and not controller.requests


def test_carried_cube_slip_blocks_release_despite_unchanged_tcp_and_support():
    import pytest
    sim,executor,scene,snapshot,resolver=release_guard_fixture()
    executor._check_release_alignment(snapshot,'red',resolver)
    objects=dict(scene.objects);objects['red']=replace(objects['red'],xyz=(.112,.1,.042))
    snapshot['scene']=replace(scene,objects=objects).to_dict()
    with pytest.raises(ValueError):executor._check_release_alignment(snapshot,'red',resolver)
    from tests.test_jev_motion import FakeJev
    from embodied_arm.control.jev_motion import JevMotionDriver
    controller=FakeJev();sim.state['gripper_openness']=.15
    driver=JevMotionDriver(sim,controller,clock=lambda:1.,sleep=sim.external_tick)
    def before_open():
        executor._check_release_alignment(snapshot,'red',resolver)
        return snapshot
    verdict=driver.gripper(1,before_decision=before_open)
    assert verdict['status']=='stopped' and verdict['reason']=='perception_rejected'
    assert not sim.commands and not controller.requests
