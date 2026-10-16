"""Recipe-level wiring tests, explicit fake perception/driver, no physics claim."""
import threading
from types import SimpleNamespace
from embodied_arm.control.manipulation import VisualManipulationExecutor

class Sim:
    def stop(self):pass
    def grasp_geometry(self,center):return {'approach_offset_world':[.01,0,0],'tcp_below_top_m':.007}
class Observer:
    validated=True
    def __init__(self,failures=0):self.failures=failures;self.observations=0
    def configure_task(self,action):pass
    def begin_phase(self,*args):return {'x':.18}
    def observe(self):
        self.observations+=1
        if self.observations<=self.failures:raise ValueError('RGB association lost')
        return {'x':.183}
class Driver:
    validated=True
    def __init__(self):self.updated=None;self.budget=None
    def gripper(self,*args,**kwargs):return {'status':'completed','calls':0,'actions':0,'provenance':'offline_fixture'}
    def move_to(self,*args,**kwargs):
        self.budget=(kwargs['max_calls'],kwargs['deadline'])
        try:self.updated=kwargs['target_update'](kwargs['before_decision']())
        except ValueError:return {'status':'stopped','reason':'perception_rejected','calls':0,'actions':0}
        return {'status':'stopped','reason':'test_stop','calls':1,'actions':0}

def fixture(failures=0):
    observer=Observer(failures);driver=Driver()
    executor=VisualManipulationExecutor(simulation=Sim(),observer=observer,driver=driver,clock=lambda:0)
    executor._point=lambda snap,obj:SimpleNamespace(xyz=[snap['x'],-.047,.02],covariance_m2=[[1e-8,0,0],[0,1e-8,0],[0,0,1e-8]])
    executor._check_path=lambda *args:None
    return executor,observer,driver
PLAN={'action':'pick','object_id':'red','target_id':None}

def test_fresh_image_updates_recipe_target_and_preserves_outer_budget():
    executor,observer,driver=fixture()
    result=executor.bounded_execute(PLAN,max_calls=7,deadline=20)
    assert driver.updated['target_xyz']==[.193,-.047,.06]
    assert driver.budget==(7,20)
    assert result['verified'] is False and result['calls']==1

def test_transient_loss_has_one_stationary_retry_and_explicit_event():
    executor,observer,driver=fixture(1);events=[]
    result=executor.execute(PLAN,event_callback=events.append)
    assert observer.observations==2 and result['observation_retries']==1
    assert [e['phase'] for e in events].count('observation_recovery')==1
    assert result['actions']==0 and driver.updated is not None

def test_persistent_loss_never_substitutes_last_known_point():
    executor,observer,driver=fixture(3)
    result=executor.execute(PLAN)
    assert observer.observations==2 and driver.updated is None
    assert result['reason']=='perception_rejected' and not result['verified']

def test_stop_during_recovery_prevents_second_capture():
    executor,observer,driver=fixture(1);cancel=threading.Event()
    def event(e):
        if e['phase']=='observation_recovery':cancel.set()
    result=executor.execute(PLAN,cancel_event=cancel,event_callback=event)
    assert observer.observations==1 and driver.updated is None and not result['verified']

def test_experimental_mode_rejects_non_simulator_backend():
    import pytest
    with pytest.raises(ValueError,match='simulation_only'):
        VisualManipulationExecutor(simulation=Sim(),observer=Observer(),driver=Driver(),experimental_simulation=True)

def test_exact_simulator_experimental_factory_keeps_validation_truthful():
    from embodied_arm.simulation.so101_simulation import SO101Simulation
    from embodied_arm.control.manipulation import simulation_task_executor
    from tests.test_jev_motion import FakeJev
    from embodied_arm.services.task_service import TaskService
    from tests.test_task_service import Brain,Source
    sim=SO101Simulation(seed=0)
    try:
        source=Source();source.sim=sim
        executor=simulation_task_executor(sim,source,FakeJev())
        assert executor.experimental_simulation and not executor.validated
        state=TaskService(scene_source=source,brain=Brain(),executor=executor).status()
        assert state['execution_available'] and state['execution_validation']=='experimental_simulation'
    finally:sim.close()


def test_applied_micro_history_is_retained_when_post_motion_perception_fails():
    executor,observer,driver=fixture()
    def failed_move(*args,**kwargs):
        callback=kwargs['event_callback']
        callback({'phase':'jev_decision','action':'x_plus','calls':1,'actions':0})
        callback({'phase':'motion_applied','applied':True,'visually_verified':False,'calls':1,'actions':1})
        return {'status':'stopped','reason':'perception_rejected','calls':1,'actions':1,'provenance':'offline_fixture'}
    driver.move_to=failed_move
    result=executor.execute(PLAN)
    assert not result['verified'] and result['actions']==1
    assert result['recent_actions']==[{'action':'x_plus','applied':True}]


def test_recovery_reports_active_driver_counts_and_original_fault():
    executor,observer,driver=fixture();events=[]
    driver.gripper=lambda *a,**k:{'status':'completed','calls':1,'actions':0,'provenance':'offline_fixture'}
    def move(*args,**kwargs):
        callback=kwargs['event_callback']
        callback({'phase':'jev_decision','action':'x_minus','calls':3,'actions':2})
        callback({'phase':'motion_applied','applied':True,'calls':3,'actions':3})
        failures=iter(['Static RGB association lost: tracking_refinement_failed','Moving RGB anchors unavailable'])
        def fail():
            observer.last_error=next(failures)
            raise ValueError(observer.last_error)
        observer.observe=fail
        try:kwargs['after_motion']()
        except ValueError:pass
        return {'status':'stopped','reason':'perception_rejected','calls':3,'actions':3,'provenance':'offline_fixture'}
    driver.move_to=move
    result=executor.execute(PLAN,event_callback=events.append)
    recovery=next(e for e in events if e['phase']=='observation_recovery')
    assert (recovery['calls'],recovery['actions'])==(4,3)
    assert recovery['first_error']=='Static RGB association lost: tracking_refinement_failed'
    assert result['perception_detail']==recovery['first_error']
    assert result['recovery_detail']=='Moving RGB anchors unavailable'
    assert (result['calls'],result['actions'])==(4,3)


def test_cycle_progress_records_measured_error_and_changed_goal_without_verdict():
    from embodied_arm.providers.brain import validate_cycle_progress
    executor,observer,driver=fixture()
    def move(*args,**kwargs):
        callback=kwargs['event_callback']
        for goal,error in (([.193,-.047,.06],[-.02,0,0]),([.194,-.047,.06],[-.01,0,0])):
            kwargs['before_decision']()
            callback({'phase':'observation','kind':'cartesian','goal':goal,'error_xyz_m':error,'calls':1,'actions':0})
        return {'status':'stopped','reason':'test_stop','calls':1,'actions':0}
    driver.move_to=move
    result=executor.execute(PLAN)
    progress=validate_cycle_progress(result['progress'])
    assert len(progress)==1 and progress[0]['observations']==2
    assert progress[0]['goal_start_xyz_m']!=progress[0]['goal_end_xyz_m']
    assert progress[0]['error_start_xyz_m']==[-.02,0,0]
    assert progress[0]['error_end_xyz_m']==[-.01,0,0]
    assert 'verified' not in progress[0] and result['verified'] is False
