"""Actual rendered RGB -> fake brain/Jev -> real robot safety boundary.

This test proves the default profile fails closed at its measured precision
floor. It is not a claim that image-only picking works.
"""
import time
from types import SimpleNamespace
from embodied_arm.simulation.so101_simulation import SO101Simulation
from embodied_arm.control.so101_control import CameraSceneSource
from embodied_arm.perception.phase_observer import PhaseSceneObserver
from embodied_arm.control.manipulation import VisualManipulationExecutor
from embodied_arm.control.jev_motion import JevMotionDriver
from embodied_arm.services.task_service import TaskService
from tests.rgb_coarse_fixture import RGBColourFixture
from tests.test_jev_motion import FakeJev

class FakeBrain:
    def status(self):return {'available':True}
    def interpret(self,text,scene,history):
        assert set(scene['blocks'])=={'red','blue'}
        return SimpleNamespace(accepted=True,provenance='offline_fake_brain',plan=SimpleNamespace(
            to_dict=lambda:{'action':'pick','object_id':'red','target_id':None,'message':'离线测试：拿起红块'}))
    def cancel(self):pass

def test_default_rgb_precision_blocks_before_physical_motion(monkeypatch):
    sim=SO101Simulation(seed=0)
    try:
        monkeypatch.setattr(sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('Oracle forbidden')))
        source=CameraSceneSource(sim,vision=RGBColourFixture(),refine_rgb=True)
        observer=PhaseSceneObserver(source)
        observer.validated=True # Test-only injection, not advertised product capability.
        jev=FakeJev();driver=JevMotionDriver(sim,jev,sleep=lambda _:sim.tick(24))
        executor=VisualManipulationExecutor(simulation=sim,observer=observer,driver=driver)
        task=TaskService(scene_source=source,brain=FakeBrain(),executor=executor)
        task.submit('夹起红色方块')
        deadline=time.monotonic()+60
        while task.status()['pending'] and time.monotonic()<deadline:time.sleep(.02)
        status=task.status()
        assert status['state']=='failed'
        assert status['provenance']=='offline_fake_brain'
        assert sim.proprioception()['actions']==0
        assert all(r['target']['kind']=='gripper' for r in jev.requests)
        assert status['outcome']['reason']=='precision_floor'
        assert status['outcome']['verified'] is False
    finally:sim.close()
