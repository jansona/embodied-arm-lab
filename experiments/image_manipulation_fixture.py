"""Offline acceptance probe: rendered RGB, fake semantic/Jev providers, real physics.

No simulator object poses are read. This is not live Qwen/Jev validation.
Run from repository root: python -m experiments.image_manipulation_fixture
"""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from so101_simulation import SO101Simulation
from so101_control import CameraSceneSource
from rgb_coarse_fixture import RGBColourFixture
from test_jev_motion import FakeJev
from embodied_arm.jev_motion import JevMotionDriver
from embodied_arm.phase_observer import PhaseSceneObserver
from embodied_arm.manipulation import VisualManipulationExecutor

def run():
    sim=SO101Simulation(seed=0,camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    try:
        sim.evaluation_state=lambda:(_ for _ in ()).throw(AssertionError('Oracle forbidden'))
        source=CameraSceneSource(sim,vision=RGBColourFixture(),refine_rgb=True,refinement_profile='scaled',roi_tracking=True)
        source.refresh();observer=PhaseSceneObserver(source)
        observer.validated=True # Offline acceptance injection, never a production default.
        controller=FakeJev();driver=JevMotionDriver(sim,controller,sleep=lambda _:sim.tick(24))
        executor=VisualManipulationExecutor(simulation=sim,observer=observer,driver=driver)
        try:
            outcome=executor.execute({'action':'pick','object_id':'red','target_id':None},event_callback=lambda event:print(json.dumps(event),flush=True))
        except Exception as exc:
            outcome={'verified':False,'failure':type(exc).__name__,'reason':str(exc),'calls':len(controller.requests),'actions':sim.proprioception()['actions']}
        if outcome.get('verified') is not True:
            import base64
            for frame in getattr(observer,'last_captured_frames',()):
                Path('/tmp/robot-rgb-error-'+frame.camera_id+'.jpg').write_bytes(base64.b64decode(frame.jpeg_base64))
            outcome['rgb_drift']=getattr(observer.static,'last_drift',None)
        print(json.dumps({'provenance':'offline_RGB_fixture_fake_Jev_real_physics','outcome':outcome}),flush=True)
        return outcome
    finally:sim.close()
if __name__=='__main__':run()
