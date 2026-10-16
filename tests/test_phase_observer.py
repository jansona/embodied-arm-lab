"""Rendered-RGB integration; detector is an explicit offline color fixture."""
import pytest
from embodied_arm.simulation.so101_simulation import SO101Simulation
from embodied_arm.control.so101_control import CameraSceneSource
from embodied_arm.perception.phase_observer import PhaseSceneObserver
from tests.rgb_coarse_fixture import RGBColourFixture

def test_actual_rgb_phase_initialization_never_reads_oracle(monkeypatch):
    sim=SO101Simulation(seed=0)
    try:
        def denied(*args,**kwargs):raise AssertionError('Oracle access forbidden')
        monkeypatch.setattr(sim,'evaluation_state',denied)
        source=CameraSceneSource(sim,vision=RGBColourFixture(),refine_rgb=True)
        source.refresh();observer=PhaseSceneObserver(source)
        first=observer.begin_phase('approach','red');second=observer.observe()
        assert first['scene']['frame_ids']!=second['scene']['frame_ids']
        assert first['visual_evidence']['identity_token']==second['visual_evidence']['identity_token']
        assert second['robot']['actions']==0
        assert len(second['visual_evidence']['detections'])==4
        assert not observer.validated # physical manipulation evidence still pending
    finally:sim.close()

def test_shadow_free_rgb_v1_high_resolution_roi_meets_precision_floor(monkeypatch):
    from embodied_arm.perception.so101_uncertainty import axis_bands
    sim=SO101Simulation(seed=0,camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    try:
        monkeypatch.setattr(sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('Oracle forbidden')))
        source=CameraSceneSource(sim,vision=RGBColourFixture(),refine_rgb=True,refinement_profile='scaled',roi_tracking=True)
        source.refresh();observer=PhaseSceneObserver(source)
        current=observer.begin_phase('approach','red')
        for name in ('red','blue'):
            covariance=current['scene']['objects'][name]['covariance_m2']
            assert all(x['status']=='inside' for x in axis_bands([0,0,0],covariance))
        assert current['robot']['actions']==0
    finally:sim.close()

def test_goal_memory_recorded_before_slow_tracker_setup(monkeypatch):
    """CPU setup time must not invalidate evidence before it is recorded."""
    from types import SimpleNamespace
    from tests.test_landmark_memory import scene
    from tests.test_camera_perception import rgb_frame,calibration
    from embodied_arm.perception.camera_perception import PixelDetection
    now=[100.2];value=scene()
    frames=tuple(rgb_frame(calibration(cid,[0,0,0]),when=100.,frame_id=fid)
                 for cid,fid in value.frame_ids.items())
    detections=tuple(PixelDetection('red',f.camera_id,f.frame_id,(100.,100.),.9,2.) for f in frames)
    snapshot={'source':'rgb_only','scene':value.to_dict(),'robot':{'epoch':7}}
    source=SimpleNamespace(sim=SimpleNamespace(camera_profile='legacy_fixed_stereo'),
        track=lambda:None,observation_evidence=lambda:(snapshot,frames,detections),
        semantic_deadline=160.,refinement_profile='baseline')
    class Tracker:
        def __init__(self,**kwargs):pass
        def start(self,*args,**kwargs):return SimpleNamespace(accepted=True)
    monkeypatch.setattr('embodied_arm.perception.phase_observer.RGBRegionTracker',Tracker)
    observer=PhaseSceneObserver(source,clock=lambda:now[0])
    def slow_setup(required):now[0]+=5.
    monkeypatch.setattr(observer,'_start_static',slow_setup)
    monkeypatch.setattr(observer,'observe',lambda:observer.latest)
    observer.begin_phase('approach','red')
    record=observer.goal_memory.read({'epoch':7},frames,now=now[0])
    assert record['captured_at']==100. and now[0]>103.
    assert record['estimate']['covariance_m2']==snapshot['scene']['objects']['goal']['covariance_m2']


def test_symbolic_stack_target_is_normalized_before_held_validation():
    from types import SimpleNamespace
    observer=PhaseSceneObserver(SimpleNamespace(sim=SimpleNamespace()))
    # A missing hold proof still fails, but a supported symbolic target is valid.
    with pytest.raises(ValueError,match='Verified held'):
        observer.begin_step('place','blue',target_id='on:red')
    with pytest.raises(ValueError,match='Invalid placement support'):
        observer.begin_step('place','red',target_id='on:red')
