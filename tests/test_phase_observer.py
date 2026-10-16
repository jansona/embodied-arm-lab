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


def test_hold_reacquisition_preserves_original_semantic_capture(monkeypatch):
    from types import SimpleNamespace
    original={'scene':{'frame_ids':{'left':'semantic','right':'semantic'},'captured_at':100.},
              'robot':{'epoch':7,'actions':3}}
    source=SimpleNamespace(sim=SimpleNamespace(),last_vision={
        'fresh_model_request':True,'base_detector_provenance':'qwen_rgb'},
        observation_evidence=lambda:(original,(),()))
    observer=PhaseSceneObserver(source)
    fresh={'scene':{'frame_ids':{'left':'tracked','right':'tracked'},'captured_at':102.},
           'robot':{'epoch':7,'actions':3},'visual_evidence':{
               'semantic_captured_at':100.,'identity_token':'new','source':'dynamic_rgb_track'}}
    def begin(phase,obj):
        assert observer.required_objects==('red','blue','goal')
        observer.phase=phase;observer.object_id=obj;observer.identity_token='new'
        observer.latest=fresh
        return fresh
    monkeypatch.setattr(observer,'begin_phase',begin)
    monkeypatch.setattr(observer,'snapshot',lambda:fresh)
    result=observer.prepare_hold_reacquisition('red')
    record=result['visual_evidence']['semantic_reacquisition']
    assert record['captured_at']==100. and record['frame_ids']['left']=='semantic'
    assert result['visual_evidence']['source']=='dynamic_rgb_track'
    cert={'object_id':'red','epoch':7,'actions':3,'frame_ids':fresh['scene']['frame_ids'],
          'identity_token':'new','verified_lift':True,'source':'phase_scoped_temporal_rgb'}
    with pytest.raises(ValueError,match='certificate'):
        observer.adopt_verified_hold(dict(cert,actions=4))
    observer.adopt_verified_hold(cert)
    with pytest.raises(ValueError,match='certificate'):observer.adopt_verified_hold(cert)


def test_hold_reacquisition_requires_new_real_semantic_request():
    from types import SimpleNamespace
    for provenance in ({'fresh_model_request':False,'base_detector_provenance':'qwen_rgb'},
                       {'fresh_model_request':True,'base_detector_provenance':'offline_fixture'}):
        observer=PhaseSceneObserver(SimpleNamespace(sim=SimpleNamespace(),last_vision=provenance))
        with pytest.raises(ValueError,match='Fresh semantic'):
            observer.prepare_hold_reacquisition('red')


def test_first_rgb_fault_survives_secondary_retry_error(monkeypatch):
    from types import SimpleNamespace
    sim=SO101Simulation(seed=0)
    try:
        source=CameraSceneSource(sim,vision=RGBColourFixture(),refine_rgb=True)
        source.refresh();observer=PhaseSceneObserver(source)
        observer.begin_phase('approach','red')
        detail={'object_id':'blue','camera_id':'right','reason':'roi_clipped',
                'roi_bounds':[100,100,200,200],'uv':None}
        observer.static.last_failure_details=[detail]
        monkeypatch.setattr(observer.static,'track',lambda *a,**k:SimpleNamespace(
            accepted=False,reason='tracking_refinement_failed'))
        with pytest.raises(ValueError,match='tracking_refinement_failed'):observer.observe()
        first=observer.last_failure_details
        observer.static.last_failure_details=[]
        monkeypatch.setattr(observer.static,'track',lambda *a,**k:SimpleNamespace(
            accepted=False,reason='no_active_tracking_session'))
        with pytest.raises(ValueError,match='no_active_tracking_session'):observer.observe()
        assert observer.last_failure_details is first
        assert first['fits']==[detail]
        observer.cancel();assert observer.last_failure_details is None
    finally:sim.close()
