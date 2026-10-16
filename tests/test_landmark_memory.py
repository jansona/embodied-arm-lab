"""Fast synthetic RGB-estimate contracts; no renderer, simulator, or models."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from camera_perception import EstimatedScene, ObjectEstimate
from embodied_arm.landmark_memory import FixedGoalMemory


class CalibrationFixture:
    def __init__(self,camera_id):
        self.data={'camera_id':camera_id,'width':640,'height':480,'fx':500.,'fy':500.,
                   'cx':319.5,'cy':239.5,'world_from_camera':[
                       [1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]}
    def to_dict(self):return deepcopy(self.data)


def scene(*,captured_at=100.,xyz=(.17,.105,.001),visible=True):
    unknown=lambda name:ObjectEstimate(name,'unknown','top_face_center',reason='occluded')
    goal=(ObjectEstimate('goal','estimated','goal_center',xyz=xyz,
                        covariance_m2=((1e-6,0.,0.),(0.,4e-6,0.),(0.,0.,9e-6)),
                        position_std_m=.003,confidence=.91,reprojection_error_px=.7,
                        ray_separation_m=.0003,ray_angle_deg=40.) if visible else
          ObjectEstimate('goal','unknown','goal_center',reason='occluded'))
    return EstimatedScene(captured_at,captured_at+.05,{'left':'rgb_left_100','right':'rgb_right_100'},
                          {'red':unknown('red'),'blue':unknown('blue'),'goal':goal})


@pytest.fixture
def observation():
    value=scene()
    snapshot={'source':'rgb_only','scene':value.to_dict(),'robot':{'epoch':7}}
    frames=[SimpleNamespace(camera_id=cid,frame_id=value.frame_ids[cid],captured_at=100.,
                            calibration=CalibrationFixture(cid)) for cid in ('left','right')]
    return snapshot,frames


def test_original_measurement_provenance_covariance_frames_and_age_preserved(observation):
    snapshot,frames=observation;memory=FixedGoalMemory()
    memory.remember(snapshot,frames,now=100.2)
    result=memory.read({'epoch':7},frames,now=200.)
    assert result['source']=='prior_rgb_measurement'
    assert result['assumption']=='marker_printed_on_fixed_bench'
    assert result['estimate']==snapshot['scene']['objects']['goal']
    assert result['estimate']['covariance_m2']==[[1e-6,0.,0.],[0.,4e-6,0.],[0.,0.,9e-6]]
    assert result['captured_at']==100. and result['frame_ids']==snapshot['scene']['frame_ids']
    assert result['calibrations']=={f.camera_id:f.calibration.to_dict() for f in frames}
    assert result['epoch']==7
    # Reading prior evidence never restamps it or silently reduces uncertainty.
    assert memory.read({'epoch':7},frames,now=219.9)==result


@pytest.mark.parametrize('source',[None,'ground_truth','evaluation-oracle-v1','calibrated_depth'])
def test_non_rgb_snapshot_provenance_rejected(observation,source):
    snapshot,frames=observation;snapshot['source']=source
    memory=FixedGoalMemory()
    with pytest.raises(ValueError,match='RGB source'):memory.remember(snapshot,frames,now=100.2)
    assert memory.record is None


def test_scene_provenance_cannot_be_relabelled_at_outer_boundary(observation):
    snapshot,frames=observation;snapshot['scene']['source']='ground_truth'
    with pytest.raises(ValueError,match='provenance'):
        FixedGoalMemory().remember(snapshot,frames,now=100.2)


@pytest.mark.parametrize('now',[99.99,103.01])
def test_remember_requires_fresh_nonfuture_measurement(observation,now):
    snapshot,frames=observation
    with pytest.raises(ValueError,match='stale or future'):
        FixedGoalMemory().remember(snapshot,frames,now=now)


def test_unknown_goal_cannot_seed_memory(observation):
    snapshot,frames=observation;snapshot['scene']=scene(visible=False).to_dict()
    with pytest.raises(ValueError,match='not safely estimated'):
        FixedGoalMemory().remember(snapshot,frames,now=100.2)


@pytest.mark.parametrize('now',[99.99,220.001])
def test_expiry_and_future_clock_reject_cached_marker(observation,now):
    snapshot,frames=observation;memory=FixedGoalMemory()
    memory.remember(snapshot,frames,now=100.2)
    with pytest.raises(ValueError,match='expired'):memory.read({'epoch':7},frames,now=now)


def test_custom_lease_and_inclusive_expiry_boundary(observation):
    snapshot,frames=observation;memory=FixedGoalMemory(max_age_s=10.)
    memory.remember(snapshot,frames,now=100.2)
    assert memory.read({'epoch':7},frames,now=110.)['captured_at']==100.
    with pytest.raises(ValueError,match='expired'):memory.read({'epoch':7},frames,now=110.001)


@pytest.mark.parametrize('age',[0.,-1.,120.01,float('inf'),float('nan')])
def test_invalid_lease_is_rejected(age):
    with pytest.raises(ValueError):FixedGoalMemory(max_age_s=age)


def test_reset_epoch_invalidates_memory_even_if_marker_is_static(observation):
    snapshot,frames=observation;memory=FixedGoalMemory();memory.remember(snapshot,frames,now=100.2)
    with pytest.raises(ValueError,match='reset'):memory.read({'epoch':8},frames,now=101.)


@pytest.mark.parametrize('kind',['intrinsics','extrinsics','missing_camera','different_camera'])
def test_calibration_changes_invalidate_memory(observation,kind):
    snapshot,frames=observation;memory=FixedGoalMemory();memory.remember(snapshot,frames,now=100.2)
    if kind=='intrinsics':frames[0].calibration.data['fx']+=1
    elif kind=='extrinsics':frames[0].calibration.data['world_from_camera'][0][3]+=.001
    elif kind=='missing_camera':frames.pop()
    else:frames[0].camera_id='replacement'
    with pytest.raises(ValueError,match='calibration'):memory.read({'epoch':7},frames,now=101.)


def test_frame_order_does_not_invalidate_identical_calibrations(observation):
    snapshot,frames=observation;memory=FixedGoalMemory();memory.remember(snapshot,frames,now=100.2)
    assert memory.read({'epoch':7},list(reversed(frames)),now=101.)['frame_ids']==snapshot['scene']['frame_ids']


def test_visible_drift_invalidates_and_clears_memory(observation):
    snapshot,frames=observation;memory=FixedGoalMemory();memory.remember(snapshot,frames,now=100.2)
    moved=scene(captured_at=101.,xyz=(.174,.105,.001))
    with pytest.raises(ValueError,match='marker moved'):memory.check_visible(moved,now=101.1)
    assert memory.record is None
    with pytest.raises(ValueError,match='expired'):memory.read({'epoch':7},frames,now=101.2)


def test_small_visible_drift_does_not_rewrite_historical_evidence(observation):
    snapshot,frames=observation;memory=FixedGoalMemory();memory.remember(snapshot,frames,now=100.2)
    before=memory.read({'epoch':7},frames,now=100.3)
    memory.check_visible(scene(captured_at=101.,xyz=(.172,.105,.001)),now=101.1)
    assert memory.read({'epoch':7},frames,now=101.2)==before


def test_read_returns_independent_copy_and_remember_does_not_alias_snapshot(observation):
    snapshot,frames=observation;memory=FixedGoalMemory();memory.remember(snapshot,frames,now=100.2)
    result=memory.read({'epoch':7},frames,now=101.)
    result['estimate']['xyz'][0]=999
    result['estimate']['covariance_m2'][0][0]=0
    result['calibrations']['left']['fx']=1
    snapshot['scene']['objects']['goal']['xyz'][0]=888
    snapshot['scene']['frame_ids']['left']='tampered'
    actual=memory.read({'epoch':7},frames,now=101.)
    assert actual['estimate']['xyz'][0]==.17
    assert actual['estimate']['covariance_m2'][0][0]==1e-6
    assert actual['calibrations']['left']['fx']==500.
    assert actual['frame_ids']['left']=='rgb_left_100'


def test_clear_is_idempotent_and_removes_usable_memory(observation):
    snapshot,frames=observation;memory=FixedGoalMemory();memory.remember(snapshot,frames,now=100.2)
    memory.clear();memory.clear()
    assert memory.record is None
    with pytest.raises(ValueError,match='expired'):memory.read({'epoch':7},frames,now=101.)
    memory.check_visible(scene(),now=100.2)  # Empty memory is harmless.

@pytest.mark.parametrize('field,value',[('frame_id','other_frame'),('captured_at',100.1),('camera_id','other_camera')])
def test_mismatched_frame_evidence_cannot_seed_memory(observation,field,value):
 snapshot,frames=observation;setattr(frames[0],field,value)
 with pytest.raises(ValueError,match='frame association'):
  FixedGoalMemory().remember(snapshot,frames,now=100.2)
