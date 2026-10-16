import base64
from io import BytesIO
from types import SimpleNamespace
import time
from PIL import Image
import pytest
from camera_perception import CameraCalibration,PixelDetection
from so101_simulation import camera_calibration
from so101_control import CameraSceneSource,ground_symbolic

class SafeFake:
    def __init__(self):
        self.epoch=1;self.actions=0;self.paused=True
        self.now=time.monotonic()
    def evaluation_state(self):raise AssertionError('Forbidden oracle access')
    def state(self):raise AssertionError('Legacy GT state forbidden')
    def proprioception(self):
        return dict(name='SO-ARM101',arm_dof=5,ee=[.2,0,.06],joints=[0]*5,joint_limits=[[-2,2]]*5,
                    gripper_openness=1.,gripper_angle=1.,paused=self.paused,busy=False,epoch=self.epoch,
                    actions=self.actions,workspace=[[.07,.33],[-.19,.19],[.008,.33]],cube_edge_m=.02,
                    ground_truth_objects={'SECRET_ORACLE':[999,999,999]})
    def capture_frames(self):
        out=BytesIO();Image.new('RGB',(640,480),(100,100,100)).save(out,format='JPEG')
        return dict(frame_id='frame1',captured_at=self.now,
            images=[dict(camera_id=k,width=640,height=480,data_url='data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()) for k in ['left','right']],
            calibrations=[camera_calibration(k) for k in ['left','right']],ground_truth_objects={'SECRET_ORACLE':[999,999,999]})

class Vision:
    def __init__(self,sim,mutation=None):self.sim=sim;self.mutation=mutation;self.received=None
    def perceive(self,packet):
        self.received=packet
        detections=[]
        # Synthetic known points are test fixtures, not simulator inputs.
        for camera_id in ('left','right'):
            calibration=CameraCalibration.from_dict(camera_calibration(camera_id))
            for name,xyz in [('red',[.19,-.05,.02]),('blue',[.25,.055,.02]),('goal',[.17,.105,.001])]:
                detections.append(PixelDetection(name,camera_id,packet['frame_id'],calibration.project(xyz),.99,2.,'visible','goal_center' if name=='goal' else 'top_face_center'))
        if self.mutation:self.mutation(self.sim)
        return SimpleNamespace(accepted=True,frame_id=packet['frame_id'],captured_at=packet['captured_at'],detections=detections)

def test_only_rgb_and_robot_encoders_cross_boundary():
    s=SafeFake();v=Vision(s);source=CameraSceneSource(s,v,clock=lambda:s.now+.1)
    result=source.refresh()
    assert set(v.received)=={'frame_id','captured_at','images'}
    assert 'SECRET_ORACLE' not in repr(result)
    assert 'ground_truth' not in repr(result)
    assert source.brain_scene()['blocks']['red']==pytest.approx([.19,-.05,.01],abs=1e-8)
    assert source.brain_scene()['goal']==pytest.approx([.17,.105,.011],abs=1e-8)
    assert ground_symbolic({'action':'pick','object_id':'red'},result)['source']=='rgb_only'

@pytest.mark.parametrize('change',[lambda s:setattr(s,'epoch',2),lambda s:setattr(s,'actions',1),lambda s:setattr(s,'paused',False)])
def test_changed_scene_rejects_response(change):
    s=SafeFake();source=CameraSceneSource(s,Vision(s,change),clock=lambda:s.now+.1)
    with pytest.raises(ValueError):source.refresh()
    assert source.status()['scene'] is None

def test_motion_invalidates_estimate_and_no_stale_fallback():
    s=SafeFake();source=CameraSceneSource(s,Vision(s),clock=lambda:s.now+.1);source.refresh()
    s.actions+=1
    with pytest.raises(ValueError):source.snapshot()
    source.cancel()
    with pytest.raises(ValueError):source.brain_scene()

def test_frozen_window_is_bounded_and_requires_paused_scene():
    s=SafeFake();source=CameraSceneSource(s,Vision(s),clock=lambda:s.now+31)
    with pytest.raises(ValueError):source.refresh()
    s.paused=False
    with pytest.raises(ValueError):source.refresh()

def test_ground_truth_schema_never_accepted_by_target_resolver():
    with pytest.raises(ValueError):ground_symbolic({'action':'pick','object_id':'red'},{'schema':'evaluation-oracle-v1','ground_truth_objects':{'red':[.2,0,.02]}})

def test_hybrid_receives_only_rgb_and_canonical_detections(monkeypatch):
    import rgb_face_refinement
    seen={}
    def refine(packet,detections):
        seen.update(packet=packet,detections=detections)
        return SimpleNamespace(accepted=True,reason='refined',source='hybrid_qwen_region_rgb_geometry',detections=detections,evidence=())
    import rgb_region_tracker
    class Tracker:
        def start(self,packet,detections,epoch,now=None,required_objects=None):return refine(packet,detections)
        def clear(self):pass
        def status(self):return {}
    monkeypatch.setattr(rgb_region_tracker,'RGBRegionTracker',Tracker)
    s=SafeFake();source=CameraSceneSource(s,Vision(s),clock=lambda:s.now+.1,refine_rgb=True)
    result=source.refresh()
    assert set(seen['packet'])=={'frame_id','captured_at','images'}
    assert 'ground_truth' not in repr(seen) and 'SECRET_ORACLE' not in repr(seen)
    assert result['detector_provenance']=='offline_fixture_rgb_geometry'
    assert source.status()['last_vision']['base_detector_provenance']=='unverified_fixture'

def test_failed_hybrid_has_no_coarse_detection_fallback(monkeypatch):
    import rgb_face_refinement
    import rgb_region_tracker
    class Tracker:
        def start(self,*args,**kwargs):return SimpleNamespace(accepted=False,reason='ambiguous',source='hybrid_qwen_region_rgb_geometry',detections=(),evidence=())
        def clear(self):pass
        def status(self):return {}
    monkeypatch.setattr(rgb_region_tracker,'RGBRegionTracker',Tracker)
    s=SafeFake();source=CameraSceneSource(s,Vision(s),clock=lambda:s.now+.1,refine_rgb=True)
    with pytest.raises(ValueError):source.refresh()
    assert source.status()['scene'] is None
    assert source.status()['reason']=='rgb_face_refinement_rejected'

def test_partial_scope_cannot_feed_brain_or_symbolic_plan():
    sim=SafeFake();source=CameraSceneSource(sim,Vision(sim))
    snapshot=source.refresh(required_objects=('red','blue'))
    with pytest.raises(ValueError,match='Partial scene'):source.brain_scene()
    with pytest.raises(ValueError,match='Partial scene'):ground_symbolic({'action':'pick','object_id':'red'},snapshot)

def test_required_blue_unknown_stops_scoped_scene():
    sim=SafeFake()
    class MissingBlue(Vision):
        def perceive(self,packet):
            result=super().perceive(packet);result.detections=[x for x in result.detections if x.object_id!='blue'];return result
    source=CameraSceneSource(sim,MissingBlue(sim))
    with pytest.raises(ValueError):source.refresh(required_objects=('red','blue'))
    assert source.scene is None
