"""On-demand semantic context; fixtures, no network or simulator oracle."""
from types import SimpleNamespace
import pytest
from embodied_arm.control.so101_control import CameraSceneSource

class Vision:
    def __init__(self,ready=True):self.ready=ready
    def status(self):return {'available':self.ready}

def source(ready=True):
    s=CameraSceneSource(SimpleNamespace(),vision=Vision(ready))
    s.valid=False;s.refreshes=0;s.tracks=0
    def scene():
        if not s.valid:raise ValueError('No valid image')
        return {'perception_source':'rgb_only'}
    def refresh(**kwargs):
        if kwargs.get('expected_generation',s.generation)!=s.generation:raise ValueError('cancelled')
        s.refreshes+=1;s.valid=True
    def track():s.tracks+=1;s.valid=True
    s.brain_scene=scene;s.refresh=refresh;s.track=track
    return s

def test_initial_command_acquires_scene_once_then_reuses_valid_image():
    s=source()
    assert s.ensure_brain_scene()['perception_source']=='rgb_only'
    assert s.ensure_brain_scene()['perception_source']=='rgb_only'
    assert s.refreshes==1

def test_local_tracking_refreshes_stale_evidence_without_model_call():
    s=source();s.tracker=object()
    assert s.ensure_brain_scene()['perception_source']=='rgb_only'
    assert s.tracks==1 and s.refreshes==0

def test_missing_worker_is_explicit_then_reconnect_can_recover():
    s=source(False)
    with pytest.raises(ValueError,match='vision_unavailable'):s.ensure_brain_scene()
    assert s.refreshes==0
    s.vision.ready=True
    assert s.ensure_brain_scene() and s.refreshes==1

def test_cancelled_tracking_never_falls_back_to_paid_refresh():
    s=source();s.tracker=object()
    def cancelled():s.generation+=1;raise ValueError('cancelled')
    s.track=cancelled
    with pytest.raises(ValueError,match='cancelled'):s.ensure_brain_scene()
    assert s.refreshes==0

@pytest.mark.parametrize('reason',['network_error','timeout','landmark_uncertain','rgb_face_refinement_rejected',
 'triangulation_rejected','frame_mismatch','invalid_response','vision_rejected','private detail'])
def test_failure_is_allowlisted_and_does_not_reach_brain(reason):
    s=source()
    def fail(**kwargs):s.reason=reason;raise RuntimeError('Never expose this body')
    s.refresh=fail
    with pytest.raises(ValueError) as error:s.ensure_brain_scene()
    assert str(error.value)==(reason if reason!='private detail' else 'observation_unavailable')

def test_stop_during_provider_status_prevents_paid_refresh():
 s=source()
 def status():s.generation+=1;return {'available':True}
 s.vision.status=status
 with pytest.raises(ValueError,match='cancelled'):s.ensure_brain_scene()
 assert s.refreshes==0

def test_stop_during_capture_prevents_vision_submission():
 s=CameraSceneSource(SimpleNamespace(),vision=Vision())
 s.sim.proprioception=lambda:{'name':'SO-ARM101','arm_dof':5,'epoch':1,'actions':0,'paused':True,'busy':False,'ee':[.2,0,.1],
     'joints':[0]*5,'joint_limits':[],'gripper_openness':1,'gripper_angle':0,'workspace':[],'cube_edge_m':.02}
 calls=[];captures=[]
 s.vision.perceive=lambda packet:calls.append(packet)
 def capture():
  captures.append(True);s.cancel()
  return {'frame_id':'frame','captured_at':1,'images':[]}
 s.sim.capture_frames=capture
 with pytest.raises(ValueError):s.ensure_brain_scene()
 assert captures==[True] and calls==[]


def test_full_frame_source_declares_bounded_processing_age_without_roi_switch():
    import inspect
    s=CameraSceneSource(SimpleNamespace(),max_observation_age_s=30.)
    assert s.status()['max_observation_age_s']==30.
    assert 'roi_tracking' not in inspect.signature(CameraSceneSource).parameters
    assert 'observed_roi_bounds' not in s.status()
    for age in (0,61,float('nan'),True,None,'30'):
        with pytest.raises(ValueError,match='age budget'):
            CameraSceneSource(SimpleNamespace(),max_observation_age_s=age)
