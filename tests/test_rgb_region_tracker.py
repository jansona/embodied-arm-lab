"""Offline RGB lease/geometry tests, no model calls or simulation dependencies."""
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from embodied_arm.perception.camera_perception import PixelDetection
from embodied_arm.perception.rgb_face_refinement import FaceRefinement, RefinedPacket, SOURCE
import embodied_arm.perception.rgb_region_tracker as tracker_module
from embodied_arm.perception.rgb_region_tracker import RGBRegionTracker, TRACKED_SOURCE


def jpeg_url():
    output = BytesIO(); Image.new('RGB', (640, 480), (200, 210, 220)).save(output, 'JPEG')
    return 'data:image/jpeg;base64,'+base64.b64encode(output.getvalue()).decode()


URL = jpeg_url()


def packet(fid='initial', stamp=100.):
    return {'frame_id':fid,'captured_at':stamp,'images':[
        {'camera_id':camera,'width':640,'height':480,'data_url':URL} for camera in ('left','right')]}


def detections(fid='initial'):
    return tuple(PixelDetection(obj,camera,fid,(200.+i*30,210.+j*40),.98,2.,'visible',
                                'goal_center' if obj=='goal' else 'top_face_center')
                 for i,obj in enumerate(('red','blue','goal')) for j,camera in enumerate(('left','right')))


def accepted_result(seeds, *, offset=(0.,0.)):
    output = tuple(replace(item,uv=(item.uv[0]+offset[0],item.uv[1]+offset[1]))
                   if item.visibility == 'visible' else item for item in seeds)
    evidence=tuple(FaceRefinement(item.object_id,True,'refined_visible_face',item.uv,(0,0,640,480),
                                 item.uv,1.,((0.,0.),(10.,0.),(10.,10.),(0.,10.)),.2,100.,1.,50.,(10,10,10,10))
                   for item in output if item.visibility == 'visible')
    return RefinedPacket(all(item.visibility == 'visible' for item in output),
                         'all_visible_faces_refined',output,evidence)


@pytest.fixture
def fake_refiner(monkeypatch):
    calls=[]
    def refine(frame,seeds):
        calls.append((frame,tuple(seeds)))
        return accepted_result(seeds)
    monkeypatch.setattr(tracker_module,'refine_packet',refine)
    return calls


def started(fake_refiner, *, stamp=100., now=100., epoch=7):
    tracker=RGBRegionTracker()
    result=tracker.start(packet(stamp=stamp),detections(),epoch,now=now)
    assert result.accepted
    return tracker


def assert_cleared(tracker,result,reason):
    assert not result.accepted and result.reason==reason
    assert result.detections==()
    assert not tracker.status()['active']
    assert tracker.track(packet('later',102.),7,now=102.).reason=='no_active_semantic_session'


def test_starts_original_semantic_packet_after_qwen_latency_but_never_refreshes_age(fake_refiner):
    tracker=started(fake_refiner,now=118.)
    status=tracker.status()
    assert status['semantic_deadline']==160.
    assert status['semantic_frame_id']=='initial'
    assert status['semantic_captured_at']==100.
    assert status['tracked_frames']==0
    result=tracker.track(packet('fresh',118.),7,now=118.1)
    assert result.accepted and result.source==TRACKED_SOURCE
    assert all(item.source==TRACKED_SOURCE for item in result.evidence)
    assert all(item.frame_id=='fresh' for item in result.detections)
    assert tracker.status()['semantic_deadline']==160.
    assert tracker.status()['tracked_frames']==1
    json.dumps(result.to_dict(),allow_nan=False)


def test_all_tracking_seeds_are_initial_refined_anchors_not_prior_result(monkeypatch):
    calls=[]
    def refine(frame,seeds):
        calls.append(tuple(seeds))
        # Initial refinement corrects AI coarse pixels by two pixels; subsequent
        # RGB results vary by one. Seeds must remain at the initial correction.
        return accepted_result(seeds,offset=(2.,0.) if len(calls)==1 else (1.,0.))
    monkeypatch.setattr(tracker_module,'refine_packet',refine)
    tracker=RGBRegionTracker()
    tracker.start(packet(),detections(),7,now=100.)
    first=tracker.track(packet('f1',101.),7,now=101.)
    second=tracker.track(packet('f2',102.),7,now=102.)
    assert first.accepted and second.accepted
    assert calls[1][0].uv==(202.,210.)
    assert calls[2][0].uv==(202.,210.)
    assert first.detections[0].uv==second.detections[0].uv==(203.,210.)


def test_drift_accumulation_measured_against_initial_not_previous(monkeypatch):
    sequence=iter([0.,3.,4.01])
    def refine(frame,seeds):
        return accepted_result(seeds,offset=(next(sequence),0.))
    monkeypatch.setattr(tracker_module,'refine_packet',refine)
    tracker=RGBRegionTracker(); tracker.start(packet(),detections(),7,now=100.)
    assert tracker.track(packet('f1',101.),7,now=101.).accepted
    result=tracker.track(packet('f2',102.),7,now=102.)
    assert_cleared(tracker,result,'initial_anchor_drift')


def test_euclidean_four_pixel_drift_limit(monkeypatch):
    calls=0
    def refine(frame,seeds):
        nonlocal calls
        calls+=1
        return accepted_result(seeds,offset=(0.,0.) if calls==1 else (3.,3.))
    monkeypatch.setattr(tracker_module,'refine_packet',refine)
    tracker=RGBRegionTracker(); tracker.start(packet(),detections(),7,now=100.)
    assert_cleared(tracker,tracker.track(packet('f1',101.),7,now=101.),'initial_anchor_drift')


def test_exactly_four_pixel_drift_is_within_limit(monkeypatch):
    calls=0
    def refine(frame,seeds):
        nonlocal calls
        calls+=1
        return accepted_result(seeds,offset=(0.,0.) if calls==1 else (4.,0.))
    monkeypatch.setattr(tracker_module,'refine_packet',refine)
    tracker=RGBRegionTracker(); tracker.start(packet(),detections(),7,now=100.)
    assert tracker.track(packet('f1',101.),7,now=101.).accepted


@pytest.mark.parametrize('frame,now,epoch,reason', [
    (packet('f1',101.),101.,8,'epoch_changed'),
    (packet('f1',101.),101.,True,'epoch_changed'),
    (packet('f1',101.),104.001,7,'stale_current_frame'),
    (packet('f1',102.),101.,7,'future_frame'),
    (packet('initial',101.),101.,7,'reused_frame_id'),
    (packet('f1',100.),101.,7,'nonincreasing_capture_time'),
    (packet('f1',99.),101.,7,'nonincreasing_capture_time'),
    (packet('f1',160.),160.,7,'semantic_expired'),
])
def test_guard_failure_clears_session(fake_refiner,frame,now,epoch,reason):
    tracker=started(fake_refiner)
    assert_cleared(tracker,tracker.track(frame,epoch,now=now),reason)
    assert len(fake_refiner)==1  # Guard fails before any RGB refinement.


def test_frame_age_exactly_three_seconds_allowed(fake_refiner):
    tracker=started(fake_refiner)
    assert tracker.track(packet('f1',101.),7,now=104.).accepted


def test_any_prior_frame_id_reuse_is_rejected_not_only_last(fake_refiner):
    tracker=started(fake_refiner)
    assert tracker.track(packet('f1',101.),7,now=101.).accepted
    assert tracker.track(packet('f2',102.),7,now=102.).accepted
    assert_cleared(tracker,tracker.track(packet('f1',103.),7,now=103.),'reused_frame_id')


def test_tracking_budget_thirty_frames_never_renews_semantic_lease(fake_refiner):
    tracker=started(fake_refiner)
    for n in range(1,31):
        assert tracker.track(packet(f'f{n}',100.+n),7,now=100.+n).accepted
    assert tracker.status()['tracked_frames']==30
    assert tracker.status()['semantic_deadline']==160.
    assert_cleared(tracker,tracker.track(packet('f31',131.),7,now=131.),'tracking_frame_budget')
    assert len(fake_refiner)==31  # One initial plus thirty tracked, no 32nd call.


def test_dimensions_are_bound_to_initial_images(fake_refiner):
    tracker=started(fake_refiner)
    frame=packet('f1',101.);frame['images'][0]['width']=800
    assert_cleared(tracker,tracker.track(frame,7,now=101.),'camera_layout_changed')


def test_camera_order_may_change_but_ids_cannot(fake_refiner):
    tracker=started(fake_refiner)
    frame=packet('f1',101.);frame['images'].reverse()
    assert tracker.track(frame,7,now=101.).accepted
    bad=packet('f2',102.);bad['images'][0]['camera_id']='new_camera'
    assert_cleared(tracker,tracker.track(bad,7,now=102.),'invalid_packet')


@pytest.mark.parametrize('extra',['depth','segmentation','object_xyz','calibration','body_id'])
def test_extra_state_channels_are_refused(fake_refiner,extra):
    tracker=started(fake_refiner)
    frame=packet('f1',101.);frame[extra]=[1,2,3]
    assert_cleared(tracker,tracker.track(frame,7,now=101.),'invalid_packet')


def test_nonvisible_or_ambiguous_geometry_clears_without_partial_fallback(fake_refiner,monkeypatch):
    tracker=started(fake_refiner)
    def reject(frame,seeds):
        valid=accepted_result(seeds)
        missing=replace(valid.detections[0],uv=None,pixel_sigma_px=None,visibility='unknown',confidence=0.)
        return replace(valid,accepted=False,reason='refinement_failed_closed',detections=(missing,*valid.detections[1:]))
    monkeypatch.setattr(tracker_module,'refine_packet',reject)
    result=tracker.track(packet('f1',101.),7,now=101.)
    assert_cleared(tracker,result,'tracking_refinement_failed')
    assert result.evidence  # Audit evidence preserved, actuatable pixels removed.


def test_refiner_error_clears_session(fake_refiner,monkeypatch):
    tracker=started(fake_refiner)
    def error(*args):
        raise ValueError('invalid image')
    monkeypatch.setattr(tracker_module,'refine_packet',error)
    assert_cleared(tracker,tracker.track(packet('f1',101.),7,now=101.),'tracking_refinement_error')


def test_start_failure_also_replaces_prior_active_session(fake_refiner):
    tracker=started(fake_refiner)
    result=tracker.start(packet('new',90.),detections('new'),7,now=160.)
    assert_cleared(tracker,result,'semantic_expired')


def test_expiry_and_current_freshness_rechecked_after_actual_refinement(fake_refiner):
    times=iter([100.,100.,159.9,160.01])
    tracker=RGBRegionTracker(clock=lambda:next(times))
    assert tracker.start(packet(),detections(),7).accepted
    assert_cleared(tracker,tracker.track(packet('fresh',159.9),7),'semantic_expired')
    times=iter([100.,100.,103.9,104.01])
    tracker=RGBRegionTracker(clock=lambda:next(times))
    assert tracker.start(packet(),detections(),7).accepted
    assert_cleared(tracker,tracker.track(packet('fresh',101.),7),'stale_current_frame')


def test_explicit_clear_prevents_automatic_restart(fake_refiner):
    tracker=started(fake_refiner)
    tracker.clear()
    assert_cleared(tracker,tracker.track(packet('fresh',101.),7,now=101.),'no_active_semantic_session')


def test_packet_and_status_copies_cannot_mutate_fixed_anchors(fake_refiner):
    frame=packet();tracker=RGBRegionTracker();tracker.start(frame,detections(),7,now=100.)
    frame['images'][0]['width']=1234
    status=tracker.status();status['semantic_deadline']=9999
    assert tracker.track(packet('fresh',101.),7,now=101.).accepted
    assert tracker.status()['semantic_deadline']==160.


@pytest.mark.parametrize('scope',[('red','blue','goal'),('red',),('red','blue')])
def test_saved_rgb_can_track_without_qwen_or_world_state(scope):
    root=Path(__file__).resolve().parents[1]/'artifacts'
    files=[root/'so101-vision-third-diagnostic.json',root/'so101-live-left.jpg',root/'so101-live-right.jpg']
    if not all(path.exists() for path in files):
        pytest.skip('Optional fixed RGB experiment fixture unavailable')
    diagnostic=json.loads(files[0].read_text())
    frame=packet()
    for item,path in zip(frame['images'],files[1:]):
        item['data_url']='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()
    ai=tuple(PixelDetection(item['object_id'],item['camera_id'],'initial',
                           (item['uv_1000'][0]*639/1000,item['uv_1000'][1]*479/1000),
                           item['confidence'],2.,'visible',item['landmark']) for item in diagnostic['detections'])
    tracker=RGBRegionTracker()
    initial=tracker.start(frame,ai,7,now=118.,required_objects=scope)
    assert initial.accepted
    fresh={**frame,'frame_id':'fresh-rgb','captured_at':118.}
    tracked=tracker.track(fresh,7,now=118.1)
    assert tracked.accepted,tracked.reason
    assert tracked.source==TRACKED_SOURCE
    assert len(tracked.detections)==2*len(scope)
    assert {item.object_id for item in tracked.detections}==set(scope)
    for actual, initial_item in zip(tracked.detections, initial.detections):
        assert actual.uv == pytest.approx(initial_item.uv, abs=1e-9)


def test_backwards_analysis_clock_cannot_extend_the_lease(fake_refiner):
    tracker=started(fake_refiner)
    assert tracker.track(packet('f1',101.),7,now=103.).accepted
    result=tracker.track(packet('f2',102.),7,now=102.)
    assert_cleared(tracker,result,'clock_moved_backwards')


@pytest.mark.parametrize('required',[(), [], ('yellow',), ('red','red'), 'red', ('red',True)])
def test_invalid_required_scope_cannot_start(fake_refiner,required):
    tracker=RGBRegionTracker()
    result=tracker.start(packet(),detections(),7,now=100.,required_objects=required)
    assert_cleared(tracker,result,'invalid_required_objects')
    assert not fake_refiner


def test_required_red_succeeds_when_irrelevant_goal_failed_with_only_two_outputs(monkeypatch):
    calls=[]
    def refine(frame,seeds):
        calls.append(tuple(seeds))
        result=accepted_result(seeds)
        changed=tuple(replace(item,uv=None,pixel_sigma_px=None,visibility='unknown',confidence=0.)
                      if item.object_id=='goal' else item for item in result.detections)
        evidence=tuple(replace(item,accepted=False,reason='occluded') if item.object_id=='goal' else item
                       for item in result.evidence)
        return replace(result,accepted=False,detections=changed,evidence=evidence)
    monkeypatch.setattr(tracker_module,'refine_packet',refine)
    tracker=RGBRegionTracker()
    initial=tracker.start(packet(),detections(),7,now=100.,required_objects=('red',))
    assert initial.accepted and len(initial.detections)==2
    result=tracker.track(packet('fresh',101.),7,now=101.)
    assert result.accepted and len(result.detections)==2
    assert {item.object_id for item in result.detections}=={'red'}
    assert all(item.frame_id=='fresh' for item in result.detections)
    assert len(result.evidence)==2
    assert tracker.status()['required_objects']==['red']
    assert all(item.uv is None and item.visibility=='unknown' for item in calls[1] if item.object_id!='red')


def test_required_target_failure_still_clears_scoped_lease(fake_refiner,monkeypatch):
    tracker=RGBRegionTracker()
    assert tracker.start(packet(),detections(),7,now=100.,required_objects=('red',)).accepted
    def fail_red(frame,seeds):
        result=accepted_result(seeds)
        output=tuple(replace(item,uv=None,pixel_sigma_px=None,visibility='unknown',confidence=0.)
                     if item.object_id=='red' and item.camera_id=='left' else item for item in result.detections)
        return replace(result,accepted=False,detections=output)
    monkeypatch.setattr(tracker_module,'refine_packet',fail_red)
    assert_cleared(tracker,tracker.track(packet('fresh',101.),7,now=101.),'tracking_refinement_failed')


def test_scope_immutable_and_cannot_expand_during_track(fake_refiner):
    scope=['red']
    tracker=RGBRegionTracker()
    assert tracker.start(packet(),detections(),7,now=100.,required_objects=scope).accepted
    scope.append('blue')
    status=tracker.status();status['required_objects'].append('goal')
    assert tracker.status()['required_objects']==['red']
    with pytest.raises(TypeError):
        tracker.track(packet('fresh',101.),7,now=101.,required_objects=('red','blue'))
    result=tracker.track(packet('fresh',101.),7,now=101.)
    assert result.accepted and {item.object_id for item in result.detections}=={'red'}


def test_default_scope_still_rejects_irrelevant_goal_failure(fake_refiner,monkeypatch):
    tracker=started(fake_refiner)
    def fail_goal(frame,seeds):
        result=accepted_result(seeds)
        output=tuple(replace(item,uv=None,pixel_sigma_px=None,visibility='unknown',confidence=0.)
                     if item.object_id=='goal' else item for item in result.detections)
        return replace(result,accepted=False,detections=output)
    monkeypatch.setattr(tracker_module,'refine_packet',fail_goal)
    assert_cleared(tracker,tracker.track(packet('fresh',101.),7,now=101.),'tracking_refinement_failed')


def test_red_hover_scope_requires_fresh_blue_obstacle_too(fake_refiner,monkeypatch):
    tracker=RGBRegionTracker()
    initial=tracker.start(packet(),detections(),7,now=100.,required_objects=('red','blue'))
    assert initial.accepted and len(initial.detections)==4
    def fail_blue(frame,seeds):
        result=accepted_result(seeds)
        output=tuple(replace(item,uv=None,pixel_sigma_px=None,visibility='unknown',confidence=0.)
                     if item.object_id=='blue' and item.camera_id=='right' else item for item in result.detections)
        return replace(result,accepted=False,detections=output)
    monkeypatch.setattr(tracker_module,'refine_packet',fail_blue)
    assert_cleared(tracker,tracker.track(packet('fresh',101.),7,now=101.),'tracking_refinement_failed')
