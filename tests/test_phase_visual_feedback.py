"""Synthetic contract fixtures; these tests do not claim physical grasp success."""
from dataclasses import replace
import pytest
from camera_perception import EstimatedScene, ObjectEstimate, PixelDetection, LANDMARKS
from embodied_arm.visual_feedback import VisualObservation, VisualPhaseVerifier


def observation(t=1., z=.02, tcp_z=.02, sigma=.0005, **kwargs):
    objects = {k: ObjectEstimate(k, 'unknown', LANDMARKS[k], reason='not_required') for k in LANDMARKS}
    objects['red'] = ObjectEstimate('red', 'estimated', 'top_face_center', (.1, .1, z),
        ((sigma*sigma, 0., 0.), (0., sigma*sigma, 0.), (0., 0., sigma*sigma)), sigma,
        .95, 0., 0., 45.)
    scene = EstimatedScene(t, t, {k: f'{k}{t}' for k in ('left', 'right')}, objects)
    detections = tuple(PixelDetection('red', cam, frame, (100., 100.), .95, 2.) for cam, frame in scene.frame_ids.items())
    return replace(VisualObservation(scene, detections, (.1, .1, tcp_z), t, 1, 'lift1',
        'red-initial-stereo-identity', 1, .95, 'dynamic_rgb_track', 1., .0005), **kwargs)


def lift(before=None):
    verifier = VisualPhaseVerifier()
    verifier.begin_lift('red', before or observation(), commanded_tcp_delta=(0., 0., .04), now=1.)
    return verifier


def test_observed_rise_and_preserved_tcp_offset_accept():
    result = lift().update(observation(2., .06, .06), now=2.)
    assert result.verified and not result.physical_contact_verified
    assert result.uncertainty_bound_m == pytest.approx(.006)


@pytest.mark.parametrize('obs,reason', [
    (observation(2., .02, .06), 'lift_or_offset_not_verified'),
    (observation(2., .04, .06), 'lift_or_offset_not_verified'),
    (observation(2., .06, .06, identity_token='other'), 'phase_or_identity_changed'),
    (observation(2., .06, .06, epoch=2), 'phase_or_identity_changed'),
    (observation(), 'replayed_rgb_frame'),
])
def test_no_gripper_or_motion_only_success(obs, reason):
    result = lift().update(obs, now=2.)
    assert not result.verified and result.reason == reason


def test_stale_rejected():
    result = lift().update(observation(2., .06, .06), now=6.)
    assert not result.verified and result.reason == 'stale_rgb'


def test_large_covariance_cannot_relax_offset_gate():
    first = observation(sigma=.004)
    result = lift(first).update(observation(2., .06, .06, sigma=.004), now=2.)
    assert not result.verified


def test_ambiguous_local_identity_requests_one_refresh():
    verifier = lift()
    result = verifier.update(observation(2., .06, .06, association_candidates=2), now=2.)
    assert result.refresh_required and not result.verified
    refreshed = observation(3., .06, .06, source='semantic_rgb', semantic_captured_at=3.)
    assert verifier.update(refreshed, now=3.).verified


def test_ambiguous_semantic_refresh_fails_closed():
    verifier = lift()
    verifier.update(observation(2., .06, .06, association_candidates=2), now=2.)
    result = verifier.update(observation(3., .06, .06, source='semantic_rgb',
                                      semantic_captured_at=3., association_candidates=2), now=3.)
    assert not result.verified and result.terminal and not result.refresh_required


def test_pixel_frame_provenance_mismatch_rejected():
    obs = observation(2., .06, .06)
    obs = replace(obs, detections=(replace(obs.detections[0], frame_id='old'), obs.detections[1]))
    result = lift().update(obs, now=2.)
    assert not result.verified and result.reason == 'missing_or_mismatched_rgb_evidence'


def release():
    verifier = VisualPhaseVerifier()
    verifier.begin_release('red', observation(), expected_landmark_xyz=(.1, .1, .02),
                           target_std_m=.0005, now=1.)
    return verifier


def test_release_requires_stationary_object_after_tcp_retreat():
    assert release().update(observation(2., .02, .07), now=2.).verified
    assert not release().update(observation(2., .07, .07), now=2.).verified
    assert not release().update(observation(2., .02, .02), now=2.).verified
    assert not release().update(observation(2., .035, .07), now=2.).verified


def test_release_requires_dwell():
    assert not release().update(observation(1.05, .02, .07), now=1.05).verified


def dynamic_fixture(shift=0, t=1., duplicate=False):
    from camera_perception import CameraCalibration, RGBFrame
    from test_rgb_face_refinement import image_bytes, TOP
    raw = image_bytes(top=[(x+shift, y) for x, y in TOP], duplicate=duplicate)
    frames = []
    for cam in ('left', 'right'):
        calibration = CameraCalibration(cam, 640, 480, 500., 500.,
            316.-500*.1/1.02, 212.-500*.1/1.02,
            ((1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,-1.),(0.,0.,0.,1.)))
        frames.append(RGBFrame.from_jpeg(f'{cam}{t}', t, calibration, raw))
    return tuple(frames)


def test_dynamic_rgb_tracks_real_shift_beyond_static_four_pixel_gate():
    from embodied_arm.visual_feedback import DynamicRGBPhaseTracker
    first = observation()
    first = replace(first, detections=tuple(replace(d, uv=(316.,212.)) for d in first.detections))
    tracker = DynamicRGBPhaseTracker()
    tracker.start(first, dynamic_fixture(), object_id='red', now=1.)
    result = tracker.track(dynamic_fixture(10, 2.), tcp_xyz=(.1204,.1,.02), tcp_captured_at=2., now=2.)
    assert result.accepted, result.reason
    assert all(d.uv[0] > 324. for d in result.detections)
    assert result.identity_token == first.identity_token


def test_dynamic_rgb_does_not_return_encoder_prediction_on_slip():
    from embodied_arm.visual_feedback import DynamicRGBPhaseTracker
    first = observation()
    first = replace(first, detections=tuple(replace(d, uv=(316.,212.)) for d in first.detections))
    tracker = DynamicRGBPhaseTracker()
    tracker.start(first, dynamic_fixture(), object_id='red', now=1.)
    result = tracker.track(dynamic_fixture(0, 2.), tcp_xyz=(.1408,.1,.02), tcp_captured_at=2., now=2.)
    assert not result.accepted and not result.detections and result.refresh_required


def test_dynamic_rgb_rejects_duplicate_same_colour_object():
    from embodied_arm.visual_feedback import DynamicRGBPhaseTracker
    first = observation()
    first = replace(first, detections=tuple(replace(d, uv=(316.,212.)) for d in first.detections))
    tracker = DynamicRGBPhaseTracker()
    tracker.start(first, dynamic_fixture(), object_id='red', now=1.)
    result = tracker.track(dynamic_fixture(0, 2., duplicate=True), tcp_xyz=(.14284,.1,.02), tcp_captured_at=2., now=2.)
    assert not result.accepted and result.refresh_required


def test_closure_rejects_replayed_frames_and_does_not_claim_grasp():
    from embodied_arm.visual_feedback import DynamicRGBPhaseTracker
    first=observation()
    first=replace(first,detections=tuple(replace(d,uv=(316.,212.)) for d in first.detections))
    tracker=DynamicRGBPhaseTracker()
    tracker.start(first,dynamic_fixture(),object_id='red',now=1.)
    result=tracker.reacquire_closure(dynamic_fixture(),tcp_xyz=first.tcp_xyz,
        tcp_captured_at=1.,now=1.,workspace=((-1,1),(-1,1),(-1,1)))
    assert not result.accepted and result.reason=='replayed_rgb_frame'
    assert not result.physical_contact_verified


def test_closure_rejects_transport_disguised_as_jaw_closing():
    from embodied_arm.visual_feedback import DynamicRGBPhaseTracker
    first=observation()
    first=replace(first,detections=tuple(replace(d,uv=(316.,212.)) for d in first.detections))
    tracker=DynamicRGBPhaseTracker()
    tracker.start(first,dynamic_fixture(),object_id='red',now=1.)
    result=tracker.reacquire_closure(dynamic_fixture(0,2.),tcp_xyz=(.14,.1,.02),
        tcp_captured_at=2.,now=2.,workspace=((-1,1),(-1,1),(-1,1)))
    assert not result.accepted and result.reason=='closure_tcp_moved'
