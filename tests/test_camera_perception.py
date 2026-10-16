"""Offline tests use only mathematical projections and generated RGB JPEGs."""
from dataclasses import replace
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image
import pytest

from embodied_arm.perception.camera_perception import (CameraCalibration, RGBFrame, PixelDetection,
                               PerceptionConfig, ObjectEstimate, EstimatedScene,
                               estimate_scene, _closest_point)

BOUNDS = ((-2, 2), (-2, 2), (-2, 3))
NOW = 100.


def calibration(name, origin, rotation=None, focal=600):
    transform = np.eye(4)
    transform[:3, 3] = origin
    if rotation is not None:
        transform[:3, :3] = rotation
    return CameraCalibration(name, 640, 480, focal, focal, 319.5, 239.5,
                             transform.tolist())


def rgb_frame(calib, *, when=NOW, frame_id='frame-1'):
    out = BytesIO()
    Image.new('RGB', (calib.width, calib.height), (30, 40, 50)).save(out, 'JPEG')
    return RGBFrame.from_jpeg(frame_id, when, calib, out.getvalue())


def stereo(baseline=.6, when=NOW):
    return [rgb_frame(calibration('left', [-baseline/2, 0, 0]), when=when),
            rgb_frame(calibration('right', [baseline/2, 0, 0]), when=when)]


def detections(frames, point=(0, 0, 1), *, object_id='red', sigma=2., confidence=.9):
    return [PixelDetection(object_id, f.camera_id, f.frame_id, f.calibration.project(point),
                           confidence, sigma, landmark='goal_center' if object_id == 'goal' else 'top_face_center')
            for f in frames]


def result(frames, items, *, now=NOW, config=None, workspace=BOUNDS, object_id='red'):
    return estimate_scene(frames, items, now=now, workspace=workspace,
                          config=config).objects[object_id]


def test_exact_synthetic_projection_recovers_landmark_and_covariance():
    frames = stereo()
    point = (.03, -.07, 1.05)
    estimate = result(frames, detections(frames, point))
    assert estimate.valid
    assert estimate.landmark == 'top_face_center'
    np.testing.assert_allclose(estimate.xyz, point, atol=1e-12)
    assert 0 < estimate.position_std_m < .015
    assert estimate.reprojection_error_px < 1e-10
    assert estimate.ray_separation_m < 1e-12
    assert np.all(np.linalg.eigvalsh(estimate.covariance_m2) > 0)


def test_rotated_tabletop_cameras_recover_full_xyz_without_table_height():
    point = np.array([.22, .025, .086])  # An elevated landmark, not a table assumption.
    frames = []
    for name, origin in [('left', [.1, -.3, .35]), ('right', [.1, .3, .35])]:
        direction = np.array([.22, 0, .04]) - origin
        direction /= np.linalg.norm(direction)
        right = np.cross(direction, [0, 0, 1]); right /= np.linalg.norm(right)
        down = np.cross(direction, right)
        frames.append(rgb_frame(calibration(name, origin, np.column_stack((right, down, direction)))))
    estimate = result(frames, detections(frames, tuple(float(v) for v in point)))
    assert estimate.valid
    np.testing.assert_allclose(estimate.xyz, point, atol=1e-12)


def test_scene_keeps_explicit_unknowns_and_goal_landmark_semantics():
    frames = stereo()
    scene = estimate_scene(frames, detections(frames, object_id='goal'), now=NOW, workspace=BOUNDS)
    assert scene.objects['goal'].valid
    assert scene.objects['goal'].landmark == 'goal_center'
    for name in ('red', 'blue'):
        assert scene.objects[name].status == 'unknown'
        assert scene.objects[name].reason == 'missing_view'
        assert scene.objects[name].xyz is None


def test_complete_scene_round_trips_strict_finite_json():
    frames = stereo()
    items = sum([detections(frames, object_id=name) for name in ('red', 'blue', 'goal')], [])
    scene = estimate_scene(frames, items, now=NOW, workspace=BOUNDS)
    for frame in frames:
        assert RGBFrame.from_dict(json.loads(json.dumps(frame.to_dict()))).to_dict() == frame.to_dict()
    for item in items:
        assert PixelDetection.from_dict(item.to_dict()) == item
    payload = json.loads(json.dumps(scene.to_dict(), allow_nan=False))
    assert EstimatedScene.from_dict(payload).to_dict() == payload
    assert scene.require('red', now=NOW+.5).valid
    with pytest.raises(TypeError):
        scene.objects['red'] = scene.objects['blue']


@pytest.mark.parametrize('age,expected', [(3.01, 'stale_frame'), (-.03, 'future_frame')])
def test_stale_and_future_frames_are_unknown(age, expected):
    frames = stereo(when=NOW-age)
    estimate = result(frames, detections(frames))
    assert estimate.reason == expected
    assert estimate.xyz is None


def test_unsynchronized_frames_are_rejected():
    frames = stereo()
    frames[1] = replace(frames[1], captured_at=NOW-.051)
    assert result(frames, detections(frames)).reason == 'unsynchronized_frames'


def test_cached_estimate_rechecks_freshness_and_unknown_at_use_time():
    frames = stereo()
    scene = estimate_scene(frames, detections(frames), now=NOW, workspace=BOUNDS)
    for when in (NOW+4, NOW-.1):
        with pytest.raises(ValueError, match='stale or future'):
            scene.require('red', now=when)
    with pytest.raises(ValueError, match='not safely estimated'):
        scene.require('blue', now=NOW)


@pytest.mark.parametrize('visibility', ['occluded', 'absent', 'ambiguous', 'unknown'])
def test_nonvisible_landmark_never_gets_xyz_or_table_fallback(visibility):
    frames = stereo()
    items = detections(frames)
    items[1] = replace(items[1], visibility=visibility, uv=None, pixel_sigma_px=None, confidence=0.)
    estimate = result(frames, items)
    assert estimate.reason == 'landmark_not_visible'
    assert estimate.xyz is None and estimate.covariance_m2 is None
    assert PixelDetection.from_dict(items[1].to_dict()) == items[1]


def test_single_view_is_unknown_even_when_goal_is_planar():
    frames = stereo()
    estimate = result(frames, detections(frames, object_id='goal')[:1], object_id='goal')
    assert estimate.reason == 'missing_view'
    assert estimate.xyz is None


def test_duplicate_matching_feature_is_ambiguous_even_if_one_is_confident():
    frames = stereo()
    items = detections(frames)
    estimate = result(frames, items + [replace(items[0], confidence=.1)])
    assert estimate.reason == 'ambiguous_correspondence'


@pytest.mark.parametrize('field,value', [('frame_id', 'old-frame'), ('camera_id', 'unknown-camera')])
def test_detection_frame_provenance_must_match(field, value):
    frames = stereo()
    items = detections(frames)
    items[0] = replace(items[0], **{field: value})
    assert result(frames, items).reason == 'frame_mismatch'


def test_low_confidence_is_fail_closed_and_not_a_probability_of_safety():
    frames = stereo()
    assert result(frames, detections(frames, confidence=.74)).reason == 'low_confidence'


@pytest.mark.parametrize('uv', [(-1., 200.), (640., 200.), (200., 480.), (200., -1.)])
def test_pixel_bounds_are_checked_before_geometry(uv):
    frames = stereo()
    items = detections(frames)
    items[0] = replace(items[0], uv=uv)
    assert result(frames, items).reason == 'pixel_out_of_bounds'


def test_parallel_rays_do_not_return_a_fabricated_depth():
    frames = stereo()
    items = [replace(item, uv=(319.5, 239.5)) for item in detections(frames)]
    estimate = result(frames, items)
    assert estimate.reason == 'parallel_rays'
    assert estimate.xyz is None


def test_near_parallel_stereo_is_rejected():
    frames = stereo(baseline=.05)
    assert result(frames, detections(frames)).reason == 'poor_stereo_angle'


def test_reverse_disparity_is_behind_camera():
    frames = stereo()
    items = detections(frames)
    items = [replace(items[0], uv=items[1].uv), replace(items[1], uv=items[0].uv)]
    assert result(frames, items).reason == 'behind_camera'


def test_inconsistent_rays_and_reprojection_are_separate_guards():
    frames = stereo()
    items = detections(frames)
    items[1] = replace(items[1], uv=(items[1].uv[0], items[1].uv[1]+20))
    assert result(frames, items).reason == 'inconsistent_rays'
    cfg = PerceptionConfig(max_ray_separation_m=.05)
    assert result(frames, items, config=cfg).reason == 'reprojection_error'


def test_3d_observation_bounds_are_enforced():
    frames = stereo()
    assert result(frames, detections(frames), workspace=((-2, 2), (-2, 2), (0, .9))).reason == 'outside_observation_workspace'


def test_confident_but_uncertain_pixels_are_rejected():
    frames = stereo()
    assert result(frames, detections(frames, sigma=20, confidence=1)).reason == 'excessive_3d_uncertainty'


def test_pixel_uncertainty_propagates_to_metric_covariance():
    frames = stereo()
    config = PerceptionConfig(max_position_std_m=.05)
    low = result(frames, detections(frames, sigma=2), config=config)
    high = result(frames, detections(frames, sigma=4), config=config)
    assert low.valid and high.valid
    np.testing.assert_allclose(high.covariance_m2, np.asarray(low.covariance_m2)*4, rtol=1e-8, atol=1e-14)
    assert high.position_std_m == pytest.approx(low.position_std_m*2)
    floor = result(frames, detections(frames, sigma=.1), config=config)
    np.testing.assert_allclose(floor.covariance_m2, low.covariance_m2, atol=1e-14)


def test_depth_error_grows_with_distance_and_shrinking_baseline():
    config = PerceptionConfig(max_position_std_m=.05)
    frames = stereo()
    near = result(frames, detections(frames, (0, 0, 1)), config=config)
    far = result(frames, detections(frames, (0, 0, 2)), config=config)
    narrow_frames = stereo(baseline=.3)
    narrow = result(narrow_frames, detections(narrow_frames), config=config)
    assert near.valid and far.valid and narrow.valid
    assert far.covariance_m2[2][2] == pytest.approx(near.covariance_m2[2][2]*16, rel=1e-6)
    assert narrow.covariance_m2[2][2] == pytest.approx(near.covariance_m2[2][2]*4, rel=1e-6)


def test_propagated_covariance_matches_seeded_pixel_noise_experiment():
    frames = stereo()
    items = detections(frames)
    estimate = result(frames, items)
    pixels = np.array([v for item in items for v in item.uv])
    rng = np.random.default_rng(42)
    points = [_closest_point([frame.calibration for frame in frames], pixels+rng.normal(0, 2, 4))[0]
              for _ in range(2000)]
    empirical = np.cov(np.array(points).T)
    np.testing.assert_allclose(np.diag(empirical), np.diag(estimate.covariance_m2), rtol=.10)


@pytest.mark.parametrize('field,value', [('fx', 0), ('fx', True), ('width', 640.),
    ('cx', float('nan')), ('world_from_camera', np.diag([1., 1., -1., 1.]).tolist()),
    ('world_from_camera', np.diag([1., 2., 1., 1.]).tolist())])
def test_invalid_intrinsics_or_nonrigid_extrinsics_rejected(field, value):
    value_dict = stereo()[0].calibration.to_dict()
    value_dict[field] = value
    with pytest.raises(ValueError):
        CameraCalibration.from_dict(value_dict)


def test_calibration_inputs_are_defensively_copied():
    transform = np.eye(4).tolist()
    calib = CameraCalibration('left', 640, 480, 600., 600., 320., 240., transform)
    transform[0][3] = 999
    assert calib.world_from_camera[0][3] == 0


@pytest.mark.parametrize('extra', ['depth', 'segmentation', 'body_id', 'object_xyz'])
def test_contracts_reject_unrecognized_ground_truth_channels(extra):
    frame = stereo()[0]
    frame_payload = frame.to_dict()
    frame_payload[extra] = [1, 2, 3]
    with pytest.raises(ValueError, match='fields'):
        RGBFrame.from_dict(frame_payload)
    detection_payload = detections(stereo())[0].to_dict()
    detection_payload[extra] = [1, 2, 3]
    with pytest.raises(ValueError, match='fields'):
        PixelDetection.from_dict(detection_payload)


@pytest.mark.parametrize('field,value', [('confidence', True), ('confidence', math.inf),
    ('uv', [1, math.nan]), ('pixel_sigma_px', 0), ('pixel_sigma_px', None),
    ('object_id', 'body-3'), ('landmark', 'bbox_center')])
def test_malformed_visible_detections_rejected(field, value):
    payload = detections(stereo())[0].to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        PixelDetection.from_dict(payload)


def test_nonvisible_contract_cannot_retain_a_pixel_guess():
    item = detections(stereo())[0]
    with pytest.raises(ValueError, match='Non-visible'):
        replace(item, visibility='occluded')


def test_invalid_jpeg_and_wrong_dimensions_and_non_rgb_are_rejected():
    frame = stereo()[0]
    with pytest.raises(ValueError, match='JPEG'):
        replace(frame, jpeg_base64='bm90LWEtanBlZw==')
    for mode, size, fmt in [('RGB', (320, 240), 'JPEG'), ('L', (640, 480), 'JPEG'), ('RGB', (640, 480), 'PNG')]:
        out = BytesIO(); Image.new(mode, size).save(out, fmt)
        with pytest.raises(ValueError, match='RGB JPEG'):
            RGBFrame.from_jpeg('frame', NOW, frame.calibration, out.getvalue())


def test_unknown_estimate_cannot_smuggle_coordinates():
    with pytest.raises(ValueError, match='Unknown landmark'):
        ObjectEstimate('red', 'unknown', 'top_face_center', xyz=(0, 0, 0), reason='occluded')


def test_estimate_from_dict_rejects_inconsistent_covariance_scale():
    frames = stereo()
    payload = result(frames, detections(frames)).to_dict()
    payload['position_std_m'] *= .1
    with pytest.raises(ValueError, match='must match'):
        ObjectEstimate.from_dict(payload)


def test_exactly_two_distinct_cameras_required():
    frames = stereo()
    for bad in (frames[:1], frames+frames[:1], [frames[0], frames[0]]):
        with pytest.raises(ValueError, match='Exactly two'):
            estimate_scene(bad, [], now=NOW, workspace=BOUNDS)
