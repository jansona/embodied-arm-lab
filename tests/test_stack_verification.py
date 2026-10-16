"""Synthetic RGB contracts and path geometry; no physics or contact claims."""
from types import SimpleNamespace

import pytest

from embodied_arm.control.manipulation import VisualManipulationExecutor
from embodied_arm.control.stack_verification import verify_stack_release
from embodied_arm.perception.camera_perception import EstimatedScene, LANDMARKS, ObjectEstimate, PixelDetection


def estimate(name, xyz, sigma=.0005, confidence=.95):
    covariance = tuple(tuple(sigma * sigma if i == j else 0. for j in range(3)) for i in range(3))
    return ObjectEstimate(name, 'estimated', LANDMARKS[name], tuple(xyz), covariance,
                          sigma, confidence, 0., 0., 45.)


def snapshot(t, *, carried=(.2, 0., .04), support=(.2, 0., .02), tcp=(.2, 0., .03)):
    objects = {name: ObjectEstimate(name, 'unknown', LANDMARKS[name], reason='not_required')
               for name in LANDMARKS}
    objects['red'] = estimate('red', support)
    objects['blue'] = estimate('blue', carried)
    frames = {camera: f'{camera}-{t}' for camera in ('left', 'right')}
    scene = EstimatedScene(t, t, frames, objects)
    robot = {'name': 'SO-ARM101', 'arm_dof': 5, 'ee': list(tcp), 'joints': [0.] * 5,
             'joint_limits': [[-2., 2.]] * 5, 'gripper_openness': 1., 'gripper_angle': 1.,
             'paused': True, 'busy': False, 'epoch': 1, 'actions': int(t),
             'workspace': [[.06, .38], [-.25, .25], [.006, .36]], 'cube_edge_m': .02}
    detections = [PixelDetection(name, camera, frame, (100., 100.), .95, 2.).to_dict()
                  for name in ('red', 'blue') for camera, frame in frames.items()]
    return {'schema': 'so101-camera-observation-v1', 'source': 'rgb_only',
            'robot': robot, 'scene': scene.to_dict(),
            'visual_evidence': {'phase_id': 'release1', 'identity_token': 'blue-identity',
                'association_candidates': 1, 'association_confidence': .95,
                'source': 'dynamic_rgb_track', 'semantic_captured_at': 1.,
                'tcp_captured_at': t, 'tcp_std_m': .0005, 'detections': detections}}


def release_sequence():
    return (snapshot(1., carried=(.2, 0., .08), tcp=(.2, 0., .07)),
            snapshot(2.), snapshot(3., tcp=(.2, 0., .10)))


def verify(samples=None, **kwargs):
    return verify_stack_release(*(samples or release_sequence()), 'blue', 'red', now=3., **kwargs)


def test_default_lease_accepts_complete_fresh_rgb_geometry_without_contact_claim():
    result = verify()
    assert result['verified'], result
    assert result['reason'] == 'fresh_rgb_stack_geometry_verified'
    assert result['physical_contact_verified'] is False
    assert result['source'] == 'temporal_rgb_stack'
    assert result['relative_top_xyz_m'] == pytest.approx([0., 0., .02])


def test_explicit_short_lease_has_the_same_visual_only_verdict():
    result = verify(lease_seconds=60)
    assert result['verified'] and not result['physical_contact_verified']


@pytest.mark.parametrize('which', [0, 2])
def test_unknown_support_is_never_filled_from_another_frame(which):
    samples = release_sequence()
    samples[which]['scene']['objects']['red'] = ObjectEstimate(
        'red', 'unknown', LANDMARKS['red'], reason='occluded').to_dict()
    result = verify(samples, lease_seconds=60)
    assert not result['verified'] and not result['physical_contact_verified']
    assert result['reason'] == 'support_visual_evidence_unavailable'


@pytest.mark.parametrize('which', [0, 2])
@pytest.mark.parametrize('mutation', ['missing', 'occluded', 'replayed', 'low_confidence'])
def test_retained_support_coordinates_without_fresh_visible_rgb_proof_fail(which, mutation):
    samples = release_sequence()
    selected = samples[which]
    detections = selected['visual_evidence']['detections']
    if mutation == 'missing':
        selected['visual_evidence']['detections'] = [d for d in detections if d['object_id'] != 'red']
    else:
        for detection in detections:
            if detection['object_id'] != 'red':
                continue
            if mutation == 'occluded':
                detection.update(visibility='occluded', uv=None, pixel_sigma_px=None)
            elif mutation == 'replayed':
                detection['frame_id'] = 'unmatched-old-frame'
            else:
                detection['confidence'] = .1
    result = verify(samples, lease_seconds=60)
    assert not result['verified'], result
    assert not result['physical_contact_verified']


def test_low_confidence_support_estimate_is_not_a_verified_arrangement():
    samples = release_sequence()
    samples[2]['scene']['objects']['red']['confidence'] = .1
    assert not verify(samples, lease_seconds=60)['verified']


@pytest.mark.parametrize('name,xyz,sigma,reason', [
    ('red', (.2, 0., .02), .0031, 'support_uncertain'),
    ('blue', (.2, 0., .04), .0031, 'support_uncertain'),
    ('red', (.208, 0., .02), .0005, 'support_moved_or_uncertain'),
    ('blue', (.206, 0., .04), .0005, 'stack_overlap_unverified'),
    ('blue', (.2, .006, .04), .0005, 'stack_overlap_unverified'),
    ('blue', (.2, 0., .044), .0005, 'stack_height_unverified'),
    ('blue', (.2, 0., .036), .0005, 'stack_height_unverified'),
    ('blue', (.203, 0., .04), .001, 'stack_overlap_unverified'),
])
def test_uncertainty_is_added_to_support_motion_overlap_and_height_error(name, xyz, sigma, reason):
    samples = release_sequence()
    samples[2]['scene']['objects'][name] = estimate(name, xyz, sigma).to_dict()
    result = verify(samples, lease_seconds=60)
    assert not result['verified'] and result['reason'] == reason


def test_geometry_alone_without_tcp_retreat_cannot_verify_release():
    samples = release_sequence()
    samples[2]['robot']['ee'] = samples[1]['robot']['ee'][:]
    result = verify(samples, lease_seconds=60)
    assert not result['verified'] and result['reason'] == 'release_not_verified'


def test_carried_object_identity_change_fails_even_with_perfect_stack_geometry():
    samples = release_sequence()
    samples[2]['visual_evidence']['identity_token'] = 'other-object'
    assert not verify(samples, lease_seconds=60)['verified']


def test_stale_final_support_scene_is_rejected():
    result = verify_stack_release(*release_sequence(), 'blue', 'red', now=7., lease_seconds=60)
    assert not result['verified']


def test_self_support_and_non_rgb_evidence_fail_closed():
    samples = release_sequence()
    result = verify_stack_release(*samples, 'blue', 'blue', now=3., lease_seconds=60)
    assert not result['verified'] and result['reason'] == 'self_support'
    samples[0]['source'] = 'ground_truth'
    assert verify(samples, lease_seconds=60)['reason'] == 'rgb_required'


def path_executor():
    return VisualManipulationExecutor(simulation=SimpleNamespace(),
        observer=SimpleNamespace(validated=True), driver=SimpleNamespace(validated=True), clock=lambda:3.)


def path_snapshot(tcp=(.2, 0., .08)):
    return snapshot(3., tcp=tcp)


def support_policy(executor, name='red'):
    executor._support_policy = {'support_id': name, 'grip_offset': [0., 0., 0.]}


def test_only_named_support_gets_vertical_contact_approach_exception():
    executor = path_executor()
    current = path_snapshot()
    with pytest.raises(ValueError, match='observed_obstacle_clearance'):
        executor._check_path(current, 'blue', [.2, 0., .032])
    support_policy(executor)
    executor._check_path(current, 'blue', [.2, 0., .032])
    executor._check_path(path_snapshot(tcp=(.2, 0., .032)), 'blue', [.2, 0., .08])


def test_support_exception_does_not_allow_an_unrelated_obstacle():
    executor = path_executor()
    support_policy(executor, name='blue')
    with pytest.raises(ValueError, match='observed_obstacle_clearance'):
        executor._check_path(path_snapshot(), 'blue', [.2, 0., .032])


@pytest.mark.parametrize('start,target,reason', [
    ((.2, 0., .08), [.2, 0., .006], 'table_clearance'),
    ((.2, 0., .006), [.2, 0., .08], 'table_clearance'),
    ((.2, 0., .08), [.2, 0., .028], 'support_surface_clearance'),
    ((.206, 0., .08), [.206, 0., .032], 'support_alignment_unverified'),
    ((.18, 0., .04), [.2, 0., .04], 'horizontal_motion_below_transit_height'),
    ((.18, 0., .05), [.2, 0., .05], 'support_transit_clearance'),
])
def test_support_exception_preserves_table_surface_alignment_and_transit_guards(start, target, reason):
    executor = path_executor()
    support_policy(executor)
    with pytest.raises(ValueError, match=reason):
        executor._check_path(path_snapshot(tcp=start), 'blue', target)


def test_horizontal_transit_above_support_with_uncertainty_clearance_is_allowed():
    executor = path_executor()
    support_policy(executor)
    executor._check_path(path_snapshot(tcp=(.18, 0., .08)), 'blue', [.2, 0., .08])


@pytest.mark.parametrize('case', ['occluded', 'uncertain', 'stale'])
def test_named_support_policy_does_not_relax_fresh_obstacle_observation(case):
    executor = path_executor()
    support_policy(executor)
    current = path_snapshot()
    if case == 'occluded':
        current['scene']['objects']['red'] = ObjectEstimate(
            'red', 'unknown', LANDMARKS['red'], reason='occluded').to_dict()
    elif case == 'uncertain':
        current['scene']['objects']['red'] = estimate('red', (.2, 0., .02), .004).to_dict()
    else:
        current['scene']['captured_at'] = 0.
        executor.clock = lambda: 4.
    with pytest.raises(ValueError):
        executor._check_path(current, 'blue', [.2, 0., .032])
