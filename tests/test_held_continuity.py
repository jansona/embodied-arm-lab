"""Synthetic RGB/encoder contracts, not claims of physical grasp success."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from embodied_arm.control.held_continuity import remember_verified_hold, verify_stopped_hold
from embodied_arm.perception.camera_perception import EstimatedScene, LANDMARKS, ObjectEstimate, PixelDetection


def estimate(name, xyz, sigma=.0005):
    covariance = tuple(tuple(sigma*sigma if i == j else 0. for j in range(3)) for i in range(3))
    return ObjectEstimate(name, 'estimated', LANDMARKS[name], tuple(xyz), covariance,
                          sigma, .95, 0., 0., 45.)


def snapshot(t, *, epoch=2, carried=(.2, 0., .08), tcp=(.2, 0., .07),
             support=(.28, .08, .02), sigma=.0005):
    frames = {cam: f'{cam}-{t}' for cam in ('left', 'right')}
    points = {'red': estimate('red', carried, sigma), 'blue': estimate('blue', support),
              'goal': estimate('goal', (.17, .105, .0015))}
    scene = EstimatedScene(t, t, frames, points)
    robot = {'name': 'SO-ARM101', 'arm_dof': 5, 'ee': list(tcp), 'joints': [0.] * 5,
             'joint_limits': [[-2., 2.]] * 5, 'gripper_openness': .2, 'gripper_angle': .3,
             'paused': True, 'busy': False, 'epoch': epoch, 'actions': 12,
             'workspace': [[.06, .38], [-.25, .25], [.006, .36]], 'cube_edge_m': .02}
    detections = [PixelDetection(name, cam, fid, (100., 100.), .95, 2.,
                                 landmark=LANDMARKS[name]).to_dict()
                  for name in points for cam, fid in frames.items()]
    return {'schema': 'so101-camera-observation-v1', 'source': 'rgb_only',
            'scene': scene.to_dict(), 'robot': robot,
            'visual_evidence': {'phase_id': f'phase-{t}', 'identity_token': f'red-{t}',
                'association_candidates': 1, 'association_confidence': .95,
                'source': 'semantic_rgb', 'semantic_captured_at': t,
                'tcp_captured_at': t, 'tcp_std_m': .0005, 'detections': detections}}


def certificate(sample):
    return {'object_id': 'red', 'epoch': sample['robot']['epoch'],
            'actions': sample['robot']['actions'], 'frame_ids': deepcopy(sample['scene']['frame_ids']),
            'identity_token': sample['visual_evidence']['identity_token'],
            'verified_lift': True, 'source': 'phase_scoped_temporal_rgb'}


def history():
    old = snapshot(1., epoch=1)
    return remember_verified_hold(certificate(old), old)


def verify(fresh=None, **kwargs):
    fresh = fresh or snapshot(3.)
    options = {'stopped_robot': deepcopy(fresh['robot']), 'current_robot': deepcopy(fresh['robot']),
               'stopped_at': 2., 'now': 3., 'holding_state': 'held_verified'}
    options.update(kwargs)
    return verify_stopped_hold(history(), fresh, **options)


def assert_unknown(result):
    assert result['verified'] is False, result
    assert result['holding_state'] == 'unknown'
    assert result['certificate'] is None
    assert result['physical_contact_verified'] is False


def test_prior_verified_candidate_is_immutable_and_copies_mutable_inputs():
    old = snapshot(1., epoch=1)
    proof = certificate(old)
    candidate = remember_verified_hold(proof, old)
    assert candidate is not None
    old['robot']['ee'][2] = 999.
    old['scene']['frame_ids']['left'] = 'changed'
    proof['frame_ids']['right'] = 'changed'
    assert candidate.object_tcp_offset == pytest.approx((0., 0., .01))
    assert candidate.frame_ids == (('left', 'left-1.0'), ('right', 'right-1.0'))
    with pytest.raises(FrozenInstanceError):
        candidate.epoch = 999


@pytest.mark.parametrize('state', ['unknown', 'empty', 'release_pending', 'grasp_pending', None])
def test_uncertain_or_released_state_cannot_create_or_validate_history(state):
    old = snapshot(1., epoch=1)
    assert remember_verified_hold(certificate(old), old, holding_state=state) is None
    assert_unknown(verify(holding_state=state))


@pytest.mark.parametrize('field,value', [('epoch', 99), ('actions', 99),
    ('identity_token', 'other'), ('frame_ids', {'left': 'old', 'right': 'old'}),
    ('verified_lift', False), ('source', 'encoder_only'), ('object_id', 'blue')])
def test_certificate_must_match_original_verified_rgb_snapshot(field, value):
    old = snapshot(1., epoch=1)
    proof = certificate(old)
    proof[field] = value
    assert remember_verified_hold(proof, old) is None


def test_fresh_clear_unchanged_hold_gets_only_new_certificate_and_no_contact_claim():
    fresh = snapshot(3.)
    result = verify(fresh)
    assert result['verified'], result
    assert result['holding_state'] == 'held_verified'
    assert result['physical_contact_verified'] is False
    assert result['certificate'] == certificate(fresh)
    assert result['uncertainty_bound_m'] == pytest.approx(.006)
    assert fresh == snapshot(3.)  # Helper never mutates the observer's evidence.


def test_old_certificate_and_closed_gripper_cannot_create_live_proof_without_new_rgb():
    assert_unknown(verify(snapshot(1., epoch=2)))
    fresh = snapshot(3.)
    fresh.pop('visual_evidence')
    assert_unknown(verify(fresh))


def test_no_historical_hold_stays_unknown():
    fresh = snapshot(3.)
    result = verify_stopped_hold(None, fresh, stopped_robot=fresh['robot'],
                                current_robot=fresh['robot'], stopped_at=2., now=3.)
    assert_unknown(result)
    assert result['reason'] == 'no_verified_hold_history'


@pytest.mark.parametrize('stopped_at,now', [(3., 3.), (3.1, 3.1), (2., 6.1),
                                        (2., 122.), (float('nan'), 3.)])
def test_stale_future_prestop_or_expired_evidence_is_rejected(stopped_at, now):
    assert_unknown(verify(stopped_at=stopped_at, now=now))


@pytest.mark.parametrize('field,value', [('epoch', 1), ('actions', 11), ('paused', False),
    ('busy', True), ('gripper_openness', .5), ('gripper_openness', 1.),
    ('gripper_angle', .5), ('gripper_angle', float('nan'))])
def test_invalid_or_changed_robot_state_never_implies_empty_or_held(field, value):
    fresh = snapshot(3.)
    fresh['robot'][field] = value
    assert_unknown(verify(fresh))


@pytest.mark.parametrize('field,value', [('epoch', 3), ('actions', 13),
    ('ee', [.2, 0., .08]), ('joints', [.1, 0., 0., 0., 0.]),
    ('gripper_openness', .22), ('gripper_angle', .32)])
def test_robot_change_after_stop_or_during_capture_fails(field, value):
    current = snapshot(3.)['robot']
    current[field] = value
    assert_unknown(verify(current_robot=current))
    assert_unknown(verify(stopped_robot=current))


@pytest.mark.parametrize('name', ['red', 'blue', 'goal'])
@pytest.mark.parametrize('mutation', ['missing', 'occluded', 'stale_frame', 'low_confidence'])
def test_every_required_object_needs_current_visible_stereo_pixels(name, mutation):
    fresh = snapshot(3.)
    detections = fresh['visual_evidence']['detections']
    if mutation == 'missing':
        fresh['visual_evidence']['detections'] = [d for d in detections if d['object_id'] != name]
    else:
        target = next(d for d in detections if d['object_id'] == name)
        if mutation == 'occluded':
            target.update(visibility='occluded', uv=None, pixel_sigma_px=None)
        elif mutation == 'stale_frame':
            target['frame_id'] = 'stale'
        else:
            target['confidence'] = .1
    assert_unknown(verify(fresh))


@pytest.mark.parametrize('field,value', [('association_candidates', 2),
    ('association_confidence', .2), ('source', 'encoder_prediction'),
    ('source', 'dynamic_rgb_track'), ('semantic_captured_at', 1.),
    ('tcp_captured_at', 2.), ('tcp_std_m', float('nan'))])
def test_ambiguous_uncertain_or_relabelled_identity_cannot_pass(field, value):
    fresh = snapshot(3.)
    fresh['visual_evidence'][field] = value
    assert_unknown(verify(fresh))


def test_one_replayed_stereo_frame_is_enough_to_fail():
    fresh = snapshot(3.)
    fresh['scene']['frame_ids']['left'] = 'left-1.0'
    for detection in fresh['visual_evidence']['detections']:
        if detection['camera_id'] == 'left':
            detection['frame_id'] = 'left-1.0'
    assert_unknown(verify(fresh))


def test_observed_object_tcp_offset_not_just_absolute_object_position_must_match():
    # Arm can have carried the same observed offset before Stop.
    moved = snapshot(3., carried=(.23, 0., .08), tcp=(.23, 0., .07))
    moved['robot']['actions'] = 15
    assert verify(moved)['verified']
    # Dropping/slipping relative to TCP is rejected without moving to probe it.
    drifted = snapshot(3., carried=(.212, 0., .08))
    result = verify(drifted)
    assert_unknown(result)
    assert result['reason'] == 'object_tcp_relation_changed_or_uncertain'


def test_covariance_consumes_margin_instead_of_relaxing_the_offset_gate():
    assert_unknown(verify(snapshot(3., carried=(.206, 0., .08), sigma=.002)))
    assert_unknown(verify(snapshot(3., sigma=.0031)))


def test_same_gripper_and_offset_do_not_prove_hold_if_object_rests_on_table():
    fresh = snapshot(3., carried=(.2, 0., .0215), tcp=(.2, 0., .0115))
    result = verify(fresh)
    assert_unknown(result)
    assert result['reason'] == 'object_may_rest_on_table'


def test_same_offset_does_not_prove_hold_if_object_rests_on_another_cube():
    fresh = snapshot(3., carried=(.2, 0., .04), tcp=(.2, 0., .03), support=(.2, 0., .02))
    result = verify(fresh)
    assert_unknown(result)
    assert result['reason'] == 'object_may_rest_on_support'


def test_cube_footprint_and_uncertainty_are_used_to_exclude_edge_support():
    # Centres are more than a half-width apart, but faces can still overlap.
    fresh = snapshot(3., carried=(.2, 0., .04), tcp=(.2, 0., .03), support=(.219, 0., .02))
    assert_unknown(verify(fresh))


def test_support_without_overlap_does_not_hide_clear_hold_above_table():
    fresh = snapshot(3., carried=(.2, 0., .05), tcp=(.2, 0., .04), support=(.28, .08, .08))
    assert verify(fresh)['verified']


@pytest.mark.parametrize('name', ['red', 'blue', 'goal'])
def test_unknown_landmarks_never_get_filled_from_historical_scene(name):
    fresh = snapshot(3.)
    fresh['scene']['objects'][name] = ObjectEstimate(name, 'unknown', LANDMARKS[name],
                                                    reason='occluded').to_dict()
    assert_unknown(verify(fresh))


def test_snapshot_with_oracle_source_is_rejected():
    fresh = snapshot(3.)
    fresh['source'] = 'simulator_body_pose'
    assert_unknown(verify(fresh))


def tracked_reacquisition():
    fresh = snapshot(3.)
    fresh['visual_evidence'].update(source='dynamic_rgb_track', semantic_captured_at=2.5,
        semantic_reacquisition={'source': 'qwen_rgb', 'captured_at': 2.5, 'epoch': 2,
                               'actions': 12, 'frame_ids': {'left': 'left-2.5', 'right': 'right-2.5'}})
    return fresh


def test_slow_semantic_refresh_can_use_newer_tracked_pixels_without_restamping_time():
    fresh = tracked_reacquisition()
    assert verify(fresh)['verified']
    assert fresh['visual_evidence']['semantic_captured_at'] == 2.5


@pytest.mark.parametrize('field,value', [('captured_at', 1.), ('captured_at', 3.1),
    ('captured_at', 2.6), ('epoch', 1), ('actions', 11), ('source', 'encoder_guess'),
    ('frame_ids', {'left': 'left-1.0', 'right': 'right-1.0'}),
    ('frame_ids', {'left': 'left-2.5'}), ('frame_ids', None)])
def test_semantic_reacquisition_record_must_be_poststop_rgb_in_same_robot_state(field, value):
    fresh = tracked_reacquisition()
    fresh['visual_evidence']['semantic_reacquisition'][field] = value
    assert_unknown(verify(fresh))


def test_unobserved_tilt_cannot_be_assumed_upright_to_exclude_a_support():
    fresh = snapshot(3., carried=(.2, 0., .05), tcp=(.2, 0., .04), support=(.23, 0., .02))
    assert_unknown(verify(fresh))
