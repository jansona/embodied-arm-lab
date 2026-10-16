"""Conservative RGB-only continuity of an already verified hold across Stop.

A candidate is historical evidence, never permission to move. This module does
not capture images, command a probe, inspect simulator bodies, or verify physical
contact. The caller must invalidate history on release/reset/uncertain grasp and
provide genuinely new paused RGB evidence after Stop. A failed check leaves the
holding state unknown; it does not establish that the gripper is empty.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from embodied_arm.control.so101_control import robot_only
from embodied_arm.perception.visual_feedback import _from_snapshot


_CERTIFICATE_KEYS = {
    'object_id', 'epoch', 'actions', 'frame_ids', 'identity_token',
    'verified_lift', 'source',
}
_MAX_AGE_S = 3.
_MAX_HISTORY_S = 120.
_MIN_CONFIDENCE = .85
_MAX_STD_M = .003
_SIGMA = 3.
_OFFSET_TOLERANCE_M = .015
_MIN_CLEARANCE_M = .005


@dataclass(frozen=True)
class HeldCandidate:
    """Immutable historical lift record; deliberately has no live certificate."""

    object_id: str
    epoch: int
    actions: int
    frame_ids: tuple[tuple[str, str], ...]
    identity_token: str
    captured_at: float
    object_tcp_offset: tuple[float, float, float]
    object_std_m: float
    tcp_std_m: float
    gripper_openness: float
    gripper_angle: float
    cube_edge_m: float


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _robot(value):
    robot = robot_only(value)
    if (robot['paused'] is not True or robot['busy'] is not False
            or any(type(robot[k]) is not int or robot[k] < 0 for k in ('epoch', 'actions'))
            or not _finite(robot['gripper_openness'])
            or not _finite(robot['gripper_angle'])):
        raise ValueError('stopped_robot_required')
    return robot


def _same_robot(left, right):
    return (all(left[k] == right[k] for k in ('epoch', 'actions', 'cube_edge_m'))
            and math.dist(left['ee'], right['ee']) <= .001
            and max(abs(a-b) for a, b in zip(left['joints'], right['joints'])) <= .005
            and abs(left['gripper_openness']-right['gripper_openness']) <= .01
            and abs(left['gripper_angle']-right['gripper_angle']) <= .01)


def _observation(snapshot, object_id, now, max_age_s=_MAX_AGE_S):
    """Validate image/encoder provenance, without guessing missing landmarks."""
    if not _finite(max_age_s) or not 1<=max_age_s<=60:raise ValueError('invalid_observation_age')
    obs = _from_snapshot(snapshot)
    robot = _robot(snapshot['robot'])
    scene = obs.scene
    if (not _finite(now) or not scene.captured_at <= scene.estimated_at <= now
            or not 0 <= now-scene.captured_at <= max_age_s):
        raise ValueError('stale_rgb')
    if (not isinstance(obs.phase_id, str) or not obs.phase_id
            or not isinstance(obs.identity_token, str) or not obs.identity_token
            or obs.source not in ('semantic_rgb', 'dynamic_rgb_track')):
        raise ValueError('invalid_rgb_identity')
    if (type(obs.association_candidates) is not int or obs.association_candidates != 1
            or not _finite(obs.association_confidence)
            or not _MIN_CONFIDENCE <= obs.association_confidence <= 1):
        raise ValueError('ambiguous_identity')
    if (not all(_finite(v) for v in (obs.tcp_captured_at, obs.semantic_captured_at, obs.tcp_std_m))
            or not 0 <= obs.tcp_std_m <= _MAX_STD_M
            or abs(obs.tcp_captured_at-scene.captured_at) > .05
            or obs.tcp_captured_at > now):
        raise ValueError('encoder_frame_skew')
    if (not 0 <= scene.captured_at-obs.semantic_captured_at <= _MAX_HISTORY_S
            or obs.source == 'semantic_rgb' and obs.semantic_captured_at != scene.captured_at):
        raise ValueError('expired_semantic_identity')
    _visible_point(obs, object_id, now, max_age_s)
    return obs, robot


def _visible_point(obs, object_id, now, max_age_s=_MAX_AGE_S):
    point = obs.scene.require(object_id, now=now, max_age_s=max_age_s)
    if point.confidence < _MIN_CONFIDENCE or point.position_std_m > _MAX_STD_M:
        raise ValueError('uncertain_landmark')
    selected = [d for d in obs.detections if d.object_id == object_id]
    if (len(selected) != 2 or {d.camera_id for d in selected} != set(obs.scene.frame_ids)
            or any(d.frame_id != obs.scene.frame_ids[d.camera_id]
                   or d.visibility != 'visible' or d.confidence < _MIN_CONFIDENCE
                   or d.landmark != point.landmark for d in selected)):
        raise ValueError('missing_visible_rgb_evidence')
    return point


def _fresh_semantic_reacquisition(snapshot, obs, candidate, stopped, stopped_at):
    """Keep slow semantic inference tied to its real frame, never restamp it.

    The observer may use a subsequent locally tracked RGB frame for the strict
    configured bounded geometry lease, provided the original semantic refresh was
    also captured after Stop with no intervening commanded robot action.
    """
    record = snapshot['visual_evidence'].get('semantic_reacquisition')
    if (not isinstance(record, dict)
            or set(record) != {'frame_ids', 'captured_at', 'epoch', 'actions', 'source'}
            or record['source'] != 'qwen_rgb'
            or not _finite(record['captured_at'])
            or not stopped_at < record['captured_at'] <= obs.scene.captured_at
            or record['captured_at'] != obs.semantic_captured_at
            or any(type(record[k]) is not int or record[k] != stopped[k]
                   for k in ('epoch', 'actions'))
            or not isinstance(record['frame_ids'], dict)
            or set(record['frame_ids']) != set(obs.scene.frame_ids)
            or any(not isinstance(fid, str) or not fid for fid in record['frame_ids'].values())
            or any(pair in candidate.frame_ids for pair in record['frame_ids'].items())):
        return False
    return True


def remember_verified_hold(certificate, verified_snapshot, *, holding_state='held_verified', max_age_s=_MAX_AGE_S):
    """Copy only a certificate-matched verified lift into historical storage.

    Call with the snapshot that originally produced the successful lift proof,
    not the observer's latest (possibly mid-release) scene. The executor owns
    the holding_state and must pass release_pending/unknown when appropriate.
    """
    try:
        if (holding_state != 'held_verified' or not isinstance(certificate, dict)
                or set(certificate) != _CERTIFICATE_KEYS
                or certificate['object_id'] not in ('red', 'blue')
                or certificate['verified_lift'] is not True
                or certificate['source'] != 'phase_scoped_temporal_rgb'):
            return None
        object_id = certificate['object_id']
        obs, robot = _observation(verified_snapshot, object_id,
                                  verified_snapshot['scene']['estimated_at'],max_age_s)
        if (any(certificate[k] != robot[k] for k in ('epoch', 'actions'))
                or any(type(certificate[k]) is not int for k in ('epoch', 'actions'))
                or certificate['identity_token'] != obs.identity_token
                or certificate['frame_ids'] != dict(obs.scene.frame_ids)
                or robot['gripper_openness'] >= .9):
            return None
        point = obs.scene.objects[object_id]
        offset = tuple(a-b for a, b in zip(point.xyz, robot['ee']))
        if math.dist(point.xyz, robot['ee']) + _SIGMA*(point.position_std_m+obs.tcp_std_m) > .05:
            return None
        return HeldCandidate(object_id, robot['epoch'], robot['actions'],
                             tuple(sorted(obs.scene.frame_ids.items())), obs.identity_token,
                             obs.scene.captured_at, offset, point.position_std_m, obs.tcp_std_m,
                             robot['gripper_openness'], robot['gripper_angle'], robot['cube_edge_m'])
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return None


def verify_stopped_hold(candidate, fresh_snapshot, *, stopped_robot, current_robot,
                        stopped_at, now, holding_state='held_verified', max_age_s=_MAX_AGE_S):
    """Return a fresh handoff certificate only after all continuity gates pass.

    stopped_robot is captured immediately after Stop; current_robot is read
    after the fresh RGB capture. Neither an old proof nor a closed gripper alone
    is sufficient. The RGB goal marker is the observed fixed bench reference;
    the other cube must also be visible to exclude an overlapping support.
    """
    failed = {'verified': False, 'reason': 'held_continuity_unverified',
              'holding_state': 'unknown', 'certificate': None,
              'physical_contact_verified': False, 'source': 'stopped_hold_rgb_continuity'}

    def reject(reason):
        return {**failed, 'reason': reason}

    try:
        if not isinstance(candidate, HeldCandidate):
            return reject('no_verified_hold_history')
        if holding_state != 'held_verified':
            return reject('holding_state_uncertain')
        if (not _finite(now) or not _finite(stopped_at)
                or not candidate.captured_at <= stopped_at <= now
                or now-candidate.captured_at > _MAX_HISTORY_S):
            return reject('expired_hold_history')
        stopped, current = _robot(stopped_robot), _robot(current_robot)
        if (stopped['epoch'] <= candidate.epoch or stopped['actions'] < candidate.actions
                or not _same_robot(stopped, current)):
            return reject('robot_changed_since_stop')
        obs, observed_robot = _observation(fresh_snapshot, candidate.object_id, now,max_age_s)
        if not _same_robot(observed_robot, stopped) or not _same_robot(observed_robot, current):
            return reject('robot_changed_during_capture')
        if (obs.scene.captured_at <= stopped_at
                or set(obs.scene.frame_ids) != {camera for camera, _ in candidate.frame_ids}
                or any(pair in candidate.frame_ids for pair in obs.scene.frame_ids.items())):
            return reject('fresh_post_stop_rgb_required')
        # A new identity token needs new semantic image evidence. Local tracking
        # cannot relabel an old association into a fresh identity after Stop.
        if (obs.identity_token != candidate.identity_token and obs.source != 'semantic_rgb'
                and not _fresh_semantic_reacquisition(fresh_snapshot, obs, candidate, stopped, stopped_at)):
            return reject('fresh_semantic_identity_required')
        if (current['cube_edge_m'] != candidate.cube_edge_m
                or current['gripper_openness'] >= .9
                or abs(current['gripper_openness']-candidate.gripper_openness) > .01
                or abs(current['gripper_angle']-candidate.gripper_angle) > .01):
            return reject('gripper_changed_since_verified_hold')
        point = obs.scene.objects[candidate.object_id]
        offset = tuple(a-b for a, b in zip(point.xyz, current['ee']))
        margin = _SIGMA*(candidate.object_std_m+candidate.tcp_std_m
                         + point.position_std_m+obs.tcp_std_m)
        offset_error = math.dist(offset, candidate.object_tcp_offset)
        if (offset_error+margin > _OFFSET_TOLERANCE_M
                or math.dist(point.xyz, current['ee'])
                + _SIGMA*(point.position_std_m+obs.tcp_std_m) > .05):
            return reject('object_tcp_relation_changed_or_uncertain')
        try:
            table = _visible_point(obs, 'goal', now,max_age_s)
            support = _visible_point(obs, 'blue' if candidate.object_id == 'red' else 'red', now,max_age_s)
        except (ValueError, KeyError, TypeError):
            return reject('support_clearance_unobserved')
        # Orientation is not observed. The farthest cube vertex from the
        # observed face centre is sqrt(1.5)*edge; this conservative radius
        # bounds height/footprint without assuming an upright cube.
        extent = candidate.cube_edge_m*math.sqrt(1.5)
        bottom = point.xyz[2]-extent
        table_margin = _SIGMA*(point.position_std_m+table.position_std_m)
        table_clearance = bottom-table.xyz[2]-table_margin
        if table_clearance <= _MIN_CLEARANCE_M:
            return reject('object_may_rest_on_table')
        support_margin = _SIGMA*(point.position_std_m+support.position_std_m)
        # Expand both possible footprints and support height without querying
        # either cube's orientation from the simulator.
        overlaps = all(abs(point.xyz[i]-support.xyz[i])
                       <= 2*extent+support_margin for i in (0, 1))
        if overlaps and bottom-support.xyz[2]-extent-support_margin <= _MIN_CLEARANCE_M:
            return reject('object_may_rest_on_support')
        certificate = {'object_id': candidate.object_id, 'epoch': current['epoch'],
                       'actions': current['actions'], 'frame_ids': dict(obs.scene.frame_ids),
                       'identity_token': obs.identity_token, 'verified_lift': True,
                       'source': 'phase_scoped_temporal_rgb'}
        return {**failed, 'verified': True, 'reason': 'fresh_rgb_hold_continuity_verified',
                'holding_state': 'held_verified', 'certificate': certificate,
                'object_tcp_error_m': offset_error, 'uncertainty_bound_m': margin,
                'table_clearance_lower_bound_m': table_clearance}
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return failed
