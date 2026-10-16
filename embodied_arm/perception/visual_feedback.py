"""Phase-scoped temporal RGB evidence, never a physical-contact oracle.

Providers must triangulate fresh, uniquely associated visible landmarks. The
static-hover RGBRegionTracker is deliberately NOT used here: its immutable 4 px
anchor is unsuitable for a carried object. A dynamic provider can associate
image regions within a phase or request a bounded semantic refresh; neither a
closed gripper nor commanded motion substitutes for an observed object.

Covariance is retained, never reduced. Bounds use sums of endpoint standard
deviations (unknown temporal correlation) and reject when error PLUS uncertainty
exceeds a physical tolerance. This module performs no inference/network calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from embodied_arm.perception.camera_perception import EstimatedScene, PixelDetection

Vector = tuple[float, float, float]


def _vec(value: Vector) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Expected finite three-vector')
    return a


@dataclass(frozen=True)
class VisualObservation:
    scene: EstimatedScene
    detections: tuple[PixelDetection, ...]
    tcp_xyz: Vector  # encoder forward kinematics, never simulator body pose
    tcp_captured_at: float
    epoch: int
    phase_id: str
    identity_token: str
    association_candidates: int
    association_confidence: float
    source: Literal['semantic_rgb', 'dynamic_rgb_track']
    semantic_captured_at: float
    tcp_std_m: float = .001


@dataclass(frozen=True)
class VerificationConfig:
    max_age_s: float = 3.
    max_phase_s: float = 30.
    max_semantic_age_s: float = 30.
    max_encoder_skew_s: float = .05
    min_confidence: float = .8
    max_position_std_m: float = .005
    sigma_multiplier: float = 3.
    min_lift_m: float = .015
    motion_tolerance_m: float = .015
    max_initial_offset_m: float = .05
    min_retreat_m: float = .025
    stationary_tolerance_m: float = .01
    surface_tolerance_m: float = .012
    target_xy_tolerance_m: float = .02
    min_release_dwell_s: float = .15
    max_semantic_refreshes: int = 1

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid verification bound: ' + name)
        if (self.sigma_multiplier < 3 or not .5 <= self.min_confidence <= 1
                or type(self.max_semantic_refreshes) is not int
                or self.max_semantic_refreshes > 2):
            raise ValueError('Invalid confidence or refresh bound')


@dataclass(frozen=True)
class VisualVerdict:
    phase: str
    verified: bool
    reason: str
    refresh_required: bool = False
    terminal: bool = False
    object_displacement_m: Vector | None = None
    tcp_displacement_m: Vector | None = None
    uncertainty_bound_m: float | None = None
    physical_contact_verified: bool = False
    source: str = 'phase_scoped_temporal_rgb'

    @property
    def evidence(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def to_dict(self) -> dict:
        return self.evidence


class VisualPhaseVerifier:
    """One immutable identity/baseline per lift or release phase.

    begin_* validates the baseline while fresh. update requires new stereo frame
    IDs, matching phase/epoch/token, synchronized encoders and bounded semantic
    age. On local tracking loss stop motion, acquire fresh semantic RGB evidence,
    and call update again; refreshes never replace the phase baseline or identity.
    A terminal rejection requires beginning a new phase with fresh evidence.
    """
    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()
        self._baseline = None
        self._phase = 'inactive'
        self._terminal = True

    def _validate(self, obs, now):
        c = self.config
        if not isinstance(obs, VisualObservation) or not isinstance(obs.scene, EstimatedScene):
            raise ValueError('invalid_rgb_provenance')
        if (not math.isfinite(now) or not isinstance(obs.epoch, int) or obs.epoch < 0
                or not obs.phase_id or not obs.identity_token
                or obs.source not in ('semantic_rgb', 'dynamic_rgb_track')):
            raise ValueError('invalid_phase_provenance')
        _vec(obs.tcp_xyz)
        times = (obs.tcp_captured_at, obs.semantic_captured_at, obs.tcp_std_m,
                 obs.association_confidence)
        if any(not math.isfinite(x) for x in times) or obs.tcp_std_m < 0:
            raise ValueError('invalid_encoder_or_identity')
        s = obs.scene
        if not s.captured_at <= s.estimated_at <= now:
            raise ValueError('invalid_estimation_time')
        if not 0 <= now - s.captured_at <= c.max_age_s:
            raise ValueError('stale_rgb')
        if abs(obs.tcp_captured_at - s.captured_at) > c.max_encoder_skew_s or obs.tcp_captured_at > now:
            raise ValueError('encoder_frame_skew')
        if not 0 <= s.captured_at - obs.semantic_captured_at <= c.max_semantic_age_s:
            raise ValueError('expired_semantic_identity')
        if obs.source == 'semantic_rgb' and obs.semantic_captured_at != s.captured_at:
            raise ValueError('semantic_refresh_not_current')
        if obs.association_candidates != 1 or not c.min_confidence <= obs.association_confidence <= 1:
            raise ValueError('ambiguous_identity')
        obj = s.require(self._object_id, now=now, max_age_s=c.max_age_s)
        if obj.confidence < c.min_confidence or obj.position_std_m > c.max_position_std_m:
            raise ValueError('uncertain_landmark')
        selected = [d for d in obs.detections if d.object_id == self._object_id]
        if (len(selected) != 2 or {d.camera_id for d in selected} != set(s.frame_ids)
                or any(d.frame_id != s.frame_ids[d.camera_id] or d.visibility != 'visible'
                       or d.confidence < c.min_confidence or d.landmark != obj.landmark for d in selected)):
            raise ValueError('missing_or_mismatched_rgb_evidence')
        return obj

    def _begin(self, phase, object_id, observation, now):
        self._terminal = True
        if object_id not in ('red', 'blue'):
            raise ValueError('Unknown manipulation object')
        self._object_id = object_id
        obj = self._validate(observation, now)
        self._phase = phase
        self._baseline = observation
        self._first = obj
        self._seen = set(observation.scene.frame_ids.items())
        self._last_time = observation.scene.captured_at
        self._refreshes = 0
        self._needs_refresh = False
        self._terminal = False

    def begin_lift(self, object_id: str, observation: VisualObservation, *,
                   commanded_tcp_delta: Vector, now: float):
        command = _vec(commanded_tcp_delta)
        if command[2] < self.config.min_lift_m:
            raise ValueError('Command must include an observable lift')
        self._begin('lift', object_id, observation, now)
        self._command = command
        margin = self.config.sigma_multiplier * (self._first.position_std_m + observation.tcp_std_m)
        if np.linalg.norm(_vec(self._first.xyz) - _vec(observation.tcp_xyz)) + margin > self.config.max_initial_offset_m:
            self._terminal = True
            raise ValueError('Landmark is not near encoder TCP')

    def begin_release(self, object_id: str, observation: VisualObservation, *,
                      expected_landmark_xyz: Vector, target_std_m: float, now: float):
        """Target is intended TOP landmark at support, not object body centre.

        Supply surface/target uncertainty from RGB geometry and any known object
        geometry; it is added to observed uncertainty rather than subtracted.
        Baseline must be captured at release before TCP retreat.
        """
        target = _vec(expected_landmark_xyz)
        if not math.isfinite(target_std_m) or target_std_m < 0:
            raise ValueError('Invalid target uncertainty')
        self._begin('release', object_id, observation, now)
        self._target = target
        self._target_std = target_std_m

    def update(self, observation: VisualObservation, *, now: float) -> VisualVerdict:
        if self._terminal:
            return VisualVerdict(self._phase, False, 'phase_inactive', terminal=True)
        def fail(reason, refresh=False):
            self._terminal = not refresh
            self._needs_refresh = refresh
            return VisualVerdict(self._phase, False, reason, refresh, not refresh)
        base, c = self._baseline, self.config
        if not isinstance(observation, VisualObservation):
            return fail('invalid_rgb_provenance')
        if (observation.epoch != base.epoch or observation.phase_id != base.phase_id
                or observation.identity_token != base.identity_token):
            return fail('phase_or_identity_changed')
        s = observation.scene
        if not isinstance(s, EstimatedScene):
            return fail('invalid_rgb_provenance')
        if now - base.scene.captured_at > c.max_phase_s:
            return fail('phase_expired')
        if s.captured_at <= self._last_time or any(pair in self._seen for pair in s.frame_ids.items()):
            return fail('replayed_rgb_frame')
        # Even rejected evidence cannot be replayed as a successful refresh.
        self._last_time = s.captured_at
        self._seen.update(s.frame_ids.items())
        if self._needs_refresh and observation.source != 'semantic_rgb':
            return fail('semantic_refresh_required', self._refreshes < c.max_semantic_refreshes)
        if observation.source == 'semantic_rgb':
            if self._refreshes >= c.max_semantic_refreshes:
                return fail('semantic_refresh_budget_exhausted')
            self._refreshes += 1
        try:
            obj = self._validate(observation, now)
        except (ValueError, TypeError, AttributeError) as exc:
            recoverable = str(exc) in ('Requested landmark is not safely estimated',
                                      'ambiguous_identity', 'missing_or_mismatched_rgb_evidence')
            refresh = (recoverable and observation.source == 'dynamic_rgb_track'
                       and self._refreshes < c.max_semantic_refreshes)
            return fail(str(exc), refresh)
        self._needs_refresh = False
        delta = _vec(obj.xyz) - _vec(self._first.xyz)
        tcp = _vec(observation.tcp_xyz) - _vec(base.tcp_xyz)
        visual_bound = c.sigma_multiplier * (obj.position_std_m + self._first.position_std_m)
        bound = visual_bound + c.sigma_multiplier * (observation.tcp_std_m + base.tcp_std_m)
        if self._phase == 'lift':
            ok = (delta[2] - visual_bound >= c.min_lift_m
                  and tcp[2] - c.sigma_multiplier * (observation.tcp_std_m + base.tcp_std_m) >= c.min_lift_m
                  and np.linalg.norm(delta - tcp) + bound <= c.motion_tolerance_m
                  and np.linalg.norm(tcp - self._command) + c.sigma_multiplier * (observation.tcp_std_m + base.tcp_std_m) <= c.motion_tolerance_m)
            reason = 'visual_lift_consistent' if ok else 'lift_or_offset_not_verified'
        else:
            target_bound = c.sigma_multiplier * (obj.position_std_m + self._target_std)
            first_bound = c.sigma_multiplier * (self._first.position_std_m + self._target_std)
            near = lambda p, b: (abs(p[2] - self._target[2]) + b <= c.surface_tolerance_m
                                and np.linalg.norm(p[:2] - self._target[:2]) + b <= c.target_xy_tolerance_m)
            separation = np.linalg.norm(_vec(obj.xyz) - _vec(observation.tcp_xyz))
            initial_separation = np.linalg.norm(_vec(self._first.xyz) - _vec(base.tcp_xyz))
            ok = (s.captured_at - base.scene.captured_at >= c.min_release_dwell_s
                  and np.linalg.norm(delta) + visual_bound <= c.stationary_tolerance_m
                  and np.linalg.norm(tcp) - c.sigma_multiplier * (observation.tcp_std_m + base.tcp_std_m) >= c.min_retreat_m
                  and separation - initial_separation - bound >= c.min_retreat_m
                  and near(_vec(obj.xyz), target_bound) and near(_vec(self._first.xyz), first_bound))
            reason = 'visual_release_consistent' if ok else 'release_not_verified'
        self._terminal = True
        return VisualVerdict(self._phase, bool(ok), reason, terminal=True,
                             object_displacement_m=tuple(float(x) for x in delta),
                             tcp_displacement_m=tuple(float(x) for x in tcp),
                             uncertainty_bound_m=bound)


def _from_snapshot(snapshot: dict) -> VisualObservation:
    if (snapshot.get('schema') != 'so101-camera-observation-v1'
            or snapshot.get('source') != 'rgb_only'):
        raise ValueError('RGB observation required')
    e = snapshot['visual_evidence']
    robot = snapshot['robot']
    return VisualObservation(
        EstimatedScene.from_dict(snapshot['scene']),
        tuple(PixelDetection.from_dict(d) for d in e['detections']),
        tuple(robot['ee']), e['tcp_captured_at'], robot['epoch'], e['phase_id'],
        e['identity_token'], e['association_candidates'], e['association_confidence'],
        e['source'], e['semantic_captured_at'], e.get('tcp_std_m', .001))


def verify_lift(before_snapshot, after_snapshot, object_id, *, commanded_tcp_delta=None,
                now=None, config=None) -> VisualVerdict:
    """Strict CameraSceneSource adapter. Missing identity evidence fails closed."""
    try:
        before, after = map(_from_snapshot, (before_snapshot, after_snapshot))
        # Explicit command is preferred. Without it this checks observed encoder
        # co-motion only; it does not attest that the command was executed.
        delta = commanded_tcp_delta if commanded_tcp_delta is not None else tuple(
            _vec(after.tcp_xyz) - _vec(before.tcp_xyz))
        verifier = VisualPhaseVerifier(config)
        verifier.begin_lift(object_id, before, commanded_tcp_delta=delta,
                            now=before.scene.estimated_at)
        return verifier.update(after, now=after.scene.estimated_at if now is None else now)
    except (ValueError, KeyError, TypeError) as exc:
        return VisualVerdict('lift', False, 'invalid_evidence:' + str(exc), terminal=True)


def verify_release(held_snapshot, released_snapshot, retreated_snapshot, object_id,
                   support_xyz, *, target_std_m, now=None, config=None) -> VisualVerdict:
    """Verify release at expected TOP landmark, after previously verified holding.

    Held observation must belong to the release phase and precede released RGB.
    It establishes identity continuity only; this function never infers a prior
    grasp. The caller must retain the successful lift verdict separately.
    """
    try:
        held, released, retreated = map(_from_snapshot,
                                      (held_snapshot, released_snapshot, retreated_snapshot))
        limits = config or VerificationConfig()
        if (held.phase_id != released.phase_id or held.identity_token != released.identity_token
                or held.epoch != released.epoch or held.scene.captured_at >= released.scene.captured_at
                or retreated.scene.captured_at - held.scene.captured_at > limits.max_phase_s
                or any(held.scene.frame_ids.get(cam) == frame
                       for cam, frame in released.scene.frame_ids.items())):
            raise ValueError('release_identity_discontinuity')
        verifier = VisualPhaseVerifier(config)
        verifier.begin_release(object_id, released, expected_landmark_xyz=support_xyz,
                               target_std_m=target_std_m, now=released.scene.estimated_at)
        verifier._validate(held, held.scene.estimated_at)
        return verifier.update(retreated, now=retreated.scene.estimated_at if now is None else now)
    except (ValueError, KeyError, TypeError) as exc:
        return VisualVerdict('release', False, 'invalid_evidence:' + str(exc), terminal=True)


@dataclass(frozen=True)
class DynamicTrackResult:
    accepted: bool
    reason: str
    detections: tuple[PixelDetection, ...] = ()
    identity_token: str | None = None
    refresh_required: bool = False
    source: str = 'dynamic_rgb_track'


class DynamicRGBPhaseTracker:
    """Encoder-predicted bounded RGB face association for one carried object.

    Prediction seeds a search, NEVER a returned pixel. Actual visible face edges
    supply every result. Unique-colour component/quad tests in the face refiner,
    original face area, and bounded image residual reject lookalikes/slip. This
    is restricted coloured-cuboid tracking, not general reidentification. Any
    failure invalidates this tracker; caller may acquire ONE semantic refresh
    through VisualPhaseVerifier without changing its original phase identity.
    """
    def __init__(self, *, profile='baseline', max_residual_px=12., semantic_lifetime_s=30., moving_wrist=False):
        if not 0 < semantic_lifetime_s <= 120:raise ValueError('Invalid bounded semantic lease')
        self.moving_wrist=bool(moving_wrist)
        self.semantic_lifetime_s=float(semantic_lifetime_s)
        if profile not in ('baseline', 'scaled'):
            raise ValueError('Unknown refinement profile')
        if not math.isfinite(max_residual_px) or not 0 < max_residual_px <= 12:
            raise ValueError('Invalid prediction gate')
        self.profile, self.max_residual_px = profile, max_residual_px
        self.active = False

    def _refine(self, frame, object_id, uv, confidence):
        import base64
        if self.profile == 'scaled':
            from embodied_arm.perception.rgb_scaled_refinement import refine_rgb_landmark
        else:
            from embodied_arm.perception.rgb_face_refinement import refine_rgb_landmark
        c = frame.calibration
        return refine_rgb_landmark(base64.b64decode(frame.jpeg_base64), object_id=object_id,
            uv_1000=[uv[0]/(c.width-1)*1000, uv[1]/(c.height-1)*1000],
            confidence=confidence, width=c.width, height=c.height)

    def _area_ratio(self,frame,area,predicted):
        ratio=area/self._areas[frame.camera_id]
        if self.moving_wrist:
            from embodied_arm.perception.moving_camera_geometry import projected_face_area
            try:ratio*=projected_face_area(self._calibrations[frame.camera_id],self._xyz)/projected_face_area(frame.calibration,predicted)
            except ValueError:return float('inf')
        return ratio

    def start(self, observation: VisualObservation, frames, *, object_id: str, now: float):
        from embodied_arm.perception.camera_perception import RGBFrame
        self.active = False
        validator = VisualPhaseVerifier(VerificationConfig(max_semantic_age_s=self.semantic_lifetime_s))
        validator._object_id = object_id
        obj = validator._validate(observation, now)
        if (len(frames) != 2 or any(not isinstance(f, RGBFrame) for f in frames)
                or {f.camera_id: f.frame_id for f in frames} != dict(observation.scene.frame_ids)):
            raise ValueError('Initial RGB frame mismatch')
        self._areas, self._calibrations = {}, {}
        for f in frames:
            detection = next(d for d in observation.detections
                             if d.object_id == object_id and d.camera_id == f.camera_id)
            result = self._refine(f, object_id, detection.uv, detection.confidence)
            if not result.accepted:
                raise ValueError('Initial face unsupported:' + result.reason)
            if math.dist(result.uv, detection.uv) > self.max_residual_px:
                raise ValueError('Initial landmark disagrees with RGB face')
            self._areas[f.camera_id] = result.face_area_px2
            self._calibrations[f.camera_id] = f.calibration
        self._base, self._object, self._xyz = observation, object_id, _vec(obj.xyz)
        self._last = observation.scene.captured_at
        self._seen = set(observation.scene.frame_ids.items())
        self.active = True

    def track(self, frames, *, tcp_xyz, tcp_captured_at, now) -> DynamicTrackResult:
        from embodied_arm.perception.camera_perception import RGBFrame
        def fail(reason):
            self.active = False
            return DynamicTrackResult(False, reason, refresh_required=True)
        if not self.active:
            return fail('dynamic_phase_inactive')
        if (len(frames) != 2 or any(not isinstance(f, RGBFrame) for f in frames)
                or {f.camera_id for f in frames} != set(self._calibrations)):
            return fail('invalid_rgb_frames')
        if (not all(0 <= now-f.captured_at <= 3 for f in frames)
                or max(f.captured_at for f in frames)-min(f.captured_at for f in frames) > .05
                or any(abs(tcp_captured_at-f.captured_at) > .05 for f in frames)
                or now-self._base.semantic_captured_at > self.semantic_lifetime_s):
            return fail('stale_or_unsynchronized_phase')
        if any(f.captured_at <= self._last or (f.camera_id, f.frame_id) in self._seen for f in frames):
            return fail('replayed_rgb_frame')
        predicted = self._xyz + _vec(tcp_xyz) - _vec(self._base.tcp_xyz)
        detections = []
        for f in frames:
            from embodied_arm.perception.moving_camera_geometry import compatible_calibrations
            if not compatible_calibrations({f.camera_id:self._calibrations[f.camera_id]}, {f.camera_id:f.calibration}, moving_wrist=self.moving_wrist):
                return fail('camera_calibration_changed')
            try:
                seed = f.calibration.project(tuple(float(x) for x in predicted))
                result = self._refine(f, self._object, seed, self._base.association_confidence)
            except ValueError:
                return fail('prediction_outside_image')
            if not result.accepted:
                return fail('local_face_lost:' + result.reason)
            if (math.dist(result.uv, seed) + 3*result.pixel_sigma_px > self.max_residual_px
                    or not .5 <= self._area_ratio(f,result.face_area_px2,predicted) <= 2):
                return fail('dynamic_identity_or_prediction_mismatch')
            detections.append(PixelDetection(self._object, f.camera_id, f.frame_id,
                result.uv, self._base.association_confidence, result.pixel_sigma_px))
        self._last = min(f.captured_at for f in frames)
        self._seen.update((f.camera_id, f.frame_id) for f in frames)
        return DynamicTrackResult(True, 'observed_face_matches_phase_prediction', tuple(detections),
                                  self._base.identity_token)

    def reacquire_closure(self, frames, *, tcp_xyz, tcp_captured_at, now, workspace,
                          max_world_shift=.03, max_tcp_distance=.035):
        """Observe bounded jaw-induced sliding, then establish the carry baseline.

        Only use immediately after a commanded close while the robot is paused.
        The original RGB-derived anchor seeds the search; no desired coordinates
        become measurements. Both views must show the unique complete face, and
        stereo displacement/proximity PLUS uncertainty must fit the aperture.
        Success is identity association, explicitly NOT grasp verification.
        """
        from dataclasses import replace
        from embodied_arm.perception.camera_perception import RGBFrame, estimate_scene, PerceptionConfig
        def fail(reason):
            self.active = False
            return ClosureTrackResult(False, reason)
        if (not self.active or not 0 < max_world_shift <= .03
                or not 0 < max_tcp_distance <= .035):
            return fail('invalid_closure_phase_or_bounds')
        if (len(frames) != 2 or any(not isinstance(f, RGBFrame) for f in frames)
                or {f.camera_id for f in frames} != set(self._calibrations)):
            return fail('invalid_rgb_frames')
        if (not all(0 <= now-f.captured_at <= 3 for f in frames)
                or max(f.captured_at for f in frames)-min(f.captured_at for f in frames) > .05
                or any(abs(tcp_captured_at-f.captured_at) > .05 for f in frames)
                or now-self._base.semantic_captured_at > self.semantic_lifetime_s):
            return fail('stale_or_unsynchronized_closure')
        if any(f.captured_at <= self._last or (f.camera_id,f.frame_id) in self._seen for f in frames):
            return fail('replayed_rgb_frame')
        tcp = _vec(tcp_xyz)
        # Closure is not a transport maneuver. A large encoder motion cannot
        # expand the association search or renew the prior semantic identity.
        if np.linalg.norm(tcp-_vec(self._base.tcp_xyz)) + 6*self._base.tcp_std_m > .015:
            return fail('closure_tcp_moved')
        detections = []
        for frame in frames:
            from embodied_arm.perception.moving_camera_geometry import compatible_calibrations
            if not compatible_calibrations({frame.camera_id:self._calibrations[frame.camera_id]}, {frame.camera_id:frame.calibration}, moving_wrist=self.moving_wrist):
                return fail('camera_calibration_changed')
            try:
                seed = frame.calibration.project(tuple(float(x) for x in self._xyz))
                face = self._refine(frame, self._object, seed, self._base.association_confidence)
            except ValueError:
                return fail('closure_prediction_outside_image')
            if not face.accepted:
                return fail('closure_face_lost:' + face.reason)
            if not .5 <= self._area_ratio(frame,face.face_area_px2,self._xyz) <= 2:
                return fail('closure_face_identity_mismatch')
            detections.append(PixelDetection(self._object,frame.camera_id,frame.frame_id,
                face.uv,self._base.association_confidence,face.pixel_sigma_px))
        scene = estimate_scene(tuple(frames), tuple(detections), now=now, workspace=workspace,
            config=PerceptionConfig(max_frame_age_s=3.,max_position_std_m=.005,max_ray_separation_m=.006))
        try:
            current = scene.require(self._object,now=now)
        except ValueError:
            return fail('closure_stereo_uncertain')
        initial = self._base.scene.objects[self._object]
        shift_bound = 3*(current.position_std_m + initial.position_std_m)
        near_bound = 3*(current.position_std_m + self._base.tcp_std_m)
        if np.linalg.norm(_vec(current.xyz)-self._xyz) + shift_bound > max_world_shift:
            return fail('closure_object_outside_aperture')
        if np.linalg.norm(_vec(current.xyz)-tcp) + near_bound > max_tcp_distance:
            return fail('closure_object_not_near_tcp')
        associated = replace(self._base, scene=scene, detections=tuple(detections),
            tcp_xyz=tuple(float(x) for x in tcp),tcp_captured_at=tcp_captured_at,
            source='dynamic_rgb_track')
        # The new measured offset becomes the carry anchor, while identity and
        # semantic deadline remain immutable. No success is reported as a grasp.
        self.start(associated,frames,object_id=self._object,now=now)
        return ClosureTrackResult(True,'observed_face_inside_closing_aperture',scene,
                                  tuple(detections),self._base.identity_token)


@dataclass(frozen=True)
class ClosureTrackResult:
    accepted: bool
    reason: str
    scene: EstimatedScene | None = None
    detections: tuple[PixelDetection, ...] = ()
    identity_token: str | None = None
    physical_contact_verified: bool = False
