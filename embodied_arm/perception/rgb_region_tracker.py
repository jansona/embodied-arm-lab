"""Bounded static-hover tracking of AI-seeded regions in fresh RGB images.

No API/model calls, calibration, world coordinates, simulator, depth or supplied
masks enter this module. The original frozen RGB face refiner is reused without
modification. Its initial refined *pixel* anchors never move. A tracked result is
explicitly attributed to image geometry, not a new Qwen observation.

The caller must separately enforce static-hover task scope, robot-motion safety,
camera calibration validity and its own generation/proprioception checks. Starting
from an 18-second-old frozen-scene AI image may establish semantic anchors, but
motion must wait for track() on a fresh image. No tracker session grants motion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Callable, Sequence

from embodied_arm.perception.camera_perception import PixelDetection
from embodied_arm.perception.rgb_face_refinement import RefinedPacket, SOURCE as INITIAL_SOURCE, refine_packet

TRACKED_SOURCE = 'hybrid_qwen_region_rgb_geometry_tracked'
SEMANTIC_LIFETIME_S = 60.
MAX_TRACKING_FRAMES = 30
MAX_CURRENT_FRAME_AGE_S = 3.
MAX_INITIAL_ANCHOR_DRIFT_PX = 4.


@dataclass(frozen=True)
class _Session:
    epoch: int
    semantic_frame_id: str
    semantic_captured_at: float
    dimensions: tuple[tuple[str, int, int], ...]
    required_objects: tuple[str, ...]
    anchors: tuple[PixelDetection, ...]
    last_frame_id: str
    last_captured_at: float
    last_checked_at: float
    seen_frame_ids: frozenset[str]
    tracked_frames: int = 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _snapshot(packet):
    """Allowlisted detached metadata/RGB copy; arbitrary attached state is refused."""
    if not isinstance(packet, dict) or set(packet) != {'frame_id', 'captured_at', 'images'}:
        raise ValueError('invalid_packet')
    frame_id, captured_at = packet['frame_id'], packet['captured_at']
    if (not isinstance(frame_id, str) or not 1 <= len(frame_id) <= 128
            or any(not (c.isalnum() or c in '_-.:') for c in frame_id)
            or not _number(captured_at)):
        raise ValueError('invalid_packet')
    if not isinstance(packet['images'], (list, tuple)) or len(packet['images']) != 2:
        raise ValueError('invalid_packet')
    images = []
    cameras = set()
    for item in packet['images']:
        if (not isinstance(item, dict) or set(item) != {'camera_id', 'width', 'height', 'data_url'}
                or item['camera_id'] not in ('left', 'right') or item['camera_id'] in cameras
                or any(type(item[key]) is not int or not 32 <= item[key] <= 4096 for key in ('width', 'height'))
                or not isinstance(item['data_url'], str)
                or not item['data_url'].startswith('data:image/jpeg;base64,')
                or len(item['data_url']) > 12*1024*1024):
            raise ValueError('invalid_packet')
        cameras.add(item['camera_id'])
        images.append(dict(item))
    return {'frame_id': frame_id, 'captured_at': float(captured_at), 'images': images}


def _dimensions(packet):
    return tuple(sorted((item['camera_id'], item['width'], item['height']) for item in packet['images']))


class RGBRegionTracker:
    """One nonrenewing semantic lease, one immutable set of image-pixel anchors.

    start(packet, detections, epoch, now=None) runs the frozen refiner and retains
    its initial result. track(packet, epoch, now=None) requires genuinely newer
    current RGB frames and never refreshes semantic age or anchors. Epoch is an
    integer generation token, not a simulator/object-state object. All guard or
    required-object refiner failures clear the lease and return an empty detection
    tuple. required_objects is fixed at start; nonrequired points are never cached
    or returned for control, even if the initial full-scene refinement found them.
    """
    def __init__(self, *, clock: Callable[[], float] | None = None, refinement_profile="baseline", semantic_lifetime_s=SEMANTIC_LIFETIME_S, max_tracking_frames=MAX_TRACKING_FRAMES, moving_wrist=False, max_observation_age_s=MAX_CURRENT_FRAME_AGE_S):
        if refinement_profile not in ("baseline","scaled"):raise ValueError("Unknown RGB refinement profile")
        if not 0 < semantic_lifetime_s <= 420 or type(max_tracking_frames) is not int or not 1 <= max_tracking_frames <= 120:raise ValueError('Invalid bounded tracking lease')
        if type(max_observation_age_s) not in (int,float) or not math.isfinite(max_observation_age_s) or not 1<=max_observation_age_s<=60:raise ValueError('Invalid observation age budget')
        self.max_observation_age_s=float(max_observation_age_s)
        self.moving_wrist=bool(moving_wrist);self._geometry=None
        self.semantic_lifetime_s=float(semantic_lifetime_s);self.max_tracking_frames=max_tracking_frames
        self.refinement_profile=refinement_profile
        self.initial_source=INITIAL_SOURCE if refinement_profile=="baseline" else "hybrid_qwen_region_rgb_geometry_scaled"
        self.tracked_source=self.initial_source+"_tracked"
        if clock is not None and not callable(clock):
            raise ValueError('clock must be callable')
        self._clock = time.monotonic if clock is None else clock
        self._lock = threading.RLock()
        self._session: _Session | None = None
        self._last_reason = 'not_started'
        self.last_drift=[];self.last_fit_details=[];self.last_failure_details=[]

    def _refine(self,packet,detections):
        if self.refinement_profile=='scaled':
            from embodied_arm.perception.rgb_scaled_refinement import refine_packet as scaled_refine
            result=scaled_refine(packet,detections)
        else:result=refine_packet(packet,detections)
        visible=[d for d in detections if d.visibility=='visible']
        self.last_fit_details=[{'object_id':d.object_id,'camera_id':d.camera_id,
            'reason':e.reason,'accepted':bool(e.accepted),'roi_bounds':list(e.roi_bounds),
            'uv':list(e.uv) if e.uv is not None else None,
            'coarse_uv':list(e.coarse_uv) if e.coarse_uv is not None else None}
            for d,e in zip(visible,result.evidence)][:6]
        return result

    def clear(self):
        with self._lock:
            self._session = None
            self._last_reason = 'cleared'

    def status(self) -> dict:
        """Return provenance/limits only; active means initialized, not motion-safe."""
        with self._lock:
            session = self._session
            result = {'active': session is not None, 'reason': self._last_reason,
                      'profile': 'static_hover_only', 'source': self.tracked_source,
                      'semantic_lifetime_s': self.semantic_lifetime_s,
                      'max_tracking_frames': self.max_tracking_frames,
                      'max_current_frame_age_s': self.max_observation_age_s,
                      'max_initial_anchor_drift_px': MAX_INITIAL_ANCHOR_DRIFT_PX}
            if session is not None:
                result.update(epoch=session.epoch, semantic_frame_id=session.semantic_frame_id,
                              semantic_captured_at=session.semantic_captured_at,
                              semantic_deadline=session.semantic_captured_at+self.semantic_lifetime_s,
                              last_frame_id=session.last_frame_id,
                              last_captured_at=session.last_captured_at,
                              tracked_frames=session.tracked_frames,
                              required_objects=list(session.required_objects))
            return result

    def _fail(self, reason, *, evidence=(), source=None):
        source=self.tracked_source if source is None else source
        self._session = None
        self._last_reason = reason
        self.last_failure_details=[d for d in self.last_fit_details if not d['accepted']] if evidence else []
        return RefinedPacket(False, reason, (), tuple(evidence), source)

    def _time(self, now):
        value = self._clock() if now is None else now
        if not _number(value):
            raise ValueError('invalid_clock')
        return float(value)

    @staticmethod
    def _validate_result(result, packet, required_objects):
        if not isinstance(result, RefinedPacket):
            return False
        if any(not isinstance(item, PixelDetection) for item in result.detections):
            return False
        required = tuple(item for item in result.detections if item.object_id in required_objects)
        evidence = tuple(item for item in result.evidence if item.object_id in required_objects)
        expected = {(camera, obj) for camera in ('left', 'right') for obj in required_objects}
        # The full refiner can report accepted=False solely because an unrelated
        # landmark is hidden. Validate every required stereo pair independently.
        if (len(required) != len(expected) or len(evidence) != len(expected)
                or any(item.visibility != 'visible' or item.frame_id != packet['frame_id'] for item in required)
                or {(item.camera_id, item.object_id) for item in required} != expected
                or any(not item.accepted for item in evidence)
                or any(sum(item.object_id == obj for item in evidence) != 2 for obj in required_objects)):
            return False
        return True

    def start(self, packet: dict, canonical_ai_detections: Sequence[PixelDetection], epoch: int,
              *, now: float | None = None,
              required_objects: Sequence[str] = ('red', 'blue', 'goal'), calibrations=None) -> RefinedPacket:
        with self._lock:
            # A new semantic start replaces any prior lease even if it fails.
            self._session = None
            if type(epoch) is not int or epoch < 0:
                return self._fail('invalid_epoch', source=self.initial_source)
            if (not isinstance(required_objects, (tuple, list)) or not 1 <= len(required_objects) <= 3
                    or any(type(obj) is not str or obj not in ('red', 'blue', 'goal') for obj in required_objects)
                    or len(set(required_objects)) != len(required_objects)):
                return self._fail('invalid_required_objects', source=self.initial_source)
            required_objects = tuple(obj for obj in ('red', 'blue', 'goal') if obj in required_objects)
            try:
                stamp = self._time(now)
                snapshot = _snapshot(packet)
            except ValueError as exc:
                return self._fail(str(exc), source=self.initial_source)
            age = stamp-snapshot['captured_at']
            if age < 0:
                return self._fail('future_frame', source=self.initial_source)
            if age >= self.semantic_lifetime_s:
                return self._fail('semantic_expired', source=self.initial_source)
            try:
                result = self._refine(snapshot, canonical_ai_detections)
            except (ValueError, TypeError, OverflowError, OSError):
                return self._fail('initial_refinement_error', source=self.initial_source)
            if not self._validate_result(result, snapshot, required_objects):
                return self._fail('initial_refinement_failed', evidence=getattr(result, 'evidence', ()), source=self.initial_source)
            try:
                completed_at = self._time(now)
            except ValueError as exc:
                return self._fail(str(exc), source=self.initial_source)
            if completed_at < stamp or completed_at-snapshot['captured_at'] >= self.semantic_lifetime_s:
                return self._fail('semantic_expired', source=self.initial_source)
            required_detections = tuple(item for item in result.detections if item.object_id in required_objects)
            required_evidence = tuple(item for item in result.evidence if item.object_id in required_objects)
            self._session = _Session(epoch, snapshot['frame_id'], snapshot['captured_at'],
                                     _dimensions(snapshot), required_objects, required_detections,
                                     snapshot['frame_id'], snapshot['captured_at'], completed_at,
                                     frozenset((snapshot['frame_id'],)))
            if self.moving_wrist:
                from embodied_arm.perception.moving_camera_geometry import frames_from_packet
                from embodied_arm.perception.camera_perception import estimate_scene, PerceptionConfig
                try:
                    frames=frames_from_packet(dict(snapshot,calibrations=calibrations))
                    scene=estimate_scene(frames,required_detections,now=stamp,
                        workspace=((0.,.4),(-.25,.25),(-.005,.35)),
                        config=PerceptionConfig(max_frame_age_s=60.,max_position_std_m=.003,max_ray_separation_m=.006))
                    points={obj:scene.require(obj,now=stamp,max_age_s=60.).xyz for obj in required_objects}
                    self._geometry=({f.camera_id:f.calibration for f in frames},points)
                except (ValueError,TypeError,KeyError):
                    return self._fail('initial_moving_geometry_rejected',source=self.initial_source)
            self._last_reason = 'semantic_anchors_initialized'
            return RefinedPacket(True, 'required_visible_faces_refined', required_detections,
                                 required_evidence, self.initial_source)

    def projected_seeds(self, calibrations):
        if not self.moving_wrist or self._geometry is None or self._session is None:
            raise ValueError('Moving RGB anchors unavailable')
        from embodied_arm.perception.moving_camera_geometry import compatible_calibrations,reproject_detections
        from embodied_arm.perception.camera_perception import CameraCalibration
        current={c['camera_id']:CameraCalibration.from_dict(c) for c in calibrations}
        original,points=self._geometry
        if not compatible_calibrations(original,current,moving_wrist=True):raise ValueError('Camera calibration changed')
        return reproject_detections(self._session.anchors,points,current)

    def track(self, packet: dict, epoch: int, *, now: float | None = None, calibrations=None) -> RefinedPacket:
        with self._lock:
            session = self._session
            if session is None:
                return self._fail('no_active_semantic_session')
            if type(epoch) is not int or epoch != session.epoch:
                return self._fail('epoch_changed')
            try:
                stamp = self._time(now)
                snapshot = _snapshot(packet)
            except ValueError as exc:
                return self._fail(str(exc))
            if stamp < session.last_checked_at:
                return self._fail('clock_moved_backwards')
            if stamp < session.semantic_captured_at or stamp-snapshot['captured_at'] < 0:
                return self._fail('future_frame')
            if stamp-session.semantic_captured_at >= self.semantic_lifetime_s:
                return self._fail('semantic_expired')
            if session.tracked_frames >= self.max_tracking_frames:
                return self._fail('tracking_frame_budget')
            if stamp-snapshot['captured_at'] > self.max_observation_age_s:
                return self._fail('stale_current_frame')
            if _dimensions(snapshot) != session.dimensions:
                return self._fail('camera_layout_changed')
            if snapshot['frame_id'] in session.seen_frame_ids:
                return self._fail('reused_frame_id')
            if snapshot['captured_at'] <= session.last_captured_at:
                return self._fail('nonincreasing_capture_time')
            # Always reuse INITIAL refined semantic anchors, never the previous
            # result. Only frame provenance changes for the new image pair.
            initial_anchors=session.anchors
            if self.moving_wrist:
                from embodied_arm.perception.moving_camera_geometry import compatible_calibrations,reproject_detections
                from embodied_arm.perception.camera_perception import CameraCalibration
                try:
                    current={c['camera_id']:CameraCalibration.from_dict(c) for c in calibrations}
                    original,points=self._geometry
                    if not compatible_calibrations(original,current,moving_wrist=True):
                        return self._fail('camera_calibration_changed')
                    initial_anchors=reproject_detections(initial_anchors,points,current)
                except (ValueError,TypeError,KeyError):return self._fail('moving_view_projection_rejected')
            anchors = {(item.camera_id, item.object_id): item for item in initial_anchors}
            # The unchanged refiner requires six schema slots. Nonrequired slots
            # explicitly have no pixel anchor and are skipped, never reused.
            seeds = tuple(replace(anchors[(camera, obj)], frame_id=snapshot['frame_id'])
                          if (camera, obj) in anchors else
                          PixelDetection(obj, camera, snapshot['frame_id'], None, 0., None, 'unknown',
                                         'goal_center' if obj == 'goal' else 'top_face_center')
                          for obj in ('red', 'blue', 'goal') for camera in ('left', 'right'))
            try:
                result = self._refine(snapshot, seeds)
            except (ValueError, TypeError, OverflowError, OSError):
                return self._fail('tracking_refinement_error')
            evidence = tuple(replace(item, source=self.tracked_source) for item in getattr(result, 'evidence', ())
                             if item.object_id in session.required_objects)
            if not self._validate_result(result, snapshot, session.required_objects):
                return self._fail('tracking_refinement_failed', evidence=evidence)
            required_detections = tuple(item for item in result.detections if item.object_id in session.required_objects)
            self.last_drift=[{'object_id':item.object_id,'camera_id':item.camera_id,
                              'initial_uv':anchors[(item.camera_id,item.object_id)].uv,'observed_uv':item.uv,
                              'distance_px':math.dist(item.uv,anchors[(item.camera_id,item.object_id)].uv)}
                             for item in required_detections]
            if any(item['distance_px']>MAX_INITIAL_ANCHOR_DRIFT_PX for item in self.last_drift):
                return self._fail('initial_anchor_drift', evidence=evidence)
            try:
                completed_at = self._time(now)
            except ValueError as exc:
                return self._fail(str(exc))
            if completed_at < stamp:
                return self._fail('clock_moved_backwards')
            if completed_at-session.semantic_captured_at >= self.semantic_lifetime_s:
                return self._fail('semantic_expired', evidence=evidence)
            if completed_at-snapshot['captured_at'] > self.max_observation_age_s:
                return self._fail('stale_current_frame', evidence=evidence)
            self._session = replace(session, last_frame_id=snapshot['frame_id'],
                                    last_captured_at=snapshot['captured_at'], last_checked_at=completed_at,
                                    seen_frame_ids=session.seen_frame_ids | {snapshot['frame_id']},
                                    tracked_frames=session.tracked_frames+1)
            self._last_reason = 'fresh_rgb_regions_tracked'
            return RefinedPacket(True, 'fresh_rgb_regions_tracked', required_detections,
                                 evidence, self.tracked_source)
