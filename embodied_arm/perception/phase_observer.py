"""Phase-scoped RGB association; prediction only seeds actual image measurements.

The existing static observer is used only to initialize semantic identity. After
that, fresh captures and explicit static/carried phases own object association.
No camera depth, segmentation IDs, contact data or simulator object poses enter.
"""
import base64
from dataclasses import asdict, replace
import time
import uuid
from embodied_arm.perception.camera_perception import CameraCalibration, RGBFrame, EstimatedScene, estimate_scene, PerceptionConfig
from embodied_arm.perception.rgb_region_tracker import RGBRegionTracker
from embodied_arm.control.so101_control import robot_only, OBSERVATION_VOLUME
from embodied_arm.perception.visual_feedback import DynamicRGBPhaseTracker, VisualObservation, VerificationConfig, verify_lift, verify_release

class PhaseSceneObserver:
    validated = False

    def __init__(self, source, *, clock=time.monotonic):
        self.source = source
        self.sim = source.sim
        self.clock = clock
        self.latest = None
        self.dynamic = None
        self.static = None
        self.phase_id = None
        self.object_id = None
        self.phase = 'inactive'
        self.frames = None
        self.detections = None
        self.semantic_at = None
        self.frame_count=0;self.max_frames=120;self.lease_seconds=120.
        self.closure_action=None
        self.required_objects=("red","blue")
        self.goal_memory=None;self.goal_tracker=None;self.goal_detections=();self.task_action="pick"

    def configure_task(self,action):
        if action not in ('pick','place'):raise ValueError('Unsupported phase task')
        self.task_action=action
        self.required_objects=('red','blue')

    @staticmethod
    def _packet(frames):
        first = frames[0]
        return {'frame_id': first.frame_id, 'captured_at': first.captured_at, 'images': [{'camera_id': f.camera_id, 'width': f.calibration.width, 'height': f.calibration.height, 'data_url': f.data_url} for f in frames]}

    def _evidence(self, snapshot, detections, source):
        result = dict(snapshot)
        result['visual_evidence'] = {'phase_id': self.phase_id, 'identity_token': self.identity_token, 'association_candidates': 1, 'association_confidence': min((d.confidence for d in detections if d.object_id == self.object_id)), 'source': source, 'semantic_captured_at': self.semantic_at, 'tcp_captured_at': snapshot['scene']['captured_at'], 'detections': [d.to_dict() for d in detections]}
        return result

    def begin_phase(self, phase, object_id):
        if object_id not in ('red', 'blue'):
            raise ValueError('Unknown object')
        if self.latest is None or self.object_id != object_id:
            self.source.track()
            snapshot, frames, detections = self.source.observation_evidence()
            self.object_id = object_id
            self.phase_id = uuid.uuid4().hex
            self.identity_token = uuid.uuid4().hex
            self.semantic_at = self.source.semantic_deadline - 60.0 if self.source.semantic_deadline is not None else frames[0].captured_at
            self.frames = frames
            self.detections = detections
            self.latest = self._evidence(snapshot, detections, 'dynamic_rgb_track')
            self._start_static(self.required_objects)
            if self.task_action=='place':
                from embodied_arm.perception.landmark_memory import FixedGoalMemory
                self.goal_memory=FixedGoalMemory(self.lease_seconds)
                self.goal_memory.remember(snapshot,frames,now=self.clock())
                self.goal_detections=tuple(d for d in detections if d.object_id=='goal')
                self.goal_tracker=RGBRegionTracker(clock=self.clock,refinement_profile=self.source.refinement_profile,semantic_lifetime_s=self.lease_seconds,max_tracking_frames=self.max_frames)
                self.goal_tracker.start(self._packet(frames),detections,snapshot['robot']['epoch'],now=self.clock(),required_objects=('goal',))
        self.phase = phase
        if (phase in ('closing','lift') or self.task_action=='place' and phase=='approach') and self.dynamic is None:
            from embodied_arm.perception.visual_feedback import _from_snapshot
            self.dynamic = DynamicRGBPhaseTracker(profile=self.source.refinement_profile,semantic_lifetime_s=self.lease_seconds)
            self.dynamic.start(_from_snapshot(self.latest), self.frames, object_id=object_id, now=self.clock())
            point=EstimatedScene.from_dict(self.latest['scene']).objects[object_id]
            self.dynamic_seed=(point.xyz,tuple(self.latest['robot']['ee']))
            self._start_static(tuple((x for x in self.required_objects if x != object_id)))
        if phase=='closing':self.closure_action=self.latest['robot']['actions']
        elif phase == 'released':
            self.dynamic = None
            self._start_static(self.required_objects)
        return self.observe()

    def _start_static(self, required):
        self.static = RGBRegionTracker(clock=self.clock, refinement_profile=self.source.refinement_profile,semantic_lifetime_s=self.lease_seconds,max_tracking_frames=self.max_frames)
        result = self.static.start(self._packet(self.frames), self.detections, self.latest['robot']['epoch'], now=self.clock(), required_objects=required)
        if not result.accepted:
            raise ValueError('static_association_rejected')

    def observe(self):
        try:
            result=self._observe_impl()
            self.last_error=None
            return result
        except ValueError as exc:
            self.last_error=str(exc)[:160]
            raise

    def _observe_impl(self):
        if self.latest is None:
            raise ValueError('Phase not initialized')
        before = robot_only(self.sim.proprioception())
        if before['epoch'] != self.latest['robot']['epoch']:
            raise ValueError('Scene reset invalidated phase')
        if self.clock() - self.semantic_at > self.lease_seconds:
            raise ValueError('Semantic identity lease expired')
        if self.frame_count>=self.max_frames:raise ValueError('Phase frame budget exhausted')
        if before['busy'] or not before['paused']:
            raise ValueError('Observation requires paused robot')
        bounds=None
        if self.source.roi_tracking:
            from embodied_arm.perception.rgb_roi_geometry import bounds_from_detections
            seeds=list(self.detections)+list(self.goal_detections)
            if self.dynamic:
                xyz,tcp=self.dynamic_seed
                predicted=[p+now-old for p,now,old in zip(xyz,before['ee'],tcp)]
                calibrations={f.camera_id:f.calibration for f in self.frames}
                seeds=[replace(d,uv=calibrations[d.camera_id].project(predicted)) if d.object_id==self.object_id else d for d in seeds]
            scope=(*self.required_objects,'goal') if self.goal_detections else self.required_objects
            bounds=bounds_from_detections(self._packet(self.frames)['images'],seeds,scope)
        packet = self.sim.capture_frames(roi_bounds=bounds) if bounds else self.sim.capture_frames()
        calibrations = {x['camera_id']: CameraCalibration.from_dict(x) for x in packet['calibrations']}
        frames = tuple((RGBFrame.from_jpeg(packet['frame_id'], packet['captured_at'], calibrations[x['camera_id']], base64.b64decode(x['data_url'].split(',', 1)[1])) for x in packet['images']))
        self.last_captured_frames=frames
        previous_calibrations={f.camera_id:f.calibration for f in self.frames}
        if any(f.calibration!=previous_calibrations.get(f.camera_id) for f in frames):raise ValueError('Camera calibration changed')
        tracked = self.static.track(self._packet(frames), before['epoch'], now=self.clock())
        if not tracked.accepted:
            raise ValueError('Static RGB association lost: '+tracked.reason)
        detections = list(tracked.detections)
        if self.dynamic:
            if self.closure_action is not None and before['actions']>self.closure_action:
                moved=self.dynamic.reacquire_closure(frames,tcp_xyz=before['ee'],tcp_captured_at=packet['captured_at'],now=self.clock(),workspace=OBSERVATION_VOLUME)
                if moved.accepted:
                    self.closure_action=None
                    self.dynamic_seed=(moved.scene.objects[self.object_id].xyz,tuple(before['ee']))
            else:
                moved = self.dynamic.track(frames, tcp_xyz=before['ee'], tcp_captured_at=packet['captured_at'], now=self.clock())
            if not moved.accepted:
                raise ValueError('Dynamic RGB association lost; semantic refresh required')
            detections += list(moved.detections)
        if self.goal_tracker and self.goal_tracker.status()['active']:
            goal_result=self.goal_tracker.track(self._packet(frames),before['epoch'],now=self.clock())
            if goal_result.accepted:
                goal_scene=estimate_scene(frames,goal_result.detections,now=self.clock(),workspace=OBSERVATION_VOLUME,config=PerceptionConfig(max_frame_age_s=3,max_position_std_m=.003,max_ray_separation_m=.006))
                self.goal_memory.check_visible(goal_scene,now=self.clock())
            elif goal_result.reason=='initial_anchor_drift':
                self.goal_memory.clear();raise ValueError('Fixed marker moved')
        after = robot_only(self.sim.proprioception())
        if after != before:
            raise ValueError('Robot changed during capture')
        scene = estimate_scene(frames, detections, now=self.clock(), workspace=OBSERVATION_VOLUME, config=PerceptionConfig(max_frame_age_s=3, max_position_std_m=0.003, max_ray_separation_m=0.006))
        for obj in self.required_objects:
            scene.require(obj, now=self.clock(), max_age_s=3)
        snapshot = {'schema': 'so101-camera-observation-v1', 'source': 'rgb_only', 'required_objects': list(self.required_objects), 'robot': after, 'scene': scene.to_dict(), 'detector_provenance': 'phase_scoped_rgb_geometry'}
        self.frame_count+=1
        self.latest = self._evidence(snapshot, detections, 'dynamic_rgb_track')
        self.frames = frames
        self.detections = tuple(detections)
        return self.latest

    def snapshot(self):
        if self.latest is None:
            raise ValueError('No phase observation')
        robot=robot_only(self.sim.proprioception())
        prior=self.latest['robot']
        if robot['epoch']!=prior['epoch'] or robot['actions']!=prior['actions'] or robot['busy'] or not robot['paused']:raise ValueError('Observation invalidated by robot motion')
        scene = EstimatedScene.from_dict(self.latest['scene'])
        for obj in self.required_objects:
            scene.require(obj, now=self.clock(), max_age_s=3)
        return self.latest

    def verify_lift(self, before, after, obj):
        return asdict(verify_lift(before, after, obj, now=self.clock(),config=VerificationConfig(max_semantic_age_s=self.lease_seconds,max_phase_s=self.lease_seconds)))

    def fixed_goal(self):
        if self.goal_memory is None:raise ValueError('No observed fixed goal')
        return self.goal_memory.read(robot_only(self.sim.proprioception()),self.frames,now=self.clock())

    def verify_release(self, held, released, retreated, obj, support_xyz):
        target = [support_xyz[0], support_xyz[1], support_xyz[2] + 0.02]
        from embodied_arm.perception.camera_perception import ObjectEstimate
        goal=ObjectEstimate.from_dict(self.fixed_goal()['estimate'])
        return asdict(verify_release(held, released, retreated, obj, target, target_std_m=goal.position_std_m, now=self.clock(),config=VerificationConfig(max_semantic_age_s=self.lease_seconds,max_phase_s=self.lease_seconds)))

    def cancel(self):
        if self.static:
            self.static.clear()
        self.dynamic = None
        self.latest = None
        if self.goal_memory:self.goal_memory.clear()
        self.goal_tracker=None
        self.goal_detections=()
