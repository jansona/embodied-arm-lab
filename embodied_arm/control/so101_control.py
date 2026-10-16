"""RGB-only scene boundary and symbolic grounding for SO-ARM101.

No simulator body pose, depth buffer, segmentation ID or evaluation metric is
read here. Camera calibration and encoder/FK proprioception are allowed inputs.
The initial profile observes a paused scene; it is not dynamic real-time vision.
"""
from __future__ import annotations
import base64
import math
import threading
import time
from embodied_arm.perception.camera_perception import CameraCalibration,RGBFrame,PerceptionConfig,estimate_scene

FROZEN_WINDOW_SECONDS=30.
OBSERVATION_VOLUME=((0.,.4),(-.25,.25),(-.005,.35))
MAX_POSITION_STD_M=.003


def robot_only(raw):
    if not isinstance(raw,dict) or raw.get('name')!='SO-ARM101' or raw.get('arm_dof')!=5:raise ValueError('SO101 proprioception required')
    keys=('name','arm_dof','ee','joints','joint_limits','gripper_openness','gripper_angle',
          'paused','busy','epoch','actions','workspace','cube_edge_m')
    out={k:raw[k] for k in keys}
    for key,size in [('ee',3),('joints',5)]:
        if len(out[key])!=size or not all(type(v) in (int,float) and math.isfinite(v) for v in out[key]):raise ValueError('Invalid proprioception')
    if not 0<=out['gripper_openness']<=1 or out['cube_edge_m']!=.02:raise ValueError('Invalid SO101 capability')
    return out


class CameraSceneSource:
    def __init__(self,simulation,vision=None,clock=time.monotonic,refine_rgb=False,refinement_profile="baseline",roi_tracking=False):
        if refinement_profile not in ("baseline","scaled"):raise ValueError("Unknown refinement profile")
        self.refinement_profile=refinement_profile;self.roi_tracking=bool(roi_tracking);self.tracking_rois=None
        self.sim=simulation;self.vision=vision;self.clock=clock;self.refine_rgb=bool(refine_rgb)
        self.lock=threading.RLock();self.generation=0;self.scene=None;self.epoch=None;self.actions=None
        self.reason='vision_unconfigured';self.pending=False;self.last_vision=None
        self.last_rgb_frames=None;self.last_pixel_detections=None
        self.tracker=None;self.tracker_calibrations=None;self.semantic_deadline=None;self.required_objects=('red','blue','goal')
    def cancel(self):
        with self.lock:
            self.generation+=1;self.scene=None;self.reason='cancelled'
            if self.tracker:self.tracker.clear()
            self.tracker=None;self.semantic_deadline=None
        if self.vision and hasattr(self.vision,'cancel'):self.vision.cancel()
    def status(self):
        with self.lock:
            return {'configured':self.vision is not None,'pending':self.pending,'reason':self.reason,
                    'source':'calibrated_rgb_triangulation','mode':'paused_scene_validation','rgb_refinement_enabled':self.refine_rgb,
                    'max_position_std_m':MAX_POSITION_STD_M,'frozen_window_seconds':FROZEN_WINDOW_SECONDS,
                    'tracking':self.tracker.status() if self.tracker else None,
                    'refinement_profile':self.refinement_profile,'roi_tracking':self.roi_tracking,'observed_roi_bounds':self.tracking_rois,
                    'last_vision':self.last_vision,'scene':None if self.scene is None else self.scene.to_dict()}
    def refresh(self,required_objects=('red','blue','goal'),*,expected_generation=None):
        scope=tuple(required_objects)
        if not scope or len(set(scope))!=len(scope) or any(x not in ('red','blue','goal') for x in scope):raise ValueError('Invalid observation scope')
        return self._observe(False,scope,expected_generation=expected_generation)
    def ensure_brain_scene(self):
        """Use valid RGB evidence or acquire it on demand, never on a timer."""
        generation=self.generation
        try:return self.brain_scene()
        except ValueError:pass
        if self.tracker is not None:
            try:
                self.track()
                return self.brain_scene()
            except ValueError:pass
        if generation!=self.generation:raise ValueError('cancelled')
        provider=self.vision.status() if self.vision and hasattr(self.vision,'status') else {}
        if not provider.get('available',provider.get('configured',False)):
            raise ValueError('vision_unavailable')
        try:self.refresh(expected_generation=generation)
        except Exception:
            if generation!=self.generation:raise ValueError('cancelled') from None
            reason=self.status()['reason']
            allowed={'network_error','timeout','expired','call_budget_exhausted','landmark_uncertain',
                     'occluded','occluded_or_missing_feature','uncertain_detection','low_confidence',
                     'unavailable','scene_changed_during_vision','invalid_or_stale_frame','stale_frame'}
            raise ValueError(reason if reason in allowed else 'observation_unavailable') from None
        return self.brain_scene()

    def track(self):return self._observe(True)
    def _observe(self,tracking,scope=None,expected_generation=None):
        with self.lock:
            if expected_generation is not None and expected_generation!=self.generation:raise ValueError('cancelled')
            if self.pending:raise ValueError('Vision request already pending')
            if tracking and (not self.refine_rgb or self.tracker is None):raise ValueError('No active RGB tracking session')
            if not tracking and self.vision is None:raise ValueError('Vision is not configured')
            before=robot_only(self.sim.proprioception())
            if before['busy'] or not before['paused']:raise ValueError('Pause an idle scene before RGB observation')
            if not tracking:self.required_objects=scope
            self.pending=True;generation=self.generation;self.scene=None;self.reason='tracking' if tracking else 'observing'
        failure='rgb_tracking_rejected' if tracking else 'vision_request_failed'
        try:
            packet=self.sim.capture_frames(roi_bounds=self.tracking_rois) if tracking and self.tracking_rois else self.sim.capture_frames()
            vision_packet={k:packet[k] for k in ('frame_id','captured_at','images')}
            with self.lock:
                if generation!=self.generation:raise ValueError('cancelled')
            if tracking:
                if packet['calibrations']!=self.tracker_calibrations:raise ValueError('Camera calibration changed')
                refined=self.tracker.track(vision_packet,before['epoch'],now=self.clock())
                self._record_refinement(refined,tracking=True)
                if not refined.accepted:raise ValueError('RGB tracking rejected')
                detections=list(refined.detections)
            else:
                if max(max(i['width'],i['height']) for i in vision_packet['images'])>1024:
                    from embodied_arm.perception.rgb_coarse_packet import prepare_coarse_packet,remap_result
                    prepared=prepare_coarse_packet(vision_packet);submitted=prepared.packet
                    with self.lock:
                        if generation!=self.generation:raise ValueError('cancelled')
                    result=self.vision.perceive(submitted)
                    if result.accepted:result=remap_result(result,prepared,native_packet=vision_packet,submitted_packet=submitted)
                else:
                    with self.lock:
                        if generation!=self.generation:raise ValueError('cancelled')
                    result=self.vision.perceive(vision_packet)
                self.last_vision={'accepted':bool(result.accepted),'calls':getattr(result,'calls',None),
                    'model':getattr(result,'model',None),'provenance':getattr(result,'provenance','unverified_fixture')}
                if not result.accepted:
                    candidate=getattr(result,'reason','vision_rejected')
                    failure=candidate if candidate in {'occluded','occluded_or_missing_feature','uncertain_detection','low_confidence','invalid_response','invalid_input','invalid_or_stale_frame','stale_frame','unexpected_model','incomplete_response','response_too_large','timeout','network_error','cancelled','expired','unavailable','busy','call_budget_exhausted'} else 'vision_rejected'
                    raise ValueError('Vision rejected or uncertain')
                failure='frame_mismatch'
                if result.frame_id!=packet['frame_id'] or result.captured_at!=packet['captured_at']:raise ValueError('Vision frame mismatch')
                detections=list(result.detections)
                if self.refine_rgb:
                    from embodied_arm.perception.rgb_region_tracker import RGBRegionTracker
                    self.tracker=RGBRegionTracker() if self.refinement_profile=="baseline" else RGBRegionTracker(refinement_profile=self.refinement_profile)
                    failure='rgb_face_refinement_rejected'
                    refined=self.tracker.start(vision_packet,detections,before['epoch'],now=self.clock(),required_objects=self.required_objects)
                    self._record_refinement(refined,tracking=False)
                    if not refined.accepted:raise ValueError('RGB face refinement rejected')
                    self.tracker_calibrations=packet['calibrations'];self.semantic_deadline=packet['captured_at']+60.
                    detections=list(refined.detections)
                    if self.roi_tracking:
                        if not {'red','blue'}.issubset(self.required_objects):raise ValueError('ROI tracking requires both blocks')
                        from embodied_arm.perception.rgb_roi_geometry import bounds_from_detections
                        self.tracking_rois=bounds_from_detections(packet['images'],detections,self.required_objects)
            failure='scene_changed_during_vision'
            after=robot_only(self.sim.proprioception())
            if (generation!=self.generation or not after['paused'] or after['busy']
                    or after['epoch']!=before['epoch'] or after['actions']!=before['actions']):raise ValueError('Scene changed during vision request')
            calibrations={d['camera_id']:CameraCalibration.from_dict(d) for d in packet['calibrations']}
            frames=[]
            for item in packet['images']:
                jpeg=base64.b64decode(item['data_url'].split(',',1)[1],validate=True)
                frames.append(RGBFrame.from_jpeg(packet['frame_id'],packet['captured_at'],calibrations[item['camera_id']],jpeg))
            failure='triangulation_rejected'
            scene=estimate_scene(frames,detections,now=self.clock(),workspace=OBSERVATION_VOLUME,
                 config=PerceptionConfig(max_frame_age_s=3. if tracking else FROZEN_WINDOW_SECONDS,max_position_std_m=MAX_POSITION_STD_M,max_ray_separation_m=.006))
            rejected={k:v.reason for k,v in scene.objects.items() if not v.valid}
            if rejected:
                self.last_vision['landmark_rejections']=rejected;failure='landmark_uncertain'
            for obj in self.required_objects:scene.require(obj,now=self.clock(),max_age_s=3. if tracking else FROZEN_WINDOW_SECONDS)
            with self.lock:
                if generation!=self.generation:raise ValueError('Vision cancelled')
                self.scene=scene;self.epoch=after['epoch'];self.actions=after['actions'];self.reason='estimated'
                self.last_rgb_frames=tuple(frames);self.last_pixel_detections=tuple(detections)
            return self.snapshot()
        except Exception:
            with self.lock:
                self.scene=None;self.reason=failure
                if self.tracker:self.tracker.clear()
                self.tracker=None;self.semantic_deadline=None
            raise
        finally:
            with self.lock:self.pending=False
    def observation_evidence(self):
        """Server-only RGB/encoder evidence, never simulator contacts/object poses."""
        with self.lock:
            snapshot=self.snapshot()
            if self.last_rgb_frames is None or self.last_pixel_detections is None:
                raise ValueError('Missing RGB evidence')
            return snapshot,self.last_rgb_frames,self.last_pixel_detections
    def _record_refinement(self,refined,*,tracking):
        if not tracking:self.last_vision['base_detector_provenance']=self.last_vision['provenance']
        base=self.last_vision.get('base_detector_provenance')
        self.last_vision['provenance']=refined.source if base=='qwen_rgb' else 'offline_fixture_rgb_geometry'
        self.last_vision['refinement']={'accepted':refined.accepted,'reason':refined.reason,'evidence':[v.to_dict() for v in refined.evidence]}
        self.last_vision['fresh_model_request']=not tracking
    def snapshot(self):
        with self.lock:
            if self.scene is None:raise ValueError('No validated RGB scene')
            robot=robot_only(self.sim.proprioception())
            if robot['epoch']!=self.epoch or robot['actions']!=self.actions or robot['busy']:raise ValueError('RGB observation invalidated by motion')
            if self.semantic_deadline is not None and self.clock()>=self.semantic_deadline:raise ValueError('Semantic observation expired')
            for obj in self.required_objects:self.scene.require(obj,now=self.clock(),max_age_s=FROZEN_WINDOW_SECONDS)
            return {'schema':'so101-camera-observation-v1','source':'rgb_only','required_objects':list(self.required_objects),'detector_provenance':self.last_vision['provenance'],'robot':robot,'scene':self.scene.to_dict()}
    def brain_scene(self):
        snapshot=self.snapshot()
        if set(self.required_objects)!={'red','blue','goal'}:raise ValueError('Partial scene cannot support high-level planning')
        objects=self.scene.objects;half=snapshot['robot']['cube_edge_m']/2
        centers={name:[*objects[name].xyz[:2],objects[name].xyz[2]-half] for name in ('red','blue')}
        goal=[*objects['goal'].xyz[:2],objects['goal'].xyz[2]+half]
        return {'blocks':centers,'goal':goal,'ee':snapshot['robot']['ee'],'paused':snapshot['robot']['paused'],
                'busy':snapshot['robot']['busy'],'perception_source':'rgb_only','detector_provenance':snapshot['detector_provenance'],'perception':snapshot['scene']}


def ground_symbolic(intent,snapshot):
    """Known geometric recipe using only validated camera-derived landmarks.

    This returns a preview; an executor must re-observe after motion and must not
    treat this preview as a successful grasp or permission for stale execution.
    """
    from embodied_arm.perception.camera_perception import EstimatedScene
    if set(snapshot.get('required_objects',('red','blue','goal')))!={'red','blue','goal'}:raise ValueError('Partial scene cannot support symbolic grounding')
    if snapshot.get('schema')!='so101-camera-observation-v1' or snapshot.get('source')!='rgb_only':raise ValueError('RGB provenance required')
    scene=EstimatedScene.from_dict(snapshot['scene']);robot=robot_only(snapshot['robot'])
    action=intent.get('action');obj=intent.get('object_id');target=intent.get('target_id')
    if action not in ('pick','place') or obj not in ('red','blue'):raise ValueError('Unsupported symbolic primitive')
    now=time.monotonic();point=scene.require(obj,now=now,max_age_s=FROZEN_WINDOW_SECONDS)
    if point.position_std_m>MAX_POSITION_STD_M:raise ValueError('Visual uncertainty too large')
    half=robot['cube_edge_m']/2;center=[*point.xyz[:2],point.xyz[2]-half]
    if action=='pick':
        hover=max(.055,point.xyz[2]+.025)
        if hover>.08:raise ValueError('SO101 top-down hover requires reach validation')
        return {'source':'rgb_only','object_id':obj,'landmark':point.landmark,
                'steps':[{'action':'open'},{'action':'move','target_xyz':[center[0],center[1],hover]},
                         {'action':'move','target_xyz':[center[0],center[1],center[2]+.003]},
                         {'action':'close'},{'action':'lift','height_m':hover}],
                'requires_fresh_perception_after_each_motion':True}
    if target=='goal':
        support=scene.require('goal',now=now,max_age_s=FROZEN_WINDOW_SECONDS);surface=list(support.xyz)
    elif target in ('on:red','on:blue') and target!=f'on:{obj}':
        support=scene.require(target.split(':')[1],now=now,max_age_s=FROZEN_WINDOW_SECONDS);surface=list(support.xyz)
    else:raise ValueError('Unknown target')
    # Offset is measured from RGB landmark and encoder TCP, never simulator pose.
    offset=[a-b for a,b in zip(robot['ee'],center)]
    release=[surface[0]+offset[0],surface[1]+offset[1],surface[2]+half+max(0,offset[2])+.002]
    return {'source':'rgb_only','object_id':obj,'target_id':target,
            'steps':[{'action':'move','target_xyz':[release[0],release[1],max(.06,release[2]+.02)]},
                     {'action':'move','target_xyz':release},{'action':'open'},{'action':'lift','height_m':.06}],
            'requires_verified_grasp':True,'requires_fresh_perception_after_each_motion':True}
