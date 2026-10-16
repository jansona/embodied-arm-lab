"""Bounded memory of an RGB-observed marker printed on a fixed bench.

Occlusion does not create a new measurement. The original covariance, timestamp,
frames and calibrated camera transforms remain attached to the stored estimate.
"""
from copy import deepcopy
import math
from embodied_arm.perception.camera_perception import EstimatedScene

class FixedGoalMemory:
    def __init__(self,max_age_s=120.,*,moving_wrist=False):
        if not 0<max_age_s<=120:raise ValueError('Invalid marker-memory lease')
        self.moving_wrist=bool(moving_wrist)
        self.max_age_s=max_age_s;self.record=None
    def remember(self,snapshot,frames,*,now):
        if snapshot.get('source')!='rgb_only':raise ValueError('RGB source required')
        scene=EstimatedScene.from_dict(snapshot['scene']);goal=scene.require('goal',now=now,max_age_s=3)
        if (len(frames)!=2 or {f.camera_id:f.frame_id for f in frames}!=dict(scene.frame_ids)
                or any(abs(f.captured_at-scene.captured_at)>.05 for f in frames)):
            raise ValueError('Marker frame association mismatch')
        self.record={'estimate':goal.to_dict(),'captured_at':scene.captured_at,
            'frame_ids':dict(scene.frame_ids),'epoch':snapshot['robot']['epoch'],
            'calibrations':{f.camera_id:f.calibration.to_dict() for f in frames},
            'assumption':'marker_printed_on_fixed_bench','source':'prior_rgb_measurement'}
    def read(self,robot,frames,*,now):
        r=self.record
        if r is None or not 0<=now-r['captured_at']<=self.max_age_s:raise ValueError('Marker memory expired')
        if robot['epoch']!=r['epoch']:raise ValueError('Marker memory invalidated by reset')
        current={f.camera_id:f.calibration.to_dict() for f in frames}
        original=deepcopy(r['calibrations'])
        if self.moving_wrist and set(current)==set(original):
            current=deepcopy(current)
            current['right'].pop('world_from_camera');original['right'].pop('world_from_camera')
        if current!=original:raise ValueError('Marker calibration changed')
        return deepcopy(r)
    def check_visible(self,scene,*,now):
        if self.record is None:return
        point=scene.require('goal',now=now,max_age_s=3)
        if math.dist(point.xyz,self.record['estimate']['xyz'])>.003:
            self.record=None;raise ValueError('Previously fixed marker moved')
    def clear(self):self.record=None
