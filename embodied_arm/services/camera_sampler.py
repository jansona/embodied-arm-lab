"""Bounded-rate local RGB capture. This component has no model/provider access."""
from copy import deepcopy
import math
import threading
import time

class CameraSampler:
    def __init__(self, simulation, *, interval_seconds=1., clock=time.monotonic):
        if not math.isfinite(interval_seconds) or not .25 <= interval_seconds <= 10:
            raise ValueError('camera_interval_out_of_range')
        self.sim=simulation;self.interval=interval_seconds;self.clock=clock
        self.lock=threading.RLock();self.capture_lock=threading.Lock()
        self.stop_event=threading.Event();self.thread=None;self.generation=0
        self.pair=None;self.last_capture_at=None;self.last_attempt_at=None;self.error=None;self.capturing=False
    def start(self):
        if self.thread and self.thread.is_alive():return
        self.stop_event.clear()
        self.thread=threading.Thread(target=self._loop,daemon=True,name='local-rgb-sampler')
        self.thread.start()
    def _loop(self):
        while not self.stop_event.is_set():
            self.capture()
            self.stop_event.wait(self.interval) # Never catch up or overlap slow captures.
    def capture(self):
        if self.stop_event.is_set() or not self.capture_lock.acquire(blocking=False):return
        try:
            with self.lock:
                generation=self.generation
                now=self.clock()
                if self.last_attempt_at is not None and now-self.last_attempt_at<self.interval:return
                self.last_attempt_at=now;self.capturing=True
            capture=getattr(self.sim,'capture_preview_frames',self.sim.capture_frames)
            packet=capture()
            pair={key:packet[key] for key in ('frame_id','captured_at','images')}
            # UI camera geometry is local robot calibration, never scene/oracle data.
            # Provider requests separately retain the strict image-only contract.
            pair['images']=[{k:i[k] for k in ('camera_id','width','height','data_url') if k in i}
                            for i in packet['images']]
            if 'camera_profile' in packet:
                if packet['camera_profile'] not in ('legacy_fixed_stereo','observer_wrist'):
                    raise ValueError('invalid_camera_profile')
                pair['camera_profile']=packet['camera_profile']
            from embodied_arm.perception.camera_perception import CameraCalibration
            if 'camera_profile' in pair and 'calibrations' in packet:
                pair['calibrations']=[CameraCalibration.from_dict(c).to_dict() for c in packet['calibrations']]
            if 'camera_profile' in pair and 'view_metadata' in packet:
                pair['view_metadata']=[{k:m[k] for k in ('camera_id','frame_id','captured_at','role','extrinsics_source')}
                    | {'calibration':CameraCalibration.from_dict(m['calibration']).to_dict()}
                    for m in packet['view_metadata']]
            if pair.get('camera_profile')=='observer_wrist':
                from embodied_arm.perception.moving_camera_geometry import validate_capture_geometry
                validate_capture_geometry(pair)
            images=pair['images']
            if (len(images)!=2 or {x['camera_id'] for x in images}!={'left','right'}
                    or any(not x['data_url'].startswith('data:image/jpeg;base64,') for x in images)):
                raise ValueError('invalid_rgb_pair')
            with self.lock:
                if generation==self.generation and not self.stop_event.is_set():
                    self.pair=deepcopy(pair);self.last_capture_at=self.clock();self.error=None
        except Exception:
            with self.lock:
                if generation==self.generation:self.error='camera_capture_failed';self.pair=None
        finally:
            with self.lock:self.capturing=False
            self.capture_lock.release()
    def get(self):
        with self.lock:missing=self.pair is None
        if missing:self.capture()
        with self.lock:
            if self.pair is None:raise ValueError(self.error or 'camera_not_ready')
            return deepcopy(self.pair)
    def invalidate(self):
        with self.lock:self.generation+=1;self.pair=None;self.last_capture_at=None;self.last_attempt_at=None;self.error=None
    def status(self):
        with self.lock:
            return {'mode':'automatic_local','interval_seconds':self.interval,
                'last_frame_id':self.pair['frame_id'] if self.pair else None,
                'last_capture_at':self.last_capture_at,
                'age_seconds':max(0,self.clock()-self.last_capture_at) if self.last_capture_at is not None else None,
                'status':'stopped' if self.stop_event.is_set() else 'error' if self.error else 'capturing' if self.capturing else 'ready' if self.pair else 'starting',
                'error':self.error,'model_calls':0}
    def close(self):
        self.stop_event.set();self.invalidate()
        if self.thread:self.thread.join(2)
