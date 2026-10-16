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
            packet=self.sim.capture_frames()
            pair={key:packet[key] for key in ('frame_id','captured_at','images')}
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
