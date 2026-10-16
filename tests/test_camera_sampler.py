import threading
import time
import pytest
from embodied_arm.services.camera_sampler import CameraSampler

class Camera:
    def __init__(self):self.calls=0;self.active=0;self.max_active=0;self.fail=False
    def capture_frames(self):
        self.calls+=1;self.active+=1;self.max_active=max(self.active,self.max_active)
        try:
            if self.fail:raise RuntimeError('private provider detail')
            return {'frame_id':str(self.calls),'captured_at':time.monotonic(),
                'images':[{'camera_id':x,'data_url':'data:image/jpeg;base64,/9j/2Q=='} for x in ('left','right')],
                'depth':'must_not_escape','segmentation':'must_not_escape'}
        finally:self.active-=1

def test_local_sampling_runs_without_http_clients_or_model_provider():
    sim=Camera();s=CameraSampler(sim,interval_seconds=.25)
    s.start()
    try:
        deadline=time.monotonic()+1
        while sim.calls<2 and time.monotonic()<deadline:time.sleep(.01)
        assert sim.calls>=2 and s.status()['model_calls']==0
        assert set(s.get())=={'frame_id','captured_at','images'}
        assert sim.max_active==1
    finally:s.close()
    calls=sim.calls;time.sleep(.3)
    assert sim.calls==calls and s.status()['status']=='stopped'

def test_invalidation_and_capture_failure_are_explicit_and_recoverable():
    sim=Camera();s=CameraSampler(sim);first=s.get()
    s.invalidate();sim.fail=True;s.capture()
    assert s.status()['error']=='camera_capture_failed'
    with pytest.raises(ValueError,match='camera_capture_failed'):s.get()
    sim.fail=False;s.invalidate();s.capture()
    assert s.get()['frame_id']!=first['frame_id'] and s.status()['error'] is None
    s.close()

def test_concurrent_reads_do_not_duplicate_capture_or_publish_after_invalidation():
    entered=threading.Event();release=threading.Event()
    class Slow(Camera):
        def capture_frames(self):entered.set();release.wait(1);return super().capture_frames()
    sim=Slow();s=CameraSampler(sim)
    t=threading.Thread(target=s.capture);t.start();assert entered.wait(1)
    s.capture();s.invalidate();release.set();t.join(1)
    assert sim.calls==1 and s.pair is None
    s.close()

@pytest.mark.parametrize('interval',[0,.1,11,float('nan'),float('inf')])
def test_sampling_rate_is_bounded(interval):
    with pytest.raises(ValueError):CameraSampler(Camera(),interval_seconds=interval)

def test_failed_capture_cannot_be_retried_faster_than_configured_interval():
 sim=Camera();sim.fail=True;now=[100.]
 s=CameraSampler(sim,clock=lambda:now[0])
 for _ in range(4):
  with pytest.raises(ValueError):s.get()
 assert sim.calls==1
 now[0]+=1
 with pytest.raises(ValueError):s.get()
 assert sim.calls==2
 s.close()


def test_camera_metadata_preserved_but_oracle_fields_dropped():
    from embodied_arm.simulation.so101_simulation import SO101Simulation
    from embodied_arm.perception.moving_camera_geometry import validate_capture_geometry
    sim=SO101Simulation(camera_profile='observer_wrist')
    try:
        original=sim.capture_preview_frames
        def capture():
            packet=original();packet['objects']={'red':[1,2,3]};packet['depth']='forbidden'
            packet['view_metadata'][0]['object_pose']='forbidden'
            packet['images'][0]['segmentation']='forbidden'
            return packet
        sim.capture_preview_frames=capture
        sampler=CameraSampler(sim)
        packet=sampler.get();validate_capture_geometry(packet)
        assert set(packet)=={'frame_id','captured_at','images','camera_profile','view_metadata','calibrations'}
        assert 'objects' not in packet and 'depth' not in packet
        assert 'object_pose' not in packet['view_metadata'][0]
        assert 'segmentation' not in packet['images'][0]
        sampler.close()
    finally:sim.close()

def test_undeclared_legacy_camera_extras_are_not_parsed_or_exposed():
    class Legacy(Camera):
        def capture_frames(self):
            return dict(super().capture_frames(),calibrations='untrusted legacy extras',view_metadata='ignored')
    sampler=CameraSampler(Legacy())
    try:assert set(sampler.get())=={'frame_id','captured_at','images'}
    finally:sampler.close()
