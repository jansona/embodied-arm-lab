"""Offline recorded-AI-pixel + fake-Jev physics diagnostic; no real model calls."""
import json,threading,time
from pathlib import Path
from types import SimpleNamespace
from camera_perception import PixelDetection
from so101_simulation import SO101Simulation
from so101_control import CameraSceneSource
from so101_executor import VisualWaypointExecutor
from so101_jev import parse_response
from adapters import ALIYUN_MODEL

def run(recorded_detections,object_id='red',high_resolution=False):
    class VisionFixture:
        def perceive(self,packet):
            detections=[]
            sizes={i['camera_id']:(i['width'],i['height']) for i in packet['images']}
            for d in recorded_detections:
                detections.append(PixelDetection(d['object_id'],d['camera_id'],packet['frame_id'],
                    (d['uv_1000'][0]*(sizes[d['camera_id']][0]-1)/1000,d['uv_1000'][1]*(sizes[d['camera_id']][1]-1)/1000),d['confidence'],d['sigma_1000']*(sizes[d['camera_id']][0]-1)/1000,'visible',d['landmark']))
            from vision_adapter import VisionResult
            return VisionResult(accepted=True,reason='accepted',frame_id=packet['frame_id'],captured_at=packet['captured_at'],detections=tuple(detections),provenance='offline_fixture',calls=0,model='recorded_pixel_fixture')
    class JevFixture:
        def decide(self,obs):
            i=obs['axis_index'];error=obs['target']['target_xyz'][i]-obs['robot']['ee'][i];tol=obs['target']['tolerance_m']
            choice='positive' if error>tol else 'negative' if error < -tol else 'hold'
            return parse_response({'model':ALIYUN_MODEL,'answers':{'action':{'type':'choice','choice':choice,'confidence':1.,'probabilities':{k:float(k==choice) for k in ('positive','negative','hold')}}}},obs)
    sim=SO101Simulation(camera_size=(1920,1440) if high_resolution else (640,480));source=CameraSceneSource(sim,VisionFixture(),refine_rgb=True,refinement_profile='scaled' if high_resolution else 'baseline',roi_tracking=high_resolution)
    stop=threading.Event()
    def physics():
        while not stop.is_set():sim.tick(4);time.sleep(1/60)
    thread=threading.Thread(target=physics);thread.start()
    started=time.monotonic()
    try:
        result=VisualWaypointExecutor(sim,source,JevFixture()).run(object_id,[0,0,.035],max_calls=12,max_seconds=90)
        return {'mode':'offline_recorded_ai_pixels_and_fake_jev','real_api_calls':0,'result':result,'elapsed_seconds':time.monotonic()-started,
                'proprioception':sim.proprioception(),'perception_status':source.status()}
    finally:stop.set();thread.join();sim.close()
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('detections');p.add_argument('output');p.add_argument('--high-resolution',action='store_true');args=p.parse_args()
    result=run(json.loads(Path(args.detections).read_text())['detections'],high_resolution=args.high_resolution);Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('mode','real_api_calls','result','elapsed_seconds')}))
