"""Offline workspace-derived front stereo profile, actual RGB measurements only.

Only camera geometry changes; this file does not edit production defaults.
Targets derive from initial calibrated RGB, declared cube geometry and robot FK.
"""
import base64
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
import embodied_arm.simulation.so101_simulation as simulation
from embodied_arm.perception.camera_perception import CameraCalibration,RGBFrame,estimate_scene,PerceptionConfig,PixelDetection,LANDMARKS
from embodied_arm.control.so101_control import CameraSceneSource,OBSERVATION_VOLUME
from embodied_arm.perception.rgb_roi_geometry import bounds_from_detections
from embodied_arm.perception.rgb_scaled_refinement import refine_packet
from tests.rgb_coarse_fixture import RGBColourFixture
from tests.test_so101_grasp_feasibility import _fixture_move,_settle
from embodied_arm.perception.visual_feedback import VisualObservation,DynamicRGBPhaseTracker,VisualPhaseVerifier

FRONT_CAMERAS={'left':((.44,-.30,.39),(.20,0.,.035)),
               'right':((.44,.30,.39),(.20,0.,.035))}


def run(left_eye=(.44,-.30,.39), profile='front_stereo_v1', required=('red','blue','goal')):
    cameras=dict(FRONT_CAMERAS);cameras['left']=(tuple(left_eye),(.20,0.,.035))
    original=dict(simulation.CAMERAS)
    simulation.CAMERAS.clear();simulation.CAMERAS.update(cameras)
    sim=simulation.SO101Simulation(0,camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    output={'profile':profile,'cameras':cameras,'required_objects':required,'stages':[]}
    try:
        source=CameraSceneSource(sim,RGBColourFixture(),refine_rgb=True,refinement_profile='scaled')
        try:initial=source.refresh(required_objects=required)
        except Exception as exc:
            output.update(accepted=False,reason=str(exc),source=source.status());return output
        _,original_frames,original_detections=source.observation_evidence()
        if 'goal' not in required:
            original_detections=tuple(original_detections)+tuple(PixelDetection('goal',f.camera_id,f.frame_id,None,0.,None,'unknown',LANDMARKS['goal']) for f in original_frames)
        images=[{'camera_id':f.camera_id,'width':1920,'height':1440} for f in original_frames]
        bounds=bounds_from_detections(images,original_detections,required)
        original_uv={(d.object_id,d.camera_id):d.uv for d in original_detections}
        semantic_at=initial['scene']['captured_at']
        def observe(stage,seeds=original_detections):
            current_bounds=bounds
            if stage=='lifted':
                predicted=sim.proprioception()['ee']
                hint_cal={cam:CameraCalibration.from_dict(simulation.camera_calibration(cam,1920,1440)) for cam in ('left','right')}
                seeds=tuple(replace(d,uv=hint_cal[d.camera_id].project(predicted)) if d.object_id=='red' else d for d in seeds)
                current_bounds=bounds_from_detections(images,seeds,required)
            packet=sim.capture_frames(roi_bounds=current_bounds)
            packet_rgb={k:packet[k] for k in ('frame_id','captured_at','images')}
            calibrated={c['camera_id']:CameraCalibration.from_dict(c) for c in packet['calibrations']}
            frames=tuple(RGBFrame.from_jpeg(packet['frame_id'],packet['captured_at'],calibrated[i['camera_id']],
                base64.b64decode(i['data_url'].split(',')[1])) for i in packet['images'])
            result=refine_packet(packet_rgb,tuple(replace(d,frame_id=packet['frame_id']) for d in seeds))
            row={'stage':stage,'capture_age_s':time.monotonic()-packet['captured_at'],'views':[],'tcp':sim.proprioception()['ee']}
            for f in frames:
                Path(f'/tmp/front-stereo-red-{stage}-{f.camera_id}.jpg').write_bytes(base64.b64decode(f.jpeg_base64))
            for e,d in zip(result.evidence,[d for d in result.detections if d.object_id in required]):
                row['views'].append({'object':e.object_id,'camera':d.camera_id,'accepted':e.accepted,'reason':e.reason,
                    'uv':e.uv,'sigma_px':e.pixel_sigma_px,'drift_px':None if e.uv is None else math.dist(e.uv,original_uv[(e.object_id,d.camera_id)])})
            scene=estimate_scene(frames,result.detections,now=time.monotonic(),workspace=OBSERVATION_VOLUME,
                config=PerceptionConfig(max_frame_age_s=3,max_position_std_m=.003,max_ray_separation_m=.006))
            row['objects']={k:{'status':v.status,'reason':v.reason,'std_m':v.position_std_m,'xyz':v.xyz} for k,v in scene.objects.items()}
            output['stages'].append(row)
            return scene,frames,result.detections
        observe('initial')
        obj=initial['scene']['objects']['red']['xyz']
        target=[obj[0],obj[1],obj[2]-.007]
        pose=sim.robot.forward_kinematics(sim.robot.solve_ik(target))
        shift=math.sqrt(2)*.01+.006+.002-.013
        target[:2]=[v+shift*float(a) for v,a in zip(target[:2],pose[:2,0])]
        def move(xyz):
            sim.resume();_fixture_move(sim,xyz);sim.pause_idle()
        move([target[0],target[1],.06]);observe('approach')
        move([target[0],target[1],.052]);observe('first_descent')
        move(target);scene,frames,detections=observe('preclose')
        try:
            scene.require('red',now=time.monotonic())
            obs=VisualObservation(scene,detections,tuple(sim.proprioception()['ee']),scene.captured_at,
                sim.epoch,'front-red-close-lift','front-initial-rgb-red',1,.9,'dynamic_rgb_track',semantic_at)
            tracker=DynamicRGBPhaseTracker(profile='scaled',semantic_lifetime_s=120.)
            tracker.start(obs,frames,object_id='red',now=time.monotonic())
        except Exception as exc:
            output['tracker_start_error']=str(exc);tracker=None
        sim.resume()
        for i in range(360):
            t=(i+1)/360;sim.robot.command_gripper(1-t*t*(3-2*t));sim.tick(1)
        _settle(sim,360);sim.pause_idle()
        scene,frames,detections=observe('closed')
        if tracker:
            closure=tracker.reacquire_closure(frames,tcp_xyz=sim.proprioception()['ee'],tcp_captured_at=scene.captured_at,
                now=time.monotonic(),workspace=OBSERVATION_VOLUME)
            output['closure']={'accepted':closure.accepted,'reason':closure.reason}
        move([target[0],target[1],.06]);sim.resume();_settle(sim,480);sim.pause_idle()
        observe('lifted')
        output['accepted']=all(all(stage['objects'][name]['status']=='estimated' for name in required) for stage in output['stages'])
        return output
    finally:
        sim.close();simulation.CAMERAS.clear();simulation.CAMERAS.update(original)

if __name__=='__main__':print(json.dumps(run()),flush=True)
