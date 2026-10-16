"""Offline RGB-grounded closure/carry experiment; no object-state feedback."""
import base64
from dataclasses import replace, asdict
import json
import math
import time
from embodied_arm.perception.camera_perception import CameraCalibration,RGBFrame,estimate_scene,PerceptionConfig
from embodied_arm.simulation.so101_simulation import SO101Simulation
from embodied_arm.control.so101_control import CameraSceneSource,OBSERVATION_VOLUME
from embodied_arm.perception.rgb_scaled_refinement import refine_packet
from tests.rgb_coarse_fixture import RGBColourFixture
from tests.test_so101_grasp_feasibility import _fixture_move,_settle
from embodied_arm.perception.visual_feedback import VisualObservation,DynamicRGBPhaseTracker,VisualPhaseVerifier


def run():
    sim=SO101Simulation(0,camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    try:
        source=CameraSceneSource(sim,RGBColourFixture(),refine_rgb=True,refinement_profile='scaled')
        initial=source.refresh();_,original_frames,original_detections=source.observation_evidence()
        semantic_at=initial['scene']['captured_at']
        images=[{'camera_id':f.camera_id,'width':1920,'height':1440} for f in original_frames]
        def capture():
            packet=sim.capture_frames()
            calibrations={c['camera_id']:CameraCalibration.from_dict(c) for c in packet['calibrations']}
            frames=tuple(RGBFrame.from_jpeg(packet['frame_id'],packet['captured_at'],calibrations[i['camera_id']],
                base64.b64decode(i['data_url'].split(',')[1])) for i in packet['images'])
            return packet,frames
        # All target geometry comes from initial RGB triangulation and the
        # declared20mm cube, with known robot FK for jaw asymmetry compensation.
        obj=initial['scene']['objects']['red']['xyz']
        target=[obj[0],obj[1],obj[2]-.008]
        pose=sim.robot.forward_kinematics(sim.robot.solve_ik(target))
        shift=math.sqrt(2)*.01+.006+.002-.012
        target[:2]=[v+shift*float(a) for v,a in zip(target[:2],pose[:2,0])]
        sim.resume();_fixture_move(sim,[target[0],target[1],.06]);_fixture_move(sim,target);sim.pause_idle()
        packet,frames=capture()
        seeds=tuple(replace(d,frame_id=packet['frame_id']) for d in original_detections)
        refined=refine_packet({k:packet[k] for k in ('frame_id','captured_at','images')},seeds)
        detections=tuple(d for d in refined.detections if d.object_id=='red')
        scene=estimate_scene(frames,detections,now=time.monotonic(),workspace=OBSERVATION_VOLUME,
            config=PerceptionConfig(max_position_std_m=.005,max_ray_separation_m=.006))
        scene.require('red',now=time.monotonic())
        obs=VisualObservation(scene,detections,tuple(sim.proprioception()['ee']),scene.captured_at,
            sim.epoch,'red-close-lift','initial-rgb-red',1,.9,'dynamic_rgb_track',semantic_at)
        tracker=DynamicRGBPhaseTracker(profile='scaled');tracker.start(obs,frames,object_id='red',now=time.monotonic())
        sim.resume()
        for i in range(360):
            t=(i+1)/360;sim.robot.command_gripper(1-t*t*(3-2*t));sim.tick(1)
        _settle(sim,360);sim.pause_idle()
        packet,frames=capture()
        result=tracker.reacquire_closure(frames,tcp_xyz=sim.proprioception()['ee'],tcp_captured_at=packet['captured_at'],
            now=time.monotonic(),workspace=OBSERVATION_VOLUME)
        output={'closure':{'accepted':result.accepted,'reason':result.reason}}
        if not result.accepted:return output
        before=tracker._base
        verifier=VisualPhaseVerifier();delta=(0.,0.,.06-before.tcp_xyz[2])
        verifier.begin_lift('red',before,commanded_tcp_delta=delta,now=time.monotonic())
        sim.resume();_fixture_move(sim,[before.tcp_xyz[0],before.tcp_xyz[1],.06]);_settle(sim,480);sim.pause_idle()
        packet,frames=capture();tcp=sim.proprioception()['ee']
        tracked=tracker.track(frames,tcp_xyz=tcp,tcp_captured_at=packet['captured_at'],now=time.monotonic())
        output['lift_tracking']={'accepted':tracked.accepted,'reason':tracked.reason}
        if tracked.accepted:
            scene=estimate_scene(frames,tracked.detections,now=time.monotonic(),workspace=OBSERVATION_VOLUME,
                config=PerceptionConfig(max_position_std_m=.005,max_ray_separation_m=.006))
            after=replace(before,scene=scene,detections=tracked.detections,tcp_xyz=tuple(tcp),tcp_captured_at=scene.captured_at)
            output['lift_verdict']=asdict(verifier.update(after,now=time.monotonic()))
        return output
    finally:sim.close()

if __name__=='__main__':print(json.dumps(run()),flush=True)
