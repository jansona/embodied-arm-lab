"""Bounded offline recovery diagnostic; no production changes or oracle inputs.

A single-view landmark does NOT become a new 3D point. This experiment computes
its entire possible camera-ray cone inside the existing observation workspace,
then conservatively expands its AABB by the declared20mm cube geometry. It only
reports TCP-segment clearance, not clearance of the complete articulated arm.
"""
from dataclasses import replace
from itertools import combinations
import base64
import json
import math
import time
from pathlib import Path
import numpy as np
from embodied_arm.simulation.so101_simulation import SO101Simulation
from embodied_arm.control.so101_control import CameraSceneSource,OBSERVATION_VOLUME
from embodied_arm.perception.camera_perception import CameraCalibration,PixelDetection,LANDMARKS,RGBFrame,estimate_scene,PerceptionConfig
from embodied_arm.perception.rgb_scaled_refinement import refine_rgb_landmark
from embodied_arm.perception.rgb_roi_geometry import bounds_from_detections
from tests.rgb_coarse_fixture import RGBColourFixture
from tests.test_so101_grasp_feasibility import _fixture_move,_settle


def conservative_obstacle_aabb(calibration, detection, workspace=OBSERVATION_VOLUME):
    """Exact convex cone/workspace vertices, then an enclosing box (overbound).

    No range estimate or old position is used. The cube's top-centre landmark
    bounds every cube corner within sqrt(6)*10mm even with unknown rotation.
    Clip to workspace expanded by this radius, then expand the enclosing box;
    objects just outside workspace that could touch it are therefore included.
    A3sigma pixel window has at least
    the stereo estimator's unchanged2px floor. Camera error is unmodelled, as in
    the existing stereo estimator; calibration must independently be validated.
    """
    if detection.visibility!='visible' or detection.confidence<.8:
        raise ValueError('No usable fresh monocular landmark')
    c=calibration;origin=np.asarray(c.world_from_camera)[:3,3]
    rotation=np.asarray(c.world_from_camera)[:3,:3]
    radius=3*max(2.,detection.pixel_sigma_px)
    u,v=detection.uv
    xlo,xhi=(u-radius-c.cx)/c.fx,(u+radius-c.cx)/c.fx
    ylo,yhi=(v-radius-c.cy)/c.fy,(v+radius-c.cy)/c.fy
    # n . optical <= 0; world optical = R.T @ (world-origin).
    normals=[[-1,0,xlo],[1,0,-xhi],[0,-1,ylo],[0,1,-yhi],[0,0,-1]]
    A=[];b=[]
    for n in normals:
        world=rotation@np.asarray(n,dtype=float);A.append(world);b.append(float(world@origin))
    body_radius=math.sqrt(6)*.01
    for axis,(lo,hi) in enumerate(workspace):
        lo-=body_radius;hi+=body_radius
        n=np.zeros(3);n[axis]=1;A.append(n);b.append(hi)
        A.append(-n);b.append(-lo)
    A=np.asarray(A);b=np.asarray(b);vertices=[]
    for rows in combinations(range(len(A)),3):
        mat=A[list(rows)]
        if abs(np.linalg.det(mat))<1e-10:continue
        point=np.linalg.solve(mat,b[list(rows)])
        if np.all(A@point<=b+1e-9):vertices.append(point)
    if not vertices:raise ValueError('Observed ray does not intersect workspace')
    vertices=np.asarray(vertices)
    lo=vertices.min(axis=0)-body_radius
    hi=vertices.max(axis=0)+body_radius
    return lo,hi,{'pixel_radius':radius,'vertices':len(vertices),'source':'fresh_monocular_full_ray_volume',
                  'fresh_3d_point':False,'old_point_used':False,'orientation_free_body_radius_m':body_radius}


def segment_aabb_distance(start,end,lo,hi):
    start,end,lo,hi=map(lambda x:np.asarray(x,dtype=float),(start,end,lo,hi))
    delta=end-start;knots=[0.,1.]
    for axis in range(3):
        if abs(delta[axis])>1e-15:
            knots += [float((edge-start[axis])/delta[axis]) for edge in (lo[axis],hi[axis]) if 0<(edge-start[axis])/delta[axis]<1]
    knots=sorted(set(knots));candidates=knots[:]
    for a,b in zip(knots,knots[1:]):
        mid=start+(a+b)/2*delta
        active=(mid<lo)|(mid>hi)
        if not active.any():return 0.
        boundary=np.where(mid<lo,lo,hi)
        denominator=float(delta[active]@delta[active])
        if denominator:
            t=float(delta[active]@(boundary-start)[active])/denominator
            candidates.append(min(b,max(a,t)))
    return min(float(np.linalg.norm(np.maximum(np.maximum(lo-(start+t*delta),(start+t*delta)-hi),0))) for t in candidates)


def run(inspect_chain=False, inspect_unverified_closure=False):
    sim=SO101Simulation(0,camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    sim.evaluation_state=lambda:(_ for _ in ()).throw(AssertionError('Oracle forbidden'))
    report={'schema':'occluded-obstacle-recovery-diagnostic-v1','provenance':'actual_RGB_fixture_trajectory_real_physics',
            'production_changes':False,'camera_changed':False,'stationary_retries':[]}
    try:
        source=CameraSceneSource(sim,RGBColourFixture(),refine_rgb=True,refinement_profile='scaled')
        initial=source.refresh();_,frames,detections=source.observation_evidence()
        selected=tuple(d for d in detections if d.object_id in ('red','blue'))
        anchors={(d.object_id,d.camera_id):d.uv for d in selected}
        images=[{'camera_id':f.camera_id,'width':f.calibration.width,'height':f.calibration.height} for f in frames]
        bounds=bounds_from_detections(images,selected,('red','blue'))
        point=initial['scene']['objects']['red']['xyz']
        geometry=sim.grasp_geometry([point[0],point[1],point[2]-.007])
        offset=geometry['approach_offset_world']
        target=[point[0]+offset[0],point[1]+offset[1],point[2]-geometry['tcp_below_top_m']]
        hover=[target[0],target[1],.06]
        sim.resume();_fixture_move(sim,hover);sim.pause_idle()
        safe_tcp=sim.proprioception()['ee']
        sim.resume();_fixture_move(sim,[target[0],target[1],.052]);sim.pause_idle()
        start_tcp=sim.proprioception()['ee']
        report.update(prior_hover_tcp=safe_tcp,blocked_tcp=start_tcp,intended_grasp_tcp=target)
        def capture(stage, seeds=selected):
            current_bounds=bounds_from_detections(images,seeds,('red','blue'))
            packet=sim.capture_frames(roi_bounds=current_bounds)
            calibrations={c['camera_id']:CameraCalibration.from_dict(c) for c in packet['calibrations']}
            current={};views=[]
            for seed in seeds:
                item=next(i for i in packet['images'] if i['camera_id']==seed.camera_id)
                raw=base64.b64decode(item['data_url'].split(',')[1])
                result=refine_rgb_landmark(raw,object_id=seed.object_id,
                    uv_1000=[seed.uv[0]/1919*1000,seed.uv[1]/1439*1000],confidence=seed.confidence,width=1920,height=1440)
                drift=math.dist(result.uv,anchors[(seed.object_id,seed.camera_id)]) if result.accepted else None
                views.append({'object':seed.object_id,'camera':seed.camera_id,'accepted':result.accepted,'reason':result.reason,'drift_px':drift})
                if result.accepted:
                    current[(seed.object_id,seed.camera_id)]=PixelDetection(seed.object_id,seed.camera_id,packet['frame_id'],
                        result.uv,seed.confidence,result.pixel_sigma_px,landmark=LANDMARKS[seed.object_id])
            row={'stage':stage,'frame_id':packet['frame_id'],'age_s':time.monotonic()-packet['captured_at'],
                 'views':views,'stereo_anchor_reacquired':all(v['accepted'] and v['drift_px']<=4 for v in views)}
            if ('blue','right') in current:
                lo,hi,meta=conservative_obstacle_aabb(calibrations['right'],current[('blue','right')])
                descend=segment_aabb_distance(start_tcp,target,lo,hi)
                retreat=segment_aabb_distance(start_tcp,safe_tcp,lo,hi)
                row['ray_volume']={**meta,'aabb_lo':lo.tolist(),'aabb_hi':hi.tolist(),
                    'descent_tcp_clearance_m':descend,'retreat_tcp_clearance_m':retreat,
                    'required_existing_tcp_clearance_m':.04,
                    'descent_tcp_clears_entire_volume':descend>=.04,'retreat_tcp_clears_entire_volume':retreat>=.04}
            rgb_frames=tuple(RGBFrame.from_jpeg(packet['frame_id'],packet['captured_at'],calibrations[i['camera_id']],
                base64.b64decode(i['data_url'].split(',')[1])) for i in packet['images'])
            scene=estimate_scene(rgb_frames,tuple(current.values()),now=time.monotonic(),workspace=OBSERVATION_VOLUME,
                config=PerceptionConfig(max_frame_age_s=3,max_position_std_m=.003,max_ray_separation_m=.006))
            row['tcp']=sim.proprioception()['ee']
            row['blue_stereo_restored']=all(v['accepted'] and v['drift_px']<=4 for v in views if v['object']=='blue') and scene.objects['blue'].valid
            row['objects']={k:{'status':v.status,'reason':v.reason,'xyz':v.xyz,'std_m':v.position_std_m} for k,v in scene.objects.items()}
            return row,scene,tuple(current.values()),calibrations
        for index in range(3):
            row,_,_,_=capture('stationary_retry_'+str(index+1))
            report['stationary_retries'].append(row)
        report['result']='diagnostic_only_no_recovery_motion_executed'
        if inspect_chain:
            report['chain_inspection']=[]
            # Explicit offline pose replay only; this is not a production
            # recovery implementation and does not bypass its fail-closed guard.
            for stage,z in [('mid_descent',.036),('low_descent',.020),('preclose',target[2])]:
                sim.resume();_fixture_move(sim,[target[0],target[1],z]);sim.pause_idle()
                row,scene,current,calibrations=capture(stage)
                report['chain_inspection'].append(row)
                print(json.dumps({'stage':stage,'blue_stereo_restored':row['blue_stereo_restored'],
                                 'red_status':scene.objects['red'].status}),flush=True,file=__import__('sys').stderr)
            if not row['blue_stereo_restored'] or not scene.objects['red'].valid:
                report['result']='preclose_not_fully_visible_no_close_executed'
                if not inspect_unverified_closure:
                    return report
                report['offline_pose_survey_only']=True
                report['preclose_stereo_contract_satisfied']=False
                # Explicit diagnostic pose replay, not authorization for the
                # production controller or evidence that recovery is safe.
            sim.resume()
            for i in range(360):
                t=(i+1)/360;sim.robot.command_gripper(1-t*t*(3-2*t));sim.tick(1)
            _settle(sim,360);sim.pause_idle()
            row,scene,current,calibrations=capture('closed')
            report['chain_inspection'].append(row)
            if not scene.objects['red'].valid:
                report['result']='closed_target_not_visible_no_lift_executed'
                return report
            closed_xyz=scene.objects['red'].xyz;closed_tcp=sim.proprioception()['ee']
            sim.resume();_fixture_move(sim,[closed_tcp[0],closed_tcp[1],.06]);_settle(sim,480);sim.pause_idle()
            tcp=sim.proprioception()['ee']
            predicted=[float(a+b-c) for a,b,c in zip(closed_xyz,tcp,closed_tcp)]
            seeds=tuple(replace(d,uv=calibrations[d.camera_id].project(predicted)) if d.object_id=='red' else d for d in selected)
            row,scene,_,_=capture('lifted',seeds)
            report['chain_inspection'].append(row)
            report['result']='offline_pose_chain_inspected_not_production_execution'

        return report
    finally:sim.close()

if __name__=='__main__':print(json.dumps(run(inspect_chain='--chain' in __import__('sys').argv,inspect_unverified_closure='--pose-survey-only' in __import__('sys').argv),indent=2),flush=True)
