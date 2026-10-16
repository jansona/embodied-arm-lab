"""Offline acquisition experiment; never modifies production renderer/refiner.

Reads actual RGB through CameraSceneSource. Fixtures are explicitly offline.
Run directly to emit source-backed perception metrics, not grasp-success claims.
"""
import json
import sys
import time
import numpy as np
import pybullet as p
from tests.rgb_coarse_fixture import RGBColourFixture
from embodied_arm.simulation.so101_simulation import SO101Simulation
from embodied_arm.control.so101_control import CameraSceneSource

PROFILES = {
    'shadow_off': {'shadow': 0},
    'overhead_diffuse': {'shadow': 0, 'lightDirection': [0, 0, 1],
                         'lightAmbientCoeff': .4, 'lightDiffuseCoeff': .6,
                         'lightSpecularCoeff': 0.},
    'oblique_diffuse': {'shadow': 0, 'lightDirection': [1, 1, 3],
                        'lightAmbientCoeff': .4, 'lightDiffuseCoeff': .6,
                        'lightSpecularCoeff': 0.},
}


def run(profile, seed=0):
    sim = SO101Simulation(seed=seed, camera_size=(1920, 1440))
    original_call = sim.call
    def call(fn, *args, **kwargs):
        if fn is p.getCameraImage:
            kwargs.update(PROFILES[profile])
        return original_call(fn, *args, **kwargs)
    sim.call = call
    source = CameraSceneSource(sim, RGBColourFixture(), refine_rgb=True, refinement_profile='scaled')
    result = {'profile': profile, 'seed': seed, 'renderer_options': PROFILES[profile]}
    started = time.monotonic()
    try:
        snapshot = source.refresh()
        result['accepted'] = True
        result['objects'] = {k: {'xyz': v['xyz'], 'position_std_m': v['position_std_m'],
            'axis_std_m': np.sqrt(np.diag(v['covariance_m2'])).tolist()}
            for k, v in snapshot['scene']['objects'].items()}
        result['source'] = snapshot['detector_provenance']
        # A second full capture verifies fresh provenance; bounded ROI experiment
        # is reported separately rather than pretending full capture is cheap.
        packet = sim.capture_frames()
        result['next_capture_age_s'] = time.monotonic() - packet['captured_at']
        result['fresh_frame_id'] = packet['frame_id'] not in snapshot['scene']['frame_ids'].values()
    except Exception as exc:
        result['accepted'] = False
        result['error'] = str(exc)
        status = source.status()
        result['reason'] = status['reason']
        result['refinement'] = (status.get('last_vision') or {}).get('refinement')
    finally:
        result['elapsed_s'] = time.monotonic() - started
        sim.close()
    return result


if __name__ == '__main__':
    profiles = sys.argv[1:] or list(PROFILES)
    for profile in profiles:
        print(json.dumps(run(profile)), flush=True)


def validate_roi_and_occlusion(seed=0):
    """Fresh cropped capture and an image-space partial-occlusion negative."""
    import base64
    from dataclasses import replace
    from io import BytesIO
    from PIL import Image, ImageDraw
    from embodied_arm.perception.rgb_roi_geometry import bounds_from_detections
    from embodied_arm.perception.rgb_scaled_refinement import refine_packet
    sim = SO101Simulation(seed=seed, camera_size=(1920, 1440))
    original_call = sim.call
    def call(fn, *args, **kwargs):
        if fn is p.getCameraImage:
            kwargs.update(PROFILES['shadow_off'])
        return original_call(fn, *args, **kwargs)
    sim.call = call
    source = CameraSceneSource(sim, RGBColourFixture(), refine_rgb=True, refinement_profile='scaled')
    try:
        source.refresh()
        snap, frames, detections = source.observation_evidence()
        images = [{'camera_id':f.camera_id,'width':f.calibration.width,'height':f.calibration.height,'data_url':f.data_url} for f in frames]
        source.tracking_rois = bounds_from_detections(images, detections, ('red', 'blue', 'goal'))
        started = time.monotonic()
        fresh = source.track()
        elapsed = time.monotonic() - started
        result = {'seed':seed, 'roi_tracking_accepted':True,'roi_tracking_elapsed_s':elapsed,
                  'roi_age_at_use_s':time.monotonic()-fresh['scene']['captured_at'],
                  'roi_bounds':source.tracking_rois}
        packet = sim.capture_frames(roi_bounds=source.tracking_rois)
        packet = {k:packet[k] for k in ('frame_id','captured_at','images')}
        current = tuple(replace(d,frame_id=packet['frame_id']) for d in detections)
        red = next(d for d in current if d.object_id=='red' and d.camera_id=='left')
        item = next(i for i in packet['images'] if i['camera_id']=='left')
        image = Image.open(BytesIO(base64.b64decode(item['data_url'].split(',')[1]))).convert('RGB')
        draw = ImageDraw.Draw(image)
        x,y = red.uv
        draw.rectangle((int(x)-5,int(y)-40,int(x)+5,int(y)+40),fill=(180,180,180))
        out = BytesIO();image.save(out,format='JPEG',quality=92)
        item['data_url']='data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()
        occluded = refine_packet(packet,current)
        result['occlusion_accepted']=occluded.accepted
        result['occlusion_rejections']=[{'object_id':e.object_id,'reason':e.reason} for e in occluded.evidence if not e.accepted]
        return result
    finally:
        sim.close()
