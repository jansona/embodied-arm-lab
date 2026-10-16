"""Moving-view association from prior RGB geometry, never object-state inputs.

World anchors only predict search regions. Fresh image fits and independent
triangulation remain mandatory; a prediction is never returned as a detection.
"""
import base64
from dataclasses import replace
import numpy as np
from embodied_arm.perception.camera_perception import CameraCalibration, RGBFrame


def frames_from_packet(packet):
    calibrations = {c['camera_id']: CameraCalibration.from_dict(c) for c in packet['calibrations']}
    return tuple(RGBFrame.from_jpeg(packet['frame_id'], packet['captured_at'],
        calibrations[i['camera_id']], base64.b64decode(i['data_url'].split(',', 1)[1], validate=True))
        for i in packet['images'])


def compatible_calibrations(initial, current, *, moving_wrist=False):
    """Only the explicitly enabled wrist extrinsics may move; optics stay fixed."""
    if set(initial) != set(current):return False
    for cid, old in initial.items():
        new = current[cid]
        a, b = old.to_dict(), new.to_dict()
        if moving_wrist and cid == 'right':
            a.pop('world_from_camera');b.pop('world_from_camera')
        if a != b:return False
    return True


def projected_face_area(calibration, xyz, edge=.02):
    """Known horizontal cuboid-face footprint; used only for relative scale gates."""
    x,y,z = map(float,xyz)
    pixels=np.asarray([calibration.project((x+dx*edge/2,y+dy*edge/2,z))
        for dx,dy in ((-1,-1),(1,-1),(1,1),(-1,1))])
    area=abs(float(np.dot(pixels[:,0],np.roll(pixels[:,1],-1))-
                   np.dot(pixels[:,1],np.roll(pixels[:,0],-1))))/2
    if area <= 1e-6:raise ValueError('Projected face is degenerate')
    return area


def reproject_detections(detections, world_points, calibrations):
    result=[]
    for d in detections:
        if d.object_id not in world_points:
            result.append(d);continue
        c=calibrations[d.camera_id];uv=c.project(tuple(float(v) for v in world_points[d.object_id]))
        if not 0 <= uv[0] < c.width or not 0 <= uv[1] < c.height:
            raise ValueError('predicted_landmark_outside_image')
        result.append(replace(d,uv=uv))
    return tuple(result)


def validate_capture_geometry(packet):
    """Bind local calibration to this RGB capture, never to a prior frame cache."""
    if packet.get('camera_profile')!='observer_wrist':raise ValueError('Moving camera profile required')
    metadata=packet.get('view_metadata',[])
    if len(metadata)!=2 or {m.get('camera_id') for m in metadata}!={'left','right'}:
        raise ValueError('Missing per-view calibration binding')
    calibrations={c['camera_id']:c for c in packet['calibrations']}
    images={i['camera_id']:i for i in packet['images']}
    if set(calibrations)!=set(images) or set(images)!={'left','right'}:raise ValueError('Camera layout changed')
    for m in metadata:
        cid=m['camera_id'];c=calibrations[cid]
        if (m.get('frame_id')!=packet['frame_id'] or m.get('captured_at')!=packet['captured_at']
                or m.get('calibration')!=c or (c['width'],c['height'])!=(images[cid]['width'],images[cid]['height'])
                or m.get('role')!=('observer' if cid=='left' else 'wrist')
                or m.get('extrinsics_source')!=('fixed_calibration' if cid=='left' else 'joint_encoder_fk_rigid_mount')):
            raise ValueError('Stale or mismatched per-frame calibration')
        CameraCalibration.from_dict(c)
