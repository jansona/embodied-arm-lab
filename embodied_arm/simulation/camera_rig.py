"""Calibrated observer/wrist RGB rig; mount poses use robot encoders only."""
import math
import numpy as np

PROFILES = ('legacy_fixed_stereo', 'observer_wrist')
FIXED_VIEWS = {'left': ((.08, -.46, .40), (.20, 0., .035)),
               'right': ((.44, .30, .39), (.20, 0., .035))}


def look_at(eye, target, up):
    eye, target, up = map(lambda x: np.asarray(x, dtype=float), (eye, target, up))
    forward = target-eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4); pose[:3, :3] = np.column_stack((right, down, forward)); pose[:3, 3] = eye
    return pose


# A rigid virtual bracket beside the wrist, 55 × 35 mm lateral and 70 mm back
# from the jaw TCP. The camera looks toward the jaw, never toward an object.
TCP_FROM_WRIST_CAMERA = look_at((-.055, -.035, -.070), (0., 0., .035), (0., 0., -1.))
WRIST_FOV_DEG = 95.


def calibration(camera_id, width=640, height=480, *, world_from_camera=None, fov=50.):
    pose = look_at(*FIXED_VIEWS[camera_id], (0., 0., 1.)) if world_from_camera is None else np.asarray(world_from_camera)
    focal = height/(2*math.tan(math.radians(fov)/2))
    return {'camera_id': camera_id, 'width': width, 'height': height, 'fx': focal, 'fy': focal,
            'cx': (width-1)/2, 'cy': (height-1)/2, 'world_from_camera': pose.tolist()}
