"""TEST ONLY colour coarse detector reading rendered JPEG pixels, never GT.

This is not Qwen and not a production detector. Its output must pass the same
RGB face refinement/stereo uncertainty gates. No projected object coordinates,
render segmentation, depth, body positions, or tuned covariance are used.
"""
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from embodied_arm.perception.camera_perception import PixelDetection, LANDMARKS
from embodied_arm.perception.rgb_semantic_colour import semantic_colour_mask
from embodied_arm.perception.rgb_face_refinement import _components, refine_rgb_landmark
from embodied_arm.providers.vision_adapter import VisionResult


class RGBColourFixture:
    def status(self):return {'available':True,'model':'offline-colour-fixture','provenance':'offline_fixture'}
    def __init__(self, *, profile='baseline'):
        self.profile = profile

    def perceive(self, packet):
        if self.profile == 'scaled':
            from embodied_arm.perception.rgb_scaled_refinement import refine_rgb_landmark
        else:
            from embodied_arm.perception.rgb_face_refinement import refine_rgb_landmark
        detections = []
        for item in packet['images']:
            raw = base64.b64decode(item['data_url'].split(',', 1)[1])
            rgb = np.asarray(Image.open(BytesIO(raw)).convert('RGB'), dtype=float)
            for object_id, channel in (('red', 0), ('blue', 2), ('goal', 1)):
                others = [i for i in range(3) if i != channel]
                strength = rgb[:, :, channel] - np.max(rgb[:, :, others], axis=2)
                components = _components(semantic_colour_mask(rgb, object_id))
                candidates = []
                for points in components:
                    if len(points) < 100:
                        continue
                    uv = tuple(float(x) for x in points.mean(axis=0))
                    face = refine_rgb_landmark(raw, object_id=object_id,
                        uv_1000=[uv[0]/(item['width']-1)*1000, uv[1]/(item['height']-1)*1000],
                        confidence=.9, width=item['width'], height=item['height'])
                    if face.accepted:
                        candidates.append(points)
                # Coarse compact-cuboid shape filter removes texture streaks.
                # This remains only an ROI seed: native face validation is
                # mandatory and receives unchanged pixel uncertainty.
                compact = []
                for points in components:
                    extent = np.ptp(points, axis=0) + 1
                    density = len(points) / np.prod(extent)
                    aspect = extent[0] / extent[1]
                    if len(points) >= 100 and density >= .65 and .4 <= aspect <= 2.5:
                        compact.append(points)
                if not candidates and len(compact) == 1:
                    candidates = compact
                components = candidates
                # Multiple same-colour components are ambiguous; never silently
                # select the one closest to a simulator-projected location.
                if len(components) != 1:
                    detections.append(PixelDetection(object_id, item['camera_id'], packet['frame_id'],
                        None, 0., None, 'ambiguous' if components else 'absent', LANDMARKS[object_id]))
                    continue
                uv = tuple(float(x) for x in components[0].mean(axis=0))
                detections.append(PixelDetection(object_id, item['camera_id'], packet['frame_id'],
                    uv, .9, 8., 'visible', LANDMARKS[object_id]))
        return VisionResult(True, 'accepted', packet['frame_id'], packet['captured_at'],
                            tuple(detections), model='offline-colour-fixture', calls=0,
                            provenance='offline_rgb_colour_fixture')
