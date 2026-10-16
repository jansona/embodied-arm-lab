"""Experimental hybrid Qwen-region + RGB face geometry, entirely offline.

The AI supplies object identity and a coarse normalized pixel point. Only JPEG
pixels inside its bounded ROI are inspected. No camera extrinsics, simulator,
depth, supplied masks, body IDs, object sizes, or world coordinates are inputs.

This deliberately narrow experiment requires a uniformly coloured cuboid with a
brighter visible top face, darker connected sides, and four supported edges. It
rejects unsupported illumination/shape/occlusion; it is not general segmentation
or a guarantee that a fitted patch is the intended physical surface. The goal
uses a single green planar quadrilateral. Report this as HYBRID perception, not
pure Qwen. Image-only uncertainty excludes camera-calibration/model-ID errors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import math
from typing import Sequence

import numpy as np
from PIL import Image

SOURCE = 'hybrid_qwen_region_rgb_geometry'


@dataclass(frozen=True)
class FaceRefinement:
    object_id: str
    accepted: bool
    reason: str
    coarse_uv: tuple[float, float]
    roi_bounds: tuple[int, int, int, int]
    uv: tuple[float, float] | None = None
    pixel_sigma_px: float | None = None
    corners: tuple[tuple[float, float], ...] = ()
    edge_rms_px: float | None = None
    face_area_px2: float | None = None
    face_support_ratio: float | None = None
    brightness_contrast: float | None = None
    edge_support_counts: tuple[int, ...] = ()
    source: str = SOURCE

    def to_dict(self):
        return asdict(self)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _components(binary):
    """8-connected components from internally selected RGB pixels only."""
    seen = np.zeros(binary.shape, dtype=bool)
    components = []
    height, width = binary.shape
    for y, x in zip(*np.nonzero(binary)):
        if seen[y, x]:
            continue
        todo = [(int(y), int(x))]; seen[y, x] = True; points = []
        while todo:
            cy, cx = todo.pop(); points.append((cx, cy))
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; todo.append((ny, nx))
        if len(points) >= 12:
            components.append(np.asarray(points, dtype=float))
    return components


def _hull(points):
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        raise ValueError('insufficient_boundary')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1], dtype=float)


def _area(polygon):
    x, y = polygon.T
    return abs(float(np.dot(x, np.roll(y, -1))-np.dot(y, np.roll(x, -1))))/2


def _contour(field, binary, threshold):
    """Subpixel threshold crossings of actual RGB-derived chromatic intensity."""
    points = []
    height, width = field.shape
    for y, x in zip(*np.nonzero(binary)):
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            ny, nx = y+dy, x+dx
            if not (0 <= nx < width and 0 <= ny < height):
                raise ValueError('roi_clipped')
            if not binary[ny, nx]:
                high, low = float(field[y, x]), float(field[ny, nx])
                if high <= low or not low <= threshold <= high:
                    raise ValueError('invalid_face_boundary')
                fraction = (high-threshold)/(high-low)
                points.append((x+dx*fraction, y+dy*fraction))
    if len(points) < 12:
        raise ValueError('insufficient_boundary')
    return np.asarray(points)


def _quadrilateral(contour):
    hull = _hull(contour)
    original_area = _area(hull)
    if len(hull) < 4:
        raise ValueError('not_four_sided')
    # Remove only the least consequential hull vertex until four remain. Strong
    # concavity or a non-quadrilateral silhouette fails the area/support gates.
    while len(hull) > 4:
        costs = [_area(np.asarray([hull[i-1], hull[i], hull[(i+1) % len(hull)]])) for i in range(len(hull))]
        hull = np.delete(hull, int(np.argmin(costs)), axis=0)
    if _area(hull)/original_area < .78:
        raise ValueError('not_four_sided')
    corners = hull
    residual = None
    support_counts = ()
    for _ in range(4):
        distances = []
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            v = b-a
            if np.linalg.norm(v) < 3:
                raise ValueError('face_too_small')
            n = np.array([-v[1], v[0]])/np.linalg.norm(v)
            distances.append(np.abs((contour-a) @ n))
        assignments = np.argmin(np.array(distances), axis=0)
        lines = []; residuals = []; counts = []
        for edge in range(4):
            points = contour[assignments == edge]
            if len(points) < 4:
                raise ValueError('unsupported_edge')
            centre = points.mean(axis=0)
            _, _, vh = np.linalg.svd(points-centre, full_matrices=False)
            normal = vh[-1]
            lines.append((normal, float(normal @ centre)))
            residuals.extend((points-centre) @ normal)
            counts.append(len(points))
        refined = []
        for i in range(4):
            a, b = lines[i-1], lines[i]
            matrix = np.array([a[0], b[0]])
            if abs(float(np.linalg.det(matrix))) < .15:
                raise ValueError('degenerate_corner')
            refined.append(np.linalg.solve(matrix, [a[1], b[1]]))
        corners = np.asarray(refined)
        residual = float(np.sqrt(np.mean(np.square(residuals))))
        support_counts = tuple(counts)
    if residual > 1.:
        raise ValueError('irregular_face_boundary')
    if _area(corners) < 28:
        raise ValueError('face_too_small')
    # All corners must be convex and close to observed boundary evidence.
    crosses = []
    for i in range(4):
        a, b, c = corners[i], corners[(i+1) % 4], corners[(i+2) % 4]
        crosses.append(float((b-a)[0]*(c-b)[1]-(b-a)[1]*(c-b)[0]))
        if float(np.min(np.linalg.norm(contour-a, axis=1))) > 2.:
            raise ValueError('unsupported_corner')
    if not (all(v > 0 for v in crosses) or all(v < 0 for v in crosses)):
        raise ValueError('nonconvex_face')
    return corners, residual, support_counts


def _inside(points, polygon):
    signs = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        v = b-a; w = points-a
        signs.append(v[0]*w[:, 1]-v[1]*w[:, 0])
    signs = np.array(signs)
    return np.all(signs >= -1e-8, axis=0) | np.all(signs <= 1e-8, axis=0)


def refine_rgb_landmark(jpeg: bytes, *, object_id: str, uv_1000: Sequence[float],
                        confidence: float, width: int, height: int) -> FaceRefinement:
    """Refine a Qwen coarse point to an observed face centre, in native pixels.

    The ROI is fixed by image dimensions and the AI point (7% of image width for
    cubes, 10% for the marker). Never search outside it or use a world reference.
    Returned corners are cyclic; centre is diagonal intersection, not the image
    centroid, because the physical square centre projects projectively.
    """
    if object_id not in ('red', 'blue', 'goal'):
        raise ValueError('Unsupported semantic object')
    if (not isinstance(uv_1000, (tuple, list)) or len(uv_1000) != 2
            or not all(_finite(v) and 0 <= v <= 1000 for v in uv_1000)):
        raise ValueError('Coarse point must be normalized_1000 pixels')
    if not _finite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid AI confidence')
    if any(type(v) is not int or not 32 <= v <= 4096 for v in (width, height)):
        raise ValueError('Invalid image dimensions')
    if not isinstance(jpeg, bytes) or not 1 <= len(jpeg) <= 8*1024*1024:
        raise ValueError('Expected bounded JPEG bytes')
    coarse = (float(uv_1000[0]*(width-1)/1000), float(uv_1000[1]*(height-1)/1000))
    radius = max(24, math.ceil(width*(.10 if object_id == 'goal' else .07)))
    roi = (max(0, math.floor(coarse[0]-radius)), max(0, math.floor(coarse[1]-radius)),
           min(width, math.ceil(coarse[0]+radius+1)), min(height, math.ceil(coarse[1]+radius+1)))
    def reject(reason):
        return FaceRefinement(object_id, False, reason, coarse, roi)
    if confidence < .75:
        return reject('low_ai_confidence')
    try:
        with Image.open(BytesIO(jpeg)) as image:
            if image.format != 'JPEG' or image.mode != 'RGB' or image.size != (width, height):
                raise ValueError('Expected RGB JPEG matching dimensions')
            rgb = np.asarray(image.crop(roi), dtype=float)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid JPEG') from exc
    r, g, b = np.moveaxis(rgb, 2, 0)
    if object_id == 'red':
        dominant, other_a, other_b = r, g, b
    elif object_id == 'blue':
        dominant, other_a, other_b = b, r, g
    else:
        dominant, other_a, other_b = g, r, b
    chroma = dominant-(other_a+other_b)/2
    from embodied_arm.perception.rgb_semantic_colour import semantic_colour_mask
    colour = semantic_colour_mask(rgb, object_id)
    components = _components(colour)
    local_coarse = np.asarray(coarse)-roi[:2]
    plausible = []
    for points in components:
        near = float(np.min(np.linalg.norm(points-local_coarse, axis=1)))
        if near <= radius*.5:
            plausible.append((near, points))
    if not plausible:
        return reject('no_coloured_region_near_ai_point')
    if len(plausible) != 1:
        return reject('ambiguous_coloured_regions')
    points = plausible[0][1]
    xs, ys = points.T.astype(int)
    selected = np.zeros(colour.shape, dtype=bool); selected[ys, xs] = True
    if np.any(xs <= 0) or np.any(xs >= selected.shape[1]-1) or np.any(ys <= 0) or np.any(ys >= selected.shape[0]-1):
        return reject('roi_clipped')
    values = chroma[ys, xs]
    contrast = None
    if object_id != 'goal':
        # Fit two photometric populations using eroded interior pixels, so JPEG
        # colour bleed/background edges cannot masquerade as the darker face.
        interior = selected.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            interior &= np.roll(selected, (dy, dx), axis=(0, 1))
        inner_values = chroma[interior]
        if len(inner_values) < 20:
            return reject('face_too_small')
        low, high = np.quantile(inner_values, [.2, .8])
        for _ in range(15):
            split = (low+high)/2
            lower, upper = inner_values[inner_values <= split], inner_values[inner_values > split]
            if min(len(lower), len(upper)) < 6:
                return reject('no_distinct_top_face')
            new_low, new_high = float(np.median(lower)), float(np.median(upper))
            if abs(new_low-low)+abs(new_high-high) < .01:
                break
            low, high = new_low, new_high
        contrast = float(high-low)
        if contrast < 20 or contrast/max(float(high), 1) < .12:
            return reject('no_distinct_top_face')
        threshold = (float(low)+float(high))/2
    else:
        threshold = max(32, float(np.median(values))*.5)
    bright = selected & (chroma >= threshold)
    faces = _components(bright)
    if len(faces) != 1:
        return reject('ambiguous_face_patches')
    face_points = faces[0]
    face_binary = np.zeros_like(bright)
    face_binary[face_points[:, 1].astype(int), face_points[:, 0].astype(int)] = True
    if object_id != 'goal':
        dark = points[chroma[ys, xs] < threshold]
        fraction = len(face_points)/len(points)
        if not .18 <= fraction <= .8 or len(dark) < 12:
            return reject('no_supported_side_faces')
        if float(np.mean(dark[:, 1])-np.mean(face_points[:, 1])) < 2:
            return reject('bright_patch_not_top_face')
    try:
        # Nonselected regions may not supply an extrapolated face boundary.
        contour_field = np.where(selected, chroma, np.minimum(chroma, threshold-1e-6))
        contour = _contour(contour_field, face_binary, threshold)
        corners, residual, counts = _quadrilateral(contour)
        area = _area(corners)
        grid_y, grid_x = np.indices(face_binary.shape)
        pixel_positions = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        inside = _inside(pixel_positions, corners).reshape(face_binary.shape)
        support = float(np.count_nonzero(inside & face_binary)/max(1, np.count_nonzero(inside)))
        if support < .9:
            raise ValueError('occluded_or_nonquadrilateral_face')
        if np.count_nonzero(face_binary & ~inside) > max(8, len(face_points)*.18):
            raise ValueError('unexplained_face_pixels')
        diagonal_a, diagonal_b = corners[2]-corners[0], corners[3]-corners[1]
        parameter = np.linalg.solve(np.column_stack((diagonal_a, -diagonal_b)), corners[1]-corners[0])
        if not np.all((parameter > .1) & (parameter < .9)):
            raise ValueError('invalid_diagonal_intersection')
        centre = corners[0]+parameter[0]*diagonal_a
        if not _inside(np.array([centre]), corners)[0]:
            raise ValueError('invalid_face_centre')
        # Fixed >=1px floor accounts for JPEG threshold/corner ambiguity. The
        # stereo layer independently retains its own conservative >=2px floor.
        sigma = max(1., residual*2 + .5, 1.5/math.sqrt(max(area, 1)/50))
        offset = np.asarray(roi[:2])
        return FaceRefinement(object_id, True, 'refined_visible_face', coarse, roi,
                              tuple(float(v) for v in centre+offset), float(sigma),
                              tuple(tuple(float(v) for v in corner+offset) for corner in corners),
                              residual, area, support, contrast, counts)
    except (ValueError, np.linalg.LinAlgError) as exc:
        known = {'roi_clipped', 'invalid_face_boundary', 'insufficient_boundary', 'not_four_sided',
                 'face_too_small', 'unsupported_edge', 'degenerate_corner', 'irregular_face_boundary',
                 'unsupported_corner', 'nonconvex_face', 'occluded_or_nonquadrilateral_face',
                 'unexplained_face_pixels', 'invalid_diagonal_intersection', 'invalid_face_centre'}
        return reject(str(exc) if str(exc) in known else 'invalid_face_geometry')


@dataclass(frozen=True)
class RefinedPacket:
    accepted: bool
    reason: str
    detections: tuple
    evidence: tuple[FaceRefinement, ...]
    source: str = SOURCE

    def to_dict(self):
        return {'accepted': self.accepted, 'reason': self.reason, 'source': self.source,
                'detections': [item.to_dict() for item in self.detections],
                'evidence': [item.to_dict() for item in self.evidence]}


def refine_packet(packet: dict, canonical_ai_detections: Sequence) -> RefinedPacket:
    """Adapt RGB image packet + canonical AI pixels to canonical refined pixels.

    Caller owns frame freshness and generation checks. This helper never receives
    calibration or world positions. A failed view becomes explicitly unknown;
    packet.accepted is false unless all six landmarks pass. No coarse fallback.
    """
    import base64
    import binascii
    from embodied_arm.perception.camera_perception import PixelDetection

    if not isinstance(packet, dict) or set(packet) != {'frame_id', 'captured_at', 'images'}:
        raise ValueError('Expected RGB-only packet fields')
    if not isinstance(packet['frame_id'], str) or not _finite(packet['captured_at']):
        raise ValueError('Invalid frame provenance')
    images = packet['images']
    if not isinstance(images, (list, tuple)) or len(images) != 2:
        raise ValueError('Expected two RGB images')
    by_camera = {}
    for item in images:
        if (not isinstance(item, dict) or set(item) != {'camera_id', 'width', 'height', 'data_url'}
                or item['camera_id'] not in ('left', 'right') or item['camera_id'] in by_camera):
            raise ValueError('Invalid RGB image fields')
        data = item['data_url']
        if not isinstance(data, str) or not data.startswith('data:image/jpeg;base64,') or len(data) > 12*1024*1024:
            raise ValueError('Expected JPEG data URL')
        try:
            raw = base64.b64decode(data.split(',', 1)[1], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError('Invalid JPEG base64') from exc
        by_camera[item['camera_id']] = (raw, item['width'], item['height'])
    if (not isinstance(canonical_ai_detections, (list, tuple)) or len(canonical_ai_detections) != 6
            or any(not isinstance(item, PixelDetection) for item in canonical_ai_detections)):
        raise ValueError('Expected six canonical AI pixel detections')
    expected = {(camera, obj) for camera in ('left', 'right') for obj in ('red', 'blue', 'goal')}
    actual = {(item.camera_id, item.object_id) for item in canonical_ai_detections}
    if actual != expected or any(item.frame_id != packet['frame_id'] for item in canonical_ai_detections):
        raise ValueError('Detection correspondence or frame mismatch')
    refined, evidence = [], []
    for detection in canonical_ai_detections:
        raw, width, height = by_camera[detection.camera_id]
        if detection.visibility != 'visible':
            refined.append(detection)
            continue
        if any(type(v) is not int or v <= 1 for v in (width, height)):
            raise ValueError('Invalid camera dimensions')
        normalized = [detection.uv[0]/(width-1)*1000, detection.uv[1]/(height-1)*1000]
        result = refine_rgb_landmark(raw, object_id=detection.object_id, uv_1000=normalized,
                                     confidence=detection.confidence, width=width, height=height)
        evidence.append(result)
        refined.append(PixelDetection(detection.object_id, detection.camera_id, detection.frame_id,
                                      result.uv if result.accepted else None,
                                      detection.confidence if result.accepted else 0.,
                                      result.pixel_sigma_px if result.accepted else None,
                                      'visible' if result.accepted else 'unknown', detection.landmark))
    accepted = all(item.visibility == 'visible' for item in refined)
    return RefinedPacket(accepted, 'all_visible_faces_refined' if accepted else 'refinement_failed_closed',
                         tuple(refined), tuple(evidence))
