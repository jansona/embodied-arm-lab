"""OPT-IN footprint-scaled hybrid RGB refinement; the baseline is unchanged.

Inputs and coordinate conventions match rgb_face_refinement. Every measurement is
made inside the AI-selected native-pixel ROI, with no resizing or camera/world
inputs. Geometry tolerances normalize to the observed face footprint, not camera
resolution. At sqrt(face_area/200) <= 1 the original pixel tolerances remain.

Frozen profile assumptions/limits (footprint-scaled-v1):
- A single coloured connected object, bright top face, darker connected sides
- RMS <=1 footprint unit; per-edge RMS <=1.5; corner evidence within2 units
- A unit is max(1, sqrt(observed face area /200)) native pixels
- At most8 additional bright components, each <=2% and aggregate <=3% of main
- Each minor component diameter <=0.40 sqrt(main pixel area), and every minor
  pixel is within1.00 sqrt(main area) of the main component
- All bright components, including single pixels, count toward those limits
- The unchanged90% interior-support gate rejects meaningful holes/occlusion

These bounded minor patches may be photometric/JPEG fragments on connected side
faces; competing surfaces, remote patches, and aggregate substantial fragments
are rejected. This is an experimental appearance assumption, not generic object
recognition or calibrated confidence. Native residual/corner/speckle evidence
can only increase the baseline image uncertainty formula; stereo's independent
2px floor is untouched. No production caller selects this profile implicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import math
from typing import Sequence

import numpy as np
from PIL import Image

from rgb_face_refinement import (FaceRefinement, RefinedPacket, _finite, _components,
                                 _hull, _area, _contour, _inside)

SOURCE = 'hybrid_qwen_region_rgb_geometry_scaled'
PROFILE = 'footprint_scaled_v1'
REFERENCE_FACE_AREA_PX2 = 200.
MAX_MINOR_COMPONENTS = 8
MAX_MINOR_AREA_FRACTION = .02
MAX_TOTAL_MINOR_AREA_FRACTION = .03
MAX_MINOR_DIAMETER_FOOTPRINT = .40
MAX_MINOR_DISTANCE_FOOTPRINT = 1.00


@dataclass(frozen=True)
class ScaleAwareRefinement(FaceRefinement):
    source: str = SOURCE
    profile: str = PROFILE
    footprint_scale: float | None = None
    normalized_edge_rms: float | None = None
    max_corner_gap_px: float | None = None
    max_edge_rms_px: float | None = None
    discarded_bright_component_areas: tuple[int, ...] = ()
    discarded_bright_area_fraction: float = 0.
    discarded_bright_max_distance_px: float = 0.


def _all_components(binary):
    """Unlike the baseline helper, count even one-pixel bright islands."""
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
        components.append(np.asarray(points, dtype=float))
    return components


def _select_face(components):
    if not components:
        raise ValueError('ambiguous_face_patches')
    faces = sorted(components, key=len, reverse=True)
    main, minor = faces[0], faces[1:]
    if len(main) < 28:
        raise ValueError('face_too_small')
    areas = tuple(len(points) for points in minor)
    area_fraction = sum(areas)/len(main)
    footprint = math.sqrt(len(main))
    if (len(minor) > MAX_MINOR_COMPONENTS or area_fraction > MAX_TOTAL_MINOR_AREA_FRACTION
            or any(area/len(main) > MAX_MINOR_AREA_FRACTION for area in areas)):
        raise ValueError('ambiguous_face_patches')
    maximum_distance = 0.
    for points in minor:
        diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
        if diameter > MAX_MINOR_DIAMETER_FOOTPRINT*footprint:
            raise ValueError('ambiguous_face_patches')
        # Bounded small islands make this small. Chunk the main footprint so
        # unusually large native ROIs cannot create an unbounded NxM allocation.
        distances = np.full(len(points), np.inf)
        for start in range(0, len(main), 2048):
            difference = points[:, None, :]-main[None, start:start+2048, :]
            distances = np.minimum(distances, np.sqrt(np.min(np.sum(difference**2, axis=2), axis=1)))
        farthest = float(np.max(distances))
        maximum_distance = max(maximum_distance, farthest)
        if farthest > MAX_MINOR_DISTANCE_FOOTPRINT*footprint:
            raise ValueError('ambiguous_face_patches')
    return main, areas, float(area_fraction), maximum_distance


def _quadrilateral(contour):
    hull = _hull(contour)
    original_area = _area(hull)
    scale = max(1., math.sqrt(original_area / REFERENCE_FACE_AREA_PX2))
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
            if np.linalg.norm(v) < 3*scale:
                raise ValueError('face_too_small')
            n = np.array([-v[1], v[0]])/np.linalg.norm(v)
            distances.append(np.abs((contour-a) @ n))
        assignments = np.argmin(np.array(distances), axis=0)
        lines = []; residuals = []; counts = []; edge_rms = []
        for edge in range(4):
            points = contour[assignments == edge]
            if len(points) < max(4, math.ceil(4*scale)):
                raise ValueError('unsupported_edge')
            centre = points.mean(axis=0)
            _, _, vh = np.linalg.svd(points-centre, full_matrices=False)
            normal = vh[-1]
            lines.append((normal, float(normal @ centre)))
            errors = (points-centre) @ normal
            residuals.extend(errors)
            edge_rms.append(float(np.sqrt(np.mean(errors**2))))
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
    if residual > scale or max(edge_rms) > 1.5*scale:
        raise ValueError('irregular_face_boundary')
    if _area(corners) < 28:
        raise ValueError('face_too_small')
    # All corners must be convex and close to observed boundary evidence.
    crosses = []; corner_gaps = []
    for i in range(4):
        a, b, c = corners[i], corners[(i+1) % 4], corners[(i+2) % 4]
        crosses.append(float((b-a)[0]*(c-b)[1]-(b-a)[1]*(c-b)[0]))
        gap = float(np.min(np.linalg.norm(contour-a, axis=1)))
        corner_gaps.append(gap)
        if gap > 2*scale:
            raise ValueError('unsupported_corner')
    if not (all(v > 0 for v in crosses) or all(v < 0 for v in crosses)):
        raise ValueError('nonconvex_face')
    return corners, residual, support_counts, scale, max(corner_gaps), max(edge_rms)



def refine_rgb_landmark(jpeg: bytes, *, object_id: str, uv_1000: Sequence[float],
                        confidence: float, width: int, height: int) -> ScaleAwareRefinement:
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
        return ScaleAwareRefinement(object_id, False, reason, coarse, roi)
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
    colour = (dominant >= 55) & (chroma >= 32) & (dominant > 1.25*other_a) & (dominant > 1.25*other_b)
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
    faces = _all_components(bright)
    try:
        face_points, discarded_areas, discarded_fraction, discarded_distance = _select_face(faces)
    except ValueError as exc:
        return reject(str(exc))
    component_scale = max(1., math.sqrt(len(face_points)/REFERENCE_FACE_AREA_PX2))
    face_binary = np.zeros_like(bright)
    face_binary[face_points[:, 1].astype(int), face_points[:, 0].astype(int)] = True
    if object_id != 'goal':
        dark = points[chroma[ys, xs] < threshold]
        fraction = len(face_points)/len(points)
        if not .18 <= fraction <= .8 or len(dark) < 12:
            return reject('no_supported_side_faces')
        if float(np.mean(dark[:, 1])-np.mean(face_points[:, 1])) < 2*component_scale:
            return reject('bright_patch_not_top_face')
    try:
        # Nonselected regions may not supply an extrapolated face boundary.
        contour_field = np.where(selected, chroma, np.minimum(chroma, threshold-1e-6))
        contour = _contour(contour_field, face_binary, threshold)
        corners, residual, counts, scale, corner_gap, max_edge_rms = _quadrilateral(contour)
        area = _area(corners)
        grid_y, grid_x = np.indices(face_binary.shape)
        pixel_positions = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        inside = _inside(pixel_positions, corners).reshape(face_binary.shape)
        support = float(np.count_nonzero(inside & face_binary)/max(1, np.count_nonzero(inside)))
        if support < .9:
            raise ValueError('occluded_or_nonquadrilateral_face')
        if np.count_nonzero(face_binary & ~inside) > max(8*scale**2, len(face_points)*.18):
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
        sigma = max(1., residual*2 + .5, max_edge_rms*2 + .5,
                    corner_gap/2 + discarded_fraction*discarded_distance,
                    1.5/math.sqrt(max(area, 1)/50))
        offset = np.asarray(roi[:2])
        return ScaleAwareRefinement(object_id, True, 'refined_visible_face', coarse, roi,
                              tuple(float(v) for v in centre+offset), float(sigma),
                              tuple(tuple(float(v) for v in corner+offset) for corner in corners),
                              residual, area, support, contrast, counts,
                              footprint_scale=scale, normalized_edge_rms=residual/scale,
                              max_corner_gap_px=corner_gap, max_edge_rms_px=max_edge_rms,
                              discarded_bright_component_areas=discarded_areas,
                              discarded_bright_area_fraction=discarded_fraction,
                              discarded_bright_max_distance_px=discarded_distance)
    except (ValueError, np.linalg.LinAlgError) as exc:
        known = {'roi_clipped', 'invalid_face_boundary', 'insufficient_boundary', 'not_four_sided',
                 'face_too_small', 'unsupported_edge', 'degenerate_corner', 'irregular_face_boundary',
                 'unsupported_corner', 'nonconvex_face', 'occluded_or_nonquadrilateral_face',
                 'unexplained_face_pixels', 'invalid_diagonal_intersection', 'invalid_face_centre'}
        return reject(str(exc) if str(exc) in known else 'invalid_face_geometry')


def refine_packet(packet: dict, canonical_ai_detections: Sequence) -> RefinedPacket:
    """Adapt RGB image packet + canonical AI pixels to canonical refined pixels.

    Caller owns frame freshness and generation checks. This helper never receives
    calibration or world positions. A failed view becomes explicitly unknown;
    packet.accepted is false unless all six landmarks pass. No coarse fallback.
    """
    import base64
    import binascii
    from camera_perception import PixelDetection

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
                         tuple(refined), tuple(evidence), SOURCE)
