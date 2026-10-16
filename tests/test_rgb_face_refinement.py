"""Image-only synthetic and fixed-RGB tests; no simulator/evaluation sources."""
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

from embodied_arm.perception.camera_perception import PixelDetection
from embodied_arm.perception.rgb_face_refinement import SOURCE, refine_rgb_landmark, refine_packet


WIDTH, HEIGHT = 640, 480
TOP = [(300, 210), (316, 200), (333, 212), (316, 223)]


def image_bytes(*, label='red', top=TOP, uniform=False, occlusion=False, duplicate=False,
                bright_side=False, goal=False):
    scale = 4
    image = Image.new('RGB', (WIDTH*scale, HEIGHT*scale), (210, 220, 230))
    draw = ImageDraw.Draw(image)
    def poly(points, colour):
        draw.polygon([(int(x*scale), int(y*scale)) for x, y in points], fill=colour)
    colours = {'red': ((210, 24, 24), (135, 15, 16), (160, 20, 20)),
               'blue': ((24, 70, 225), (16, 43, 140), (20, 53, 165)),
               'goal': ((18, 195, 55), (18, 195, 55), (18, 195, 55))}
    bright, dark, medium = colours[label]
    if uniform:
        dark = medium = bright
    if bright_side:
        bright, dark, medium = dark, bright, bright
    def cube(offset=0):
        points = [(x+offset, y) for x, y in top]
        p0,p1,p2,p3 = points
        if not goal:
            poly([p0, p3, (p3[0], p3[1]+18), (p0[0], p0[1]+18)], dark)
            poly([p3, p2, (p2[0], p2[1]+18), (p3[0], p3[1]+18)], medium)
        poly(points, bright)
    cube()
    if duplicate:
        cube(42)
    if occlusion:
        poly([(313, 199), (319, 199), (319, 224), (313, 224)], (220, 220, 220))
    image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    out = BytesIO(); image.save(out, 'JPEG', quality=93)
    return out.getvalue()


def refine(raw=None, label='red', coarse=(316, 223), **kwargs):
    return refine_rgb_landmark(raw if raw is not None else image_bytes(label=label), object_id=label,
                              uv_1000=[coarse[0]/(WIDTH-1)*1000, coarse[1]/(HEIGHT-1)*1000],
                              confidence=kwargs.pop('confidence', .98), width=WIDTH, height=HEIGHT, **kwargs)


def diagonal_centre(quad):
    p=np.array(quad,dtype=float)
    a,b=p[2]-p[0],p[3]-p[1]
    t=np.linalg.solve(np.column_stack((a,-b)),p[1]-p[0])[0]
    return p[0]+t*a


@pytest.mark.parametrize('label', ['red', 'blue'])
def test_recovers_visible_top_centre_from_biased_ai_body_point(label):
    result = refine(label=label)
    assert result.accepted, result.reason
    assert result.source == SOURCE
    np.testing.assert_allclose(result.uv, diagonal_centre(TOP), atol=.85)
    assert result.uv[1] < result.coarse_uv[1]-5
    assert len(result.corners) == 4
    assert min(result.edge_support_counts) >= 4
    assert result.edge_rms_px < 1
    assert result.face_support_ratio >= .9
    assert result.brightness_contrast >= 20
    assert result.pixel_sigma_px >= 1
    json.dumps(result.to_dict(), allow_nan=False)


def test_perspective_face_uses_diagonal_intersection_not_average_vertices():
    quad = [(298, 211), (320, 200), (342, 213), (311, 222)]
    result = refine(image_bytes(top=quad), coarse=(319, 225))
    assert result.accepted, result.reason
    true = diagonal_centre(quad)
    np.testing.assert_allclose(result.uv, true, atol=.85)
    assert np.linalg.norm(true-np.mean(quad, axis=0)) > .7


def test_planar_green_region_needs_no_dark_side_or_height_assumption():
    result = refine(image_bytes(label='goal', goal=True), label='goal', coarse=(313, 215))
    assert result.accepted, result.reason
    assert result.brightness_contrast is None
    np.testing.assert_allclose(result.uv, diagonal_centre(TOP), atol=.85)


@pytest.mark.parametrize('options', [dict(uniform=True), dict(occlusion=True), dict(bright_side=True)])
def test_unknown_top_face_or_occlusion_does_not_get_coordinates(options):
    result = refine(image_bytes(**options))
    assert not result.accepted
    assert result.uv is None and result.pixel_sigma_px is None and not result.corners


def test_ambiguous_multiple_coloured_objects_inside_ai_roi_rejected():
    result = refine(image_bytes(duplicate=True), coarse=(334, 223))
    assert not result.accepted
    assert result.reason in ('ambiguous_coloured_regions', 'roi_clipped')


def test_no_global_search_beyond_ai_region():
    result = refine(coarse=(100, 100))
    assert not result.accepted
    assert result.reason == 'no_coloured_region_near_ai_point'


def test_low_ai_confidence_has_no_geometry_fallback():
    result = refine(confidence=.5)
    assert not result.accepted and result.reason == 'low_ai_confidence'


@pytest.mark.parametrize('kwargs', [{'object_id':'yellow'}, {'uv_1000':[float('nan'),200]},
    {'confidence':True}, {'width':640.}, {'height':-1}])
def test_strict_inputs(kwargs):
    args = dict(object_id='red',uv_1000=[500,500],confidence=.9,width=WIDTH,height=HEIGHT)
    args.update(kwargs)
    with pytest.raises(ValueError):
        refine_rgb_landmark(image_bytes(), **args)


def test_non_jpeg_or_mismatched_dimensions_rejected():
    with pytest.raises(ValueError):
        refine_rgb_landmark(b'not image',object_id='red',uv_1000=[500,500],confidence=.9,width=WIDTH,height=HEIGHT)
    with pytest.raises(ValueError, match='dimensions'):
        refine_rgb_landmark(image_bytes(),object_id='red',uv_1000=[500,500],confidence=.9,width=WIDTH+1,height=HEIGHT)


def live_fixture():
    root = Path(__file__).resolve().parents[1]/'artifacts'
    paths = [root/'so101-vision-third-diagnostic.json',root/'so101-live-left.jpg',root/'so101-live-right.jpg']
    if not all(path.exists() for path in paths):
        pytest.skip('Fixed no-motion live RGB fixture not packaged')
    diagnostic = json.loads(paths[0].read_text())
    packet = {'frame_id':'offline-frozen-rgb-fixture','captured_at':100.,'images':[]}
    for camera, path in zip(('left','right'),paths[1:]):
        packet['images'].append({'camera_id':camera,'width':WIDTH,'height':HEIGHT,
                                 'data_url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()})
    detections = [PixelDetection(d['object_id'],d['camera_id'],packet['frame_id'],
                                (d['uv_1000'][0]*(WIDTH-1)/1000,d['uv_1000'][1]*(HEIGHT-1)/1000),
                                d['confidence'],2.,'visible',d['landmark']) for d in diagnostic['detections']]
    return packet,detections


def test_fixed_live_rgb_reports_corner_evidence_without_gt_scoring():
    packet, detections = live_fixture()
    result = refine_packet(packet,detections)
    assert result.accepted, [r.reason for r in result.evidence]
    assert result.source == SOURCE
    assert len(result.detections) == len(result.evidence) == 6
    for detection,evidence in zip(result.detections,result.evidence):
        assert detection.frame_id == packet['frame_id']
        assert detection.uv == evidence.uv
        assert len(evidence.corners) == 4
        assert evidence.edge_rms_px < 1
        assert evidence.face_support_ratio >= .9
    json.dumps(result.to_dict(),allow_nan=False)


def test_packet_rejects_extra_ground_truth_channels_and_wrong_frame():
    packet,detections=live_fixture()
    with pytest.raises(ValueError,match='RGB-only'):
        refine_packet({**packet,'depth':[1,2,3]},detections)
    packet['frame_id']='different-frame'
    with pytest.raises(ValueError,match='frame mismatch'):
        refine_packet(packet,detections)


def test_packet_failure_returns_unknown_not_coarse_pixel_fallback():
    packet,detections=live_fixture()
    item=detections[0]
    detections[0]=PixelDetection(item.object_id,item.camera_id,item.frame_id,(10.,10.),.98,2.,'visible',item.landmark)
    result=refine_packet(packet,detections)
    assert not result.accepted
    assert result.detections[0].visibility == 'unknown'
    assert result.detections[0].uv is None
