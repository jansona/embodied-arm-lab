"""Frozen image-only fixtures: no simulator, API requests, or credentials."""
import base64
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
import json
from unittest.mock import patch

from PIL import Image, ImageDraw
import pytest

from embodied_arm.perception.camera_perception import PixelDetection
from embodied_arm.perception.rgb_coarse_packet import (JPEG_PREFIX, PixelScale, prepare_coarse_packet,
                               remap_detections, remap_result)
from embodied_arm.providers.vision_adapter import VisionResult, frame_snapshot
from embodied_arm.runtime.vision_ipc import normalize_packet


def make_packet(sizes=((1920, 1440), (1920, 1440)), *, mode='RGB', fmt='JPEG'):
    images=[]
    for cid,size in zip(('left','right'),sizes):
        image=Image.new('RGB',size,(200,210,220))
        draw=ImageDraw.Draw(image);draw.rectangle((100,100,min(300,size[0]-1),min(300,size[1]-1)),fill=(240,20,20))
        if mode!='RGB':image=image.convert(mode)
        output=BytesIO();image.save(output,fmt,quality=90)
        images.append({'camera_id':cid,'width':size[0],'height':size[1],
            'data_url':JPEG_PREFIX+base64.b64encode(output.getvalue()).decode()})
    return {'frame_id':'frozen-frame-7','captured_at':100.,'images':images}


def detections(prepared, *, uv=(100.,100.), sigma=3.):
    return tuple(PixelDetection(obj,cid,prepared.frame_id,uv,.95,sigma,'visible',
                    'goal_center' if obj=='goal' else 'top_face_center')
                 for obj in ('red','blue','goal') for cid in ('left','right'))


def test_high_resolution_downsample_preserves_frame_and_passes_existing_boundary():
    native=make_packet();before=deepcopy(native)
    with patch('httpx.Client',side_effect=AssertionError('No networking')):
        prepared=prepare_coarse_packet(native,now=100.)
    coarse=prepared.packet
    assert native==before
    assert coarse['frame_id']==native['frame_id'] and coarse['captured_at']==native['captured_at']
    assert set(coarse)=={'frame_id','captured_at','images'}
    assert [(i['width'],i['height']) for i in coarse['images']]==[(960,720),(960,720)]
    assert all(i['data_url']!=n['data_url'] for i,n in zip(coarse['images'],native['images']))
    assert frame_snapshot(coarse,now=100.)==coarse
    with patch('embodied_arm.providers.vision_adapter.time.monotonic',return_value=100.):
        assert normalize_packet(coarse)==coarse
    assert len(json.dumps(coarse))<1024*1024
    with Image.open(BytesIO(base64.b64decode(coarse['images'][0]['data_url'][len(JPEG_PREFIX):]))) as image:
        pixel=image.getpixel((100,100))
        assert pixel[0]>200 and pixel[1]<40 and pixel[2]<40


def test_native_remap_uses_pixel_centres_not_align_corners():
    native=make_packet();prepared=prepare_coarse_packet(native,now=100.)
    mapped=remap_detections(detections(prepared),prepared,native_packet=native,now=100.)
    assert mapped[0].uv==(200.5,200.5)
    assert mapped[0].pixel_sigma_px==6
    assert mapped[0].confidence==.95 and mapped[0].frame_id==prepared.frame_id
    transform=prepared.transforms[0]
    assert transform.to_native((0,0))==(.5,.5)
    assert transform.to_native((959,719))==(1918.5,1438.5)
    assert transform.to_native((479.5,359.5))==(959.5,719.5)


def test_noninteger_anisotropic_rounding_uses_conservative_sigma_per_camera():
    native=make_packet(((1919,1439),(1280,1000)))
    prepared=prepare_coarse_packet(native,now=100.)
    result=remap_detections(detections(prepared),prepared,native_packet=native,now=100.)
    transforms={t.camera_id:t for t in prepared.transforms}
    for detection in result:
        transform=transforms[detection.camera_id]
        assert detection.uv==pytest.approx(((100.5)*transform.scale_x-.5,(100.5)*transform.scale_y-.5))
        assert detection.pixel_sigma_px==3*max(transform.scale_x,transform.scale_y)
        assert transform.model_width<=960 and transform.model_height<=720


def test_no_scale_preserves_jpeg_coordinates_sigma_and_is_detached():
    native=make_packet(((640,480),(640,480)))
    prepared=prepare_coarse_packet(native,now=100.)
    assert prepared.packet==native
    values=detections(prepared,uv=(1e-14,100.),sigma=2.)
    assert remap_detections(values,prepared,native_packet=native,now=100.)==values
    copy=prepared.packet;copy['images'][0]['width']=99
    assert prepared.packet['images'][0]['width']==640
    with pytest.raises(FrozenInstanceError):prepared.frame_id='changed'


@pytest.mark.parametrize('uv',[(-.001,0),(960,0),(959.1,0),(0,720),(float('nan'),0),(True,0)])
def test_out_of_bounds_coarse_centres_fail_closed_without_clamping(uv):
    transform=PixelScale('left',1920,1440,960,720)
    with pytest.raises(ValueError):transform.to_native(uv)


def test_native_pixels_are_not_clamped_to_old_coarse_uncertainty_limit():
    native=make_packet();prepared=prepare_coarse_packet(native,now=100.)
    mapped=remap_detections(detections(prepared,sigma=25),prepared,native_packet=native,now=100.)
    assert mapped[0].pixel_sigma_px==50
    with pytest.raises(ValueError):remap_detections(detections(prepared,sigma=26),prepared,native_packet=native,now=100.)


def test_nonvisible_observation_stays_missing_and_cannot_become_accepted_result():
    native=make_packet();prepared=prepare_coarse_packet(native,now=100.)
    values=list(detections(prepared))
    values[0]=replace(values[0],uv=None,pixel_sigma_px=None,confidence=0,visibility='occluded')
    mapped=remap_detections(values,prepared,native_packet=native,now=100.)
    assert mapped[0].uv is None and mapped[0].pixel_sigma_px is None and mapped[0].visibility=='occluded'
    claimed=VisionResult(True,'accepted',prepared.frame_id,prepared.captured_at,tuple(values))
    with pytest.raises(ValueError):remap_result(claimed,prepared,native_packet=native,now=100.)


def test_result_requires_frame_time_identity_and_keeps_other_metadata():
    native=make_packet();prepared=prepare_coarse_packet(native,now=100.)
    result=VisionResult(True,'accepted',prepared.frame_id,prepared.captured_at,detections(prepared),calls=2)
    native_result=remap_result(result,prepared,native_packet=native,submitted_packet=prepared.packet,now=100.)
    assert native_result.calls==2 and native_result.model==result.model and native_result.provenance==result.provenance
    assert result.detections[0].uv==(100.,100.) and native_result.detections[0].uv==(200.5,200.5)
    for change in ({'frame_id':'wrong'},{'captured_at':100.01},{'accepted':False},{'reason':'timeout'}):
        with pytest.raises(ValueError):remap_result(replace(result,**change),prepared,native_packet=native,now=100.)


def test_image_digest_binding_rejects_changed_native_and_submitted_images():
    native=make_packet();prepared=prepare_coarse_packet(native,now=100.)
    for change in ('frame','time','dimensions','pixels'):
        bad=deepcopy(native)
        if change=='frame':bad['frame_id']='other'
        if change=='time':bad['captured_at']=99.9
        if change=='dimensions':bad['images'][0]['width']=1919
        if change=='pixels':bad['images'][0]['data_url']=bad['images'][0]['data_url'][:-4]+'AAAA'
        with pytest.raises(ValueError):remap_detections(detections(prepared),prepared,native_packet=bad,now=100.)
    bad=prepared.packet;bad['images'][0]['data_url']=native['images'][0]['data_url']
    with pytest.raises(ValueError):remap_detections(detections(prepared),prepared,native_packet=native,submitted_packet=bad,now=100.)
    forged=replace(prepared,transforms=(replace(prepared.transforms[0],native_width=1919),prepared.transforms[1]))
    with pytest.raises(ValueError):remap_detections(detections(prepared),forged,native_packet=native,now=100.)


def test_mismatched_detection_frame_duplicate_camera_and_size_rejected():
    native=make_packet();prepared=prepare_coarse_packet(native,now=100.)
    values=list(detections(prepared));values[0]=replace(values[0],frame_id='other')
    with pytest.raises(ValueError):remap_detections(values,prepared,native_packet=native,now=100.)
    values=list(detections(prepared));values[1]=values[0]
    with pytest.raises(ValueError):remap_detections(values,prepared,native_packet=native,now=100.)
    values=list(detections(prepared));values.pop()
    with pytest.raises(ValueError):remap_detections(values,prepared,native_packet=native,now=100.)


def test_stale_future_and_forbidden_metadata_are_rejected():
    native=make_packet()
    for now in (99.,161.):
        with pytest.raises(ValueError):prepare_coarse_packet(native,now=now)
    prepared=prepare_coarse_packet(native,now=100.)
    with pytest.raises(ValueError):remap_detections(detections(prepared),prepared,native_packet=native,now=161.)
    for key in ('calibrations','depth','segmentation','body_id','api_key','headers','endpoint'):
        with pytest.raises(ValueError):prepare_coarse_packet({**native,key:'SECRET'},now=100.)
    with pytest.raises(ValueError):prepare_coarse_packet(native,max_width=1920,now=100.)


@pytest.mark.parametrize('kind',['url','mismatched_size','wrong_mode','wrong_format','duplicate_camera','too_large'])
def test_invalid_or_unbounded_native_rgb_is_rejected(kind):
    native=make_packet(((640,480),(640,480)))
    if kind=='url':native['images'][0]['data_url']='https://not-permitted/image.jpg'
    if kind=='mismatched_size':native['images'][0]['width']=639
    if kind=='wrong_mode':native=make_packet(((640,480),(640,480)),mode='L')
    if kind=='wrong_format':native=make_packet(((640,480),(640,480)),fmt='PNG')
    if kind=='duplicate_camera':native['images'][1]['camera_id']='left'
    if kind=='too_large':native['images'][0]['width']=4097
    with pytest.raises(ValueError):prepare_coarse_packet(native,now=100.)


def test_model_byte_limit_is_not_silently_relaxed():
    native=make_packet(((640,480),(640,480)))
    with patch('embodied_arm.perception.rgb_coarse_packet.MAX_MODEL_PACKET_BYTES',100):
        with pytest.raises(ValueError):prepare_coarse_packet(native,now=100.)
