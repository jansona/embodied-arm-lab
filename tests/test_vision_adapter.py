"""Image-only mocked Qwen protocol; never calls a live provider or reads keys."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import time

import httpx
from PIL import Image
import pytest

from embodied_arm.providers.brain import MODEL, ENDPOINT
from embodied_arm.perception.camera_perception import PixelDetection
from embodied_arm.providers.vision_adapter import QwenVision, frame_snapshot, request_preview, validate_detections, sanitize_diagnostic, convert_provider_detections


def frame():
    out=BytesIO(); Image.new('RGB',(64,48),'red').save(out,format='JPEG')
    url='data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()
    return {'frame_id':'capture-1','captured_at':time.monotonic(), 'images':[
        {'camera_id':cid,'width':64,'height':48,'data_url':url} for cid in ('left','right')]}


def observations():
    return {'frame_id':'capture-1','detections':[
        {'object_id':obj,'camera_id':cid,'frame_id':'capture-1','uv':[30,20],
         'confidence':.95,'pixel_sigma_px':2,'visibility':'visible',
         'landmark':'goal_center' if obj=='goal' else 'top_face_center'}
        for obj in ('red','blue','goal') for cid in ('left','right')]}


def normalized(data):
    data=deepcopy(data);data['coordinate_space']='normalized_1000'
    for item in data.get('detections',[]):
        uv=item.pop('uv',None);sigma=item.pop('pixel_sigma_px',None)
        item['uv_1000']=([v*1000/d for v,d in zip(uv,(63,47))]
            if isinstance(uv,list) and len(uv)==2 and all(type(v) in (int,float) for v in uv) else uv)
        item['sigma_1000']=sigma*1000/63 if type(sigma) in (int,float) else sigma
    return data


def completion(data=None):
    return {'model':MODEL,'choices':[{'finish_reason':'stop','message':{
        'content':json.dumps(normalized(observations() if data is None else data))}}]}


def adapter(payload=None,status=200,**kwargs):
    return QwenVision(api_key='fake-test-key',transport=httpx.MockTransport(
        lambda _: httpx.Response(status,json=completion() if payload is None else payload)),**kwargs)


def test_sends_actual_rgb_two_images_only_no_ground_truth():
    packet=frame();packet.update(blocks={'red':[1,2,3]},depth='not allowed',segmentation='not allowed',
                                calibrations={'left':'local-only'},api_key='never-send')
    seen=[]
    def handle(req):
        seen.append(req)
        assert str(req.url)==ENDPOINT
        body=json.loads(req.content);assert body['model']==MODEL
        assert body['response_format']=={'type':'json_object'}
        assert body['enable_thinking'] is False and 'thinking_budget' not in body
        content=body['messages'][1]['content']
        images=[x for x in content if x['type']=='image_url']
        assert len(images)==2 and images[0]['image_url']['url'].startswith('data:image/jpeg;base64,')
        serialized=json.dumps(body)
        for forbidden in ('never-send','local-only','not allowed','blocks','segmentation'):
            assert forbidden not in serialized
        return httpx.Response(200,json=completion())
    brain=QwenVision(api_key='fake-test-key',transport=httpx.MockTransport(handle))
    result=brain.perceive(packet)
    assert result.accepted and result.calls==len(seen)==1
    assert all(isinstance(d,PixelDetection) for d in result.detections)
    assert result.provenance=='qwen_rgb' and result.frame_id==packet['frame_id']
    assert 'fake-test-key' not in str(result.to_dict())+str(brain.status())


@pytest.mark.parametrize('change',[
    {'uv':[-1,20]}, {'uv':[64,20]}, {'uv':[30,48]}, {'uv':[True,2]}, {'uv':[float('nan'),20]},
    {'object_id':'yellow'}, {'camera_id':'third'}, {'frame_id':'old'}, {'landmark':'hidden_center'},
    {'confidence':1.1}, {'confidence':True}, {'pixel_sigma_px':0}, {'pixel_sigma_px':26},
    {'visibility':True}, {'visibility':'occluded'}, {'position':[0,0,0]},
])
def test_invalid_detections_fail_closed(change):
    data=observations();data['detections'][0].update(change)
    result=adapter(completion(data)).perceive(frame())
    assert not result.accepted and not result.detections


@pytest.mark.parametrize('visibility',['occluded','absent','ambiguous','unknown'])
def test_nonvisible_feature_rejects_motion(visibility):
    data=observations();data['detections'][0].update(visibility=visibility,uv=None,pixel_sigma_px=None,confidence=0)
    result=adapter(completion(data)).perceive(frame())
    assert result.reason=='occluded_or_missing_feature' and not result.accepted


def test_low_confidence_duplicate_unknown_frame_and_missing_pairs():
    data=observations();data['detections'][0]['confidence']=.7
    assert adapter(completion(data)).perceive(frame()).reason=='low_confidence'
    data=observations();data['detections'][1]=deepcopy(data['detections'][0])
    assert adapter(completion(data)).perceive(frame()).reason=='invalid_response'
    data=observations();data['frame_id']='old'
    assert adapter(completion(data)).perceive(frame()).reason=='invalid_response'
    data=observations();data['detections'].pop()
    assert adapter(completion(data)).perceive(frame()).reason=='invalid_response'


@pytest.mark.parametrize('kind',['stale','future','url','mismatch','duplicate','invalid_jpeg','grayscale','oversized','too_many'])
def test_invalid_images_no_api_calls(kind):
    packet=frame()
    if kind=='stale':packet['captured_at']-=61
    if kind=='future':packet['captured_at']+=10
    if kind=='url':packet['images'][0]['data_url']='https://external.example/image.jpg'
    if kind=='mismatch':packet['images'][0]['width']=63
    if kind=='duplicate':packet['images'][1]['camera_id']='left'
    if kind=='invalid_jpeg':packet['images'][0]['data_url']='data:image/jpeg;base64,SGVsbG8='
    if kind=='grayscale':
        out=BytesIO();Image.new('L',(64,48)).save(out,format='JPEG')
        packet['images'][0]['data_url']='data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()
    if kind=='oversized':packet['images'][0]['width']=2048
    if kind=='too_many':packet['images'].append(packet['images'][0])
    brain=adapter();result=brain.perceive(packet)
    assert result.reason=='invalid_or_stale_frame' and brain.calls==0


@pytest.mark.parametrize('status',[301,401,403,404,429,500])
def test_http_errors_redacted_no_retry(status):
    result=adapter({'error':'secret'},status=status).perceive(frame())
    assert result.reason==f'http_error_{status}' and result.calls==1
    assert 'secret' not in str(result.to_dict())


def test_budget_and_large_response():
    brain=adapter({'large':'x'*66000},max_calls=1)
    assert brain.perceive(frame()).reason=='response_too_large'
    assert brain.perceive(frame()).reason=='call_budget_exhausted'


def test_no_substitute_model_and_truncation():
    data=completion();data['model']='qwen3-vl'
    assert adapter(data).perceive(frame()).reason=='unexpected_model'
    data=completion();data['choices'][0]['finish_reason']='length'
    assert adapter(data).perceive(frame()).reason=='incomplete_response'


def test_post_response_staleness(monkeypatch):
    import embodied_arm.providers.vision_adapter as vision_adapter
    packet=frame();clock=[packet['captured_at']]
    monkeypatch.setattr(vision_adapter.time,'monotonic',lambda:clock[0])
    def handle(_):
        clock[0]+=2
        return httpx.Response(200,json=completion())
    brain=QwenVision(api_key='fake',max_frame_age=1,timeout=5,transport=httpx.MockTransport(handle))
    assert brain.perceive(packet).reason=='stale_frame'


def test_strict_configuration():
    for kwargs in ({'model':'other'},{'endpoint':'https://evil'},{'timeout':True},
        {'max_calls':101},{'max_frame_age':121},{'min_confidence':.1},{'max_uncertainty_px':100}):
        with pytest.raises(ValueError):QwenVision(api_key='fake',**kwargs)


def test_diagnostic_identifies_validator_stage_without_weakening_gate():
    data=observations();data['detections'][0]['pixel_sigma_px']=1
    brain=adapter(completion(data));result=brain.perceive(frame())
    assert not result.accepted and result.reason=='invalid_response'
    diagnostic=brain.last_diagnostic
    assert diagnostic['stage']=='detections' and diagnostic['error']=='invalid_pixel_uncertainty'
    assert diagnostic['detection_count']==6
    assert diagnostic['content']['frame_match'] is True
    assert diagnostic['detections'][0]['sigma_1000']==pytest.approx(1000/63)
    assert diagnostic['detections'][0]['object_id']=='red'
    assert 'capture-1' not in json.dumps(diagnostic)


def test_diagnostic_drops_raw_provider_text_and_unknown_labels():
    data=observations();data['SECRET_UNKNOWN_FIELD']='SECRET_VALUE'
    data['detections'][0].update(object_id='SECRET_LABEL',camera_id='SECRET_CAMERA',
                                landmark='SECRET_LANDMARK',visibility='SECRET_VISIBILITY')
    response=completion(data);response['SECRET_HEADER']='SECRET_KEY'
    response['choices'][0]['message']['reasoning_content']='SECRET_REASONING'
    brain=adapter(response);assert not brain.perceive(frame()).accepted
    diagnostic=brain.last_diagnostic
    assert diagnostic['error']=='coordinate_contract'
    assert diagnostic['response']['unknown_key_count']==1
    assert diagnostic['content']['unknown_key_count']==1
    assert diagnostic['detections'][0]['object_id']=='unrecognized'
    assert 'SECRET' not in json.dumps(diagnostic)


def test_invalid_content_json_only_records_fixed_error_and_shape():
    response=completion();response['choices'][0]['message']['content']='SECRET not JSON'
    brain=adapter(response);assert not brain.perceive(frame()).accepted
    assert brain.last_diagnostic['stage']=='content_json'
    assert brain.last_diagnostic['error']=='invalid_json'
    assert 'SECRET' not in json.dumps(brain.last_diagnostic)


def test_diagnostic_sanitizer_rejects_unknown_strings_nonfinite_and_unbounded_values():
    raw={'stage':'SECRET_STAGE','error':'SECRET_ERROR','headers':{'Authorization':'SECRET_KEY'},
         'response':{'type':'SECRET_TYPE','known_keys':['model','SECRET_FIELD'],
                     'field_types':{'model':'string','SECRET_FIELD':'SECRET_TYPE'},'unknown_key_count':10000},
         'detections':[{'type':'object','object_id':'SECRET_OBJECT','camera_id':'left','frame_id':'SECRET_ID',
                        'uv':[float('nan'),2],'confidence':float('inf'),'pixel_sigma_px':1e99,
                        'frame_match':False,'unknown_key_count':1}]*20}
    safe=sanitize_diagnostic(raw);text=json.dumps(safe,allow_nan=False)
    assert 'SECRET' not in text and len(safe['detections'])==6
    assert safe['response']['unknown_key_count']==1000
    assert 'uv' not in safe['detections'][0] and 'confidence' not in safe['detections'][0]


def test_declared_normalized_contract_converts_edges_and_conservative_sigma():
    packet=frame();data=normalized(observations())
    data['detections'][0].update(uv_1000=[1000,1000],sigma_1000=50)
    canonical=convert_provider_detections(data,packet)
    first=canonical['detections'][0]
    assert first['uv']==[63,47]
    assert first['pixel_sigma_px']==pytest.approx(3.15)
    detections=validate_detections(canonical,packet)
    assert detections[0].uv==(63,47)
    assert 'uv_1000' not in detections[0].to_dict()


def test_undeclared_old_output_and_out_of_range_normalized_values_rejected():
    packet=frame()
    with pytest.raises(ValueError):convert_provider_detections(observations(),packet)
    data=normalized(observations());data['coordinate_space']='pixels'
    with pytest.raises(ValueError):convert_provider_detections(data,packet)
    for field,value in [('uv_1000',[-1,500]),('uv_1000',[1001,500]),('uv_1000',[True,500]),
                        ('sigma_1000',float('nan')),('sigma_1000',1001)]:
        data=normalized(observations());data['detections'][0][field]=value
        with pytest.raises(ValueError):convert_provider_detections(data,packet)
    response=completion();response['choices'][0]['message']['content']=json.dumps(observations())
    brain=adapter(response)
    assert not brain.perceive(packet).accepted
    assert brain.last_diagnostic['error']=='coordinate_contract'


def test_normalized_conversion_preserves_pixel_uncertainty_and_visibility_gates():
    packet=frame()
    for sigma in (1,500):
        data=normalized(observations());data['detections'][0]['sigma_1000']=sigma
        canonical=convert_provider_detections(data,packet)
        with pytest.raises(ValueError):validate_detections(canonical,packet)
    data=normalized(observations())
    data['detections'][0].update(visibility='occluded',uv_1000=None,sigma_1000=None,confidence=0)
    canonical=convert_provider_detections(data,packet)
    missing=validate_detections(canonical,packet)[0]
    assert missing.uv is None and missing.pixel_sigma_px is None and missing.visibility=='occluded'
