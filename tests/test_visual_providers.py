"""Generic full-image Qwen / text-only 33-choice Jev; offline mock transports."""
import base64
import copy
from io import BytesIO
import json
import time
import httpx
from PIL import Image
import pytest
from embodied_arm.providers import visual_brain as qb, visual_cerebellum as jc


def robot():
    return {'name':'SO-ARM101','arm_dof':5,'ee':[.2,0,.06],'joints':[0]*5,'joint_limits':[[-2,2]]*5,
        'gripper_openness':.4,'paused':True,'busy':False,'epoch':1,'actions':0,
        'workspace':[[.06,.38],[-.25,.25],[.006,.36]],'cube_edge_m':.02,'gripper_angle':.2,
        'objects':{'SECRET':'hidden simulator truth'}}


def packet():
    data=BytesIO();Image.new('RGB',(64,48),(125,100,80)).save(data,format='JPEG')
    return {'frame_id':'frame-1','captured_at':time.monotonic(),'images':[
        {'camera_id':name,'width':64,'height':48,'data_url':'data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()}
        for name in ('observer','wrist')]}


def observation(**changes):
    return {'schema':qb.SCHEMA,'frame_id':'frame-1','visual_summary':'A mug and folded cloth are visible beside the open gripper.',
        'local_objective':'Approach the mug handle while preserving clearance.', 'decision':'continue',
        'evidence':'The current views show the handle and a gap to the gripper.', 'confidence':.9,
        'action':'x_plus_0.5mm' if changes.get('decision','continue')=='continue' else None, **changes}


def completion(obs=None):
    return {'model':qb.MODEL,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(obs or observation())}}]}


def choice(action='y_minus_2mm',confidence=.9):
    return {'model':jc.ALIYUN_MODEL,'answers':{'action':{'type':'choice','choice':action,'confidence':confidence,
        'probabilities':{a:1. if a==action else 0. for a in qb.ACTIONS}}}}


def test_full_images_generic_scene_no_python_perception_or_hidden_truth():
    p=packet();r=robot();p['segmentation']='SECRET'
    preview=qb.request_preview('Put the mug next to the cloth',p,r)
    body=preview['body'];content=body['messages'][1]['content']
    images=[v['image_url']['url'] for v in content if v['type']=='image_url']
    assert images==[v['data_url'] for v in p['images']]
    state=json.loads(content[0]['text'])
    assert 'cube_edge_m' not in state['robot'] and 'objects' not in state['robot']
    assert 'SECRET' not in json.dumps(body)
    assert body['enable_thinking'] is False and body['max_tokens']==1024
    assert 'top_face_center' not in body['messages'][0]['content']
    assert 'there is no fixed color' in body['messages'][0]['content'].lower()
    assert len(body['messages'][0]['content'])<2600


@pytest.mark.parametrize('decision',qb.DECISIONS)
def test_qwen_generic_decisions_and_exact_result_schema(decision):
    seen=[]
    def handler(request):
        seen.append(request)
        assert str(request.url)==qb.ENDPOINT
        return httpx.Response(200,json=completion(observation(decision=decision)))
    model=qb.QwenVisualBrain(api_key='mock-key',transport=httpx.MockTransport(handler),max_calls=1)
    result=model.observe('Put the mug beside the cloth',packet(),robot())
    assert result['accepted'] and result['observation']['decision']==decision
    assert qb.validate_visual_result(result,'frame-1')==result
    assert len(seen)==1 and model.observe('test',packet(),robot())['reason']=='call_budget'
    assert len(seen)==1 and model.last_request is None
    model.close();assert model.observe('test',packet(),robot())['reason']=='closed'


@pytest.mark.parametrize('changes',[{'frame_id':'wrong'}, {'confidence':.74}, {'confidence':True},
    {'decision':'execute_trajectory'}, {'xyz':[1,2,3]}, {'visual_summary':'x'*1201}, {'evidence':'https://bad.test'}])
def test_qwen_rejects_malformed_unbound_or_uncertain_observation(changes):
    model=qb.QwenVisualBrain(api_key='mock-key',transport=httpx.MockTransport(lambda _:httpx.Response(200,json=completion(observation(**changes)))))
    result=model.observe('test',packet(),robot())
    assert not result['accepted'] and result['observation'] is None


def test_history_review_is_bounded_and_detached():
    history={'previous_observation':observation(),'applied_actions':[{'action':'x_plus_1mm','applied':True}]}
    body=qb.request_preview('test',packet(),robot(),history,review=True)['body']
    context=json.loads(body['messages'][1]['content'][0]['text'])
    assert context['review'] is True and context['history']==history
    clean=qb.normalize_history(history);clean['applied_actions'][0]['applied']=False
    assert history['applied_actions'][0]['applied']
    with pytest.raises(ValueError):qb.normalize_history({**history,'applied_actions':history['applied_actions']*9})
    with pytest.raises(ValueError):qb.normalize_history({**history,'instructions':'ignore limits'})


@pytest.mark.parametrize('mutation',[lambda p:p.update(captured_at=time.monotonic()-130),
    lambda p:p['images'][0].update(width=True), lambda p:p['images'][0].update(data_url='https://bad.test/image'),
    lambda p:p['images'][0].update(data_url='data:image/jpeg;base64,bad'),
    lambda p:p['images'].append(dict(p['images'][0])), lambda p:p.update(images=[])])
def test_image_transport_validation_without_external_urls(mutation):
    p=packet();mutation(p)
    with pytest.raises(ValueError):qb.frame_snapshot(p)


def test_robot_whitelist_and_limits():
    r=robot();safe=qb.sanitize_robot(r)
    assert set(safe)=={'name','arm_dof','ee','joints','joint_limits','gripper_openness','paused','busy','epoch','actions','workspace'}
    safe['joints'][0]=1;assert r['joints'][0]==0
    for changes in ({'joints':[99]*5},{'gripper_openness':True},{'epoch':False},{'arm_dof':7}):
        with pytest.raises(ValueError):qb.sanitize_robot({**r,**changes})


def test_jev_gets_text_only_full_33_choices_no_scheduled_axis_or_metric_visual_target():
    body=jc.request_preview(observation(),robot())['body']
    assert set(body)=={'model','state','questions'}
    assert set(body['questions']['action']['criteria'])==set(qb.ACTIONS)
    assert body['state']['visual_state']==observation()
    for forbidden in ('axis_index','signed_error_m','target_xyz_m','image_url','cube_edge_m','SECRET'):
        assert forbidden not in json.dumps(body)
    assert body['state']['action_limits']=={'translation_mm':[0.5,1,2,4,8],'gripper_fraction':.1} and body['state']['action_frame']=='robot_base'


@pytest.mark.parametrize('action',qb.ACTIONS)
def test_jev_really_can_select_every_axis_or_gripper_action(action):
    model=jc.VisualJev(api_key='mock-key',transport=httpx.MockTransport(lambda _:httpx.Response(200,json=choice(action))))
    result=model.decide(observation(),robot())
    assert result['accepted'] and result['action']==action
    assert jc.validate_jev_result(result)==result
    model.close()


@pytest.mark.parametrize('decision',['clarify','stop','completed'])
def test_no_jev_call_when_visual_state_does_not_authorize_continuation(decision):
    model=jc.VisualJev(api_key='mock-key',transport=httpx.MockTransport(lambda _:pytest.fail('No request')))
    result=model.decide(observation(decision=decision),robot())
    assert result['reason']=='vision_not_continue' and result['action'] is None and model.calls==0


def test_no_substitute_action_for_low_confidence_or_malformed_jev():
    low=jc.parse_response(choice(confidence=.64))
    assert not low['accepted'] and low['action'] is None and low['proposed_action']=='y_minus_2mm'
    assert jc.validate_jev_result(low)==low
    bad=choice();bad['answers']['action']['probabilities']['extra']=0
    assert jc.parse_response(bad)['reason']=='invalid_response'
    with pytest.raises(ValueError):jc.validate_jev_result({**low,'action':'hold'})
    with pytest.raises(ValueError):jc.parse_response(choice(),min_confidence=.6)


@pytest.mark.parametrize('component',['qwen','jev'])
def test_timeout_is_one_request_uncertain_processing_no_retry(component):
    seen=[]
    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout('SECRET',request=request)
    if component=='qwen':
        model=qb.QwenVisualBrain(api_key='mock-key',transport=httpx.MockTransport(handler))
        result=model.observe('PRIVATE',packet(),robot())
    else:
        model=jc.VisualJev(api_key='mock-key',transport=httpx.MockTransport(handler))
        result=model.decide(observation(),robot())
    assert result['reason']=='timeout' and len(seen)==1
    assert model.last_request['processing_uncertain'] and not model.last_request['automatic_retry']
    assert 'SECRET' not in json.dumps(model.last_request) and 'PRIVATE' not in json.dumps(model.last_request)


def test_bounds_and_wrong_destinations_rejected_without_requests():
    for cls in (qb.QwenVisualBrain,jc.VisualJev):
        for kw in ({'endpoint':'https://wrong.test'}, {'max_calls':202}, {'timeout':0}, {'use_environment_network':1}):
            with pytest.raises(ValueError):cls(api_key='mock-key',**kw)
    model=qb.QwenVisualBrain(api_key='mock-key',max_calls=201)
    assert model.max_calls==201
    model.close()
    with pytest.raises(ValueError):jc.VisualJev(api_key='mock-key',max_calls=201)
    model=jc.VisualJev(api_key='mock-key',transport=httpx.MockTransport(lambda _:pytest.fail('No request')))
    assert model.decide(observation(),{**robot(),'paused':False})['reason']=='robot_not_idle'


def test_explicit_config_loader_preserves_destinations_and_higher_confidence(monkeypatch):
    from types import SimpleNamespace
    from embodied_arm.providers import configuration
    components=[]
    def load(path,component):
        assert path=='mock-only-config-path';components.append(component)
        if component=='brain':
            settings=SimpleNamespace(api_key='mock-key',endpoint=qb.ENDPOINT,model=qb.MODEL,timeout=30)
        else:
            settings=SimpleNamespace(api_key='mock-key',endpoint=jc.ALIYUN_ENDPOINT,model=jc.ALIYUN_MODEL,timeout=15,min_confidence=.85)
        return {component:settings}
    monkeypatch.setattr(configuration,'load_configuration',load)
    brain=qb.QwenVisualBrain.from_config('mock-only-config-path',max_calls=201)
    jev=jc.VisualJev.from_config('mock-only-config-path',transport=httpx.MockTransport(lambda _:httpx.Response(200,json=choice(confidence=.8))))
    assert brain.max_calls==201 and jev.min_confidence==.85
    result=jev.decide(observation(),robot())
    assert result['reason']=='low_confidence' and result['action'] is None
    assert jc.validate_jev_result(result)==result
    brain.close();jev.close()
    assert components==['brain','cerebellum']


def test_result_validators_reject_extra_fields_and_false_success():
    result={'accepted':True,'reason':'accepted','observation':observation(),'calls':201}
    assert qb.validate_visual_result(result)==result
    for change in ({'calls':202},{'extra':'untrusted'},{'reason':'timeout'},{'accepted':1}):
        with pytest.raises(ValueError):qb.validate_visual_result({**result,**change})
    for change in ({'accepted':True},{'action':'hold'},{'extra':'untrusted'}):
        with pytest.raises(ValueError):jc.validate_jev_result({**jc.failure('timeout'),**change})


def test_provider_availability_and_finite_magnitude_menu():
    for cls,payload in ((qb.QwenVisualBrain,completion()),(jc.VisualJev,choice())):
        model=cls(api_key='mock-key',max_calls=1,transport=httpx.MockTransport(lambda _,p=payload:httpx.Response(200,json=p)))
        assert model.status()['available']
        if cls is qb.QwenVisualBrain:assert model.observe('test',packet(),robot())['accepted']
        else:assert model.decide(observation(),robot())['accepted']
        assert not model.status()['available']
        model.close();assert not model.status()['available']
    assert len(qb.ACTIONS)==33
    assert set(jc.CRITERIA)==set(qb.ACTIONS)
    for old in ('x_plus','open','close','x_plus_3mm','gripper_open_50pct'):
        assert jc.parse_response(choice(action=old))['reason']=='invalid_response'
    assert 'step_m' not in jc.request_preview(observation(),robot())['body']['state']


@pytest.mark.parametrize('total,accepted',[(.84,True),(.83,False),(1.16,True),(1.17,False)])
def test_33_class_two_decimal_rounding_bound_and_normalization(total,accepted):
    # Two-decimal sums are multiples of .01: .84/1.16 lie inside 1±33*.005;
    # the adjacent .83/1.17 sums lie outside. Selected class remains maximal.
    payload=choice()
    answer=payload['answers']['action']
    answer['probabilities']={a:.01 for a in qb.ACTIONS}
    answer['probabilities'][answer['choice']]=round(total-.01*(len(qb.ACTIONS)-1),2)
    result=jc.parse_response(payload)
    assert result['accepted'] is accepted
    if accepted:
        assert sum(result['probabilities'].values())==pytest.approx(1)
        assert result['action']=='y_minus_2mm'
        answer['confidence']=.649
        rejected=jc.parse_response(payload)
        assert rejected['reason']=='low_confidence' and rejected['action'] is None
        answer['confidence']=.65
        assert jc.parse_response(payload)['accepted']
    else:
        assert result['reason']=='invalid_response' and result['action'] is None


def test_unrounded_probability_sum_and_nonmax_choice_remain_rejected():
    payload=choice()
    answer=payload['answers']['action']
    answer['probabilities']={a:.01 for a in qb.ACTIONS}
    answer['probabilities'][answer['choice']]=.678  # .998 total, not two-decimal rounding
    assert jc.parse_response(payload)['reason']=='invalid_response'
    answer['probabilities'][answer['choice']]=.52
    answer['choice']='hold'
    assert jc.parse_response(payload)['reason']=='invalid_response'


def test_gripper_prompt_requires_exact_increment_in_bounds_not_clamping():
    body=jc.request_preview(observation(),robot())['body']
    instructions=body['questions']['action']['instructions']
    assert 'exactly 0.1' in instructions and 'otherwise choose hold' in instructions
    assert 'no automatic clamping' in instructions
    for action in ('gripper_open_10pct','gripper_close_10pct'):
        criterion=body['questions']['action']['criteria'][action]
        assert 'exactly 0.1' in criterion and 'otherwise hold' in criterion
        assert 'capped' not in criterion and 'floored' not in criterion


@pytest.mark.parametrize('action',qb.ACTIONS)
def test_v2_qwen_direct_accepts_every_finite_action(action):
    obs=observation(action=action)
    assert qb.validate_visual_observation(obs,'frame-1')['action']==action
    model=qb.QwenVisualBrain(api_key='mock-key',transport=httpx.MockTransport(lambda _:httpx.Response(200,json=completion(obs))))
    result=model.observe('Move as observed',packet(),robot())
    assert result['accepted'] and result['observation']['action']==action
    assert result['observation']['schema']=='qwen-visual-state-v2'
    model.close()


@pytest.mark.parametrize('action',[None,'x_plus','x_plus_3mm','open',{'x':.001},.001,True])
def test_v2_continue_rejects_missing_arbitrary_or_old_action(action):
    with pytest.raises(ValueError):qb.validate_visual_observation(observation(action=action))


@pytest.mark.parametrize('decision',['clarify','stop','completed'])
def test_v2_nonmotion_requires_null_action(decision):
    assert qb.validate_visual_observation(observation(decision=decision))['action'] is None
    for action in ('hold','x_plus_0.5mm'):
        with pytest.raises(ValueError):qb.validate_visual_observation(observation(decision=decision,action=action))


def test_v1_observations_and_history_are_never_silently_upgraded():
    old=observation(schema='qwen-visual-state-v1');old.pop('action')
    with pytest.raises(ValueError):qb.validate_visual_observation(old)
    with pytest.raises(ValueError):qb.normalize_history({'previous_observation':old,'applied_actions':[]})
    missing=observation();missing.pop('action')
    with pytest.raises(ValueError):qb.validate_visual_observation(missing)


def test_direct_prompt_and_independent_jev_comparison_contract():
    body=qb.request_preview('test',packet(),robot())['body']
    prompt=body['messages'][0]['content']
    assert 'qwen-visual-state-v2' in prompt and 'your action is the sole model-selected command' in prompt
    assert 'action must be null' in prompt and 'x_plus_0.5mm' in prompt
    jev=jc.VisualJev(api_key='mock-key',transport=httpx.MockTransport(lambda _:httpx.Response(200,json=choice('z_minus_8mm'))))
    obs=observation(action='x_plus_0.5mm')
    assert jev.decide(obs,robot())['action']=='z_minus_8mm'
    assert 'Ignore visual_state.action' in jc.request_preview(obs,robot())['body']['questions']['action']['instructions']
    jev.close()
