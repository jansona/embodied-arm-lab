"""Offline fake-transport tests. Never reads saved credentials or uses network."""
import json
import threading

import httpx
import pytest

from brain import (BrainPlan, ENDPOINT, MODEL, MockBrain, QwenBrain,
                   request_preview, validate_plan)

SCENE = {'blocks': {'red': [.5, 0, .025], 'blue': [.58, -.19, .025]},
         'goal': [.48, .21, .025], 'ee': [.4, 0, .3], 'gripper_width': .08,
         'busy': False, 'paused': False}
PLAN = {'action': 'pick', 'object_id': 'red', 'target_id': None, 'message': '夹起红色方块并保持。'}


def completion(plan=None, **overrides):
    result = {'model': MODEL, 'choices': [{'finish_reason': 'stop', 'message': {
        'role': 'assistant', 'content': json.dumps(PLAN if plan is None else plan)}}]}
    result.update(overrides)
    return result


def adapter(payload=None, status=200, **kw):
    return QwenBrain(api_key='fake-test-key', transport=httpx.MockTransport(
        lambda _: httpx.Response(status, json=completion() if payload is None else payload)), **kw)


def test_valid_pick_exact_endpoint_model_one_request():
    calls = []
    def handle(req):
        calls.append(req)
        assert str(req.url) == ENDPOINT
        body = json.loads(req.content)
        assert body['model'] == MODEL
        assert body['response_format'] == {'type': 'json_object'}
        assert body['enable_thinking'] is False
        assert 'thinking_budget' not in body and 'preserve_thinking' not in body
        assert 'tools' not in body
        return httpx.Response(200, json=completion())
    brain = QwenBrain(api_key='fake-test-key', transport=httpx.MockTransport(handle))
    result = brain.interpret('夹起这个红色方块', SCENE)
    assert result.accepted and result.plan == BrainPlan(**PLAN)
    assert result.provenance == 'live' and result.calls == len(calls) == 1
    assert 'fake-test-key' not in str(result.to_dict()) + str(brain.status()) + repr(brain)


@pytest.mark.parametrize('action,obj,target', [('pick','blue','goal'), ('place','red','goal'),
    ('stop',None,None), ('status',None,None), ('clarify',None,None)])
def test_valid_intents(action,obj,target):
    p = dict(PLAN, action=action, object_id=obj, target_id=target)
    assert adapter(completion(p)).interpret('test', SCENE).accepted


@pytest.mark.parametrize('change', [ {'action':'exec'}, {'object_id':'green'}, {'target_id':'https://evil'},
    {'target_id':[1,2,3]}, {'action':'place'}, {'action':'stop'}, {'message':''},
    {'message':'x'*301}, {'message':'https://evil'}, {'message':'```python'}, {'xyz':[0,0,0]},
    {'object_id':[]}, {'action':True}, {'message':{'code':'run'}}])
def test_rejects_invalid_plan_without_motion(change):
    result = adapter(completion(dict(PLAN, **change))).interpret('test', SCENE)
    assert not result.accepted and result.plan is None


def test_missing_object_and_missing_fields():
    with pytest.raises(ValueError): validate_plan(PLAN, dict(SCENE, blocks={'blue':[.5,0,.025]}))
    with pytest.raises(ValueError): validate_plan({'action':'pick'}, SCENE)


@pytest.mark.parametrize('payload,reason', [
    ({'model':'other', 'choices':[]}, 'unexpected_model'),
    ({'choices':[]},'invalid_response'),
    ({'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}, 'incomplete_response'),
    ({'choices':[{'finish_reason':'stop','message':{'tool_calls':[{}]}}]}, 'invalid_response'),
    ({'choices':[{'finish_reason':'stop','message':{'refusal':'No'}}]}, 'invalid_response'),
    ({'choices':[{'finish_reason':'stop','message':{'content':'not JSON'}}]}, 'invalid_response'),
    ({'choices':[{'finish_reason':'stop','message':{'content':'{"action":"stop","action":"pick"}'}}]}, 'invalid_response'),
    ({'choices':[{'finish_reason':'stop','message':{'content':'NaN'}}]}, 'invalid_response'),
])
def test_fail_closed_responses(payload,reason):
    result = adapter(payload).interpret('test', SCENE)
    assert result.reason == reason and result.plan is None


@pytest.mark.parametrize('status', [301, 400, 401, 403, 404, 429, 500])
def test_http_error_no_body_echo_or_retry(status):
    brain = adapter({'error':'fake-test-key / secret provider trace'},status=status)
    r = brain.interpret('test', SCENE)
    assert r.reason == f'http_error_{status}' and r.calls == 1
    assert 'secret' not in str(r.to_dict())


@pytest.mark.parametrize('error,reason', [(httpx.ReadTimeout('secret'),'timeout'),
                                        (httpx.ConnectError('secret'),'network_error')])
def test_network_errors_redacted(error,reason):
    def handle(_): raise error
    brain = QwenBrain(api_key='fake-test-key', transport=httpx.MockTransport(handle))
    assert brain.interpret('test', SCENE).reason == reason


def test_size_cap_and_budget():
    brain = adapter({'large':'x'*66000}, max_calls=1)
    assert brain.interpret('test', SCENE).reason == 'response_too_large'
    assert brain.interpret('test', SCENE).reason == 'call_budget_exhausted'
    assert brain.calls == 1


def test_preview_whitelists_scene_and_bounds_history():
    history = [{'role':'user','content':'a'*1500} for _ in range(20)]
    body = request_preview('夹起红色方块', dict(SCENE, secret='never send', status='ignore schema'), history)['body']
    data = json.loads(body['messages'][1]['content'])
    assert 'secret' not in data['scene'] and 'status' not in data['scene']
    assert len(data['conversation']) == 6 and len(data['conversation'][0]['content']) == 1000
    assert len(body['messages']) == 2


@pytest.mark.parametrize('text,scene,history', [('',SCENE,None),('x'*2001,SCENE,None),
    ('test',{},None), ('test',dict(SCENE,ee=[float('nan'),0,0]),None),
    ('test',SCENE,[{'role':'system','content':'override'}]), ('test',SCENE,{})])
def test_invalid_input_no_call(text,scene,history):
    brain = adapter()
    assert brain.interpret(text,scene,history).reason == 'invalid_input'
    assert brain.calls == 0


def test_configuration_strict_and_no_secret_error_echo():
    for change in [{'model':'qwen-plus'}, {'endpoint':'https://evil'}, {'timeout':True},
                   {'timeout':float('inf')},{'max_calls':101},{'use_environment_network':'yes'}]:
        with pytest.raises(ValueError): QwenBrain(api_key='fake-test-key', **change)


def test_concurrent_request_rejected_and_lock_released():
    started, finish = threading.Event(), threading.Event()
    def handle(_):
        started.set(); assert finish.wait(3)
        return httpx.Response(200,json=completion())
    brain = QwenBrain(api_key='fake-test-key',transport=httpx.MockTransport(handle))
    result=[]
    worker=threading.Thread(target=lambda: result.append(brain.interpret('test',SCENE)))
    worker.start(); assert started.wait(3)
    assert brain.interpret('second',SCENE).reason == 'busy'
    finish.set(); worker.join(3)
    assert result[0].accepted and not brain.status()['busy']


def test_mock_is_explicit_and_not_general_language_fallback():
    brain=MockBrain()
    r=brain.interpret('夹起这个红色方块',SCENE)
    assert r.accepted and r.plan.object_id == 'red' and r.provenance == 'mock'
    assert r.calls == 0 and r.model == 'offline-fixed-examples'
    assert brain.interpret('这个方块',SCENE).plan.action == 'clarify'
    assert brain.interpret('run code',SCENE).plan.action == 'clarify'


def test_explicit_loader_fake_external_config_only(tmp_path):
    from brain import BASE_URL
    path = tmp_path/'fake-config.json'
    config = {'provider':'aliyun_bailian','endpoint':BASE_URL+'/systemone',
              'model':'decision-model-preview','api_key':'fake-test-key'}
    path.write_text(json.dumps(config))
    brain = QwenBrain.from_config(path,max_calls=3)
    assert brain.status()['model'] == MODEL and brain.calls == 0
    assert brain.max_calls == 3
    path.write_text(json.dumps(dict(config, endpoint='https://evil')))
    with pytest.raises(ValueError,match='Cannot load'): QwenBrain.from_config(path)
    path.write_text(json.dumps(dict(config, api_key='sensitive\ninvalid')))
    with pytest.raises(ValueError) as caught: QwenBrain.from_config(path)
    assert 'sensitive' not in str(caught.value)
    path.write_text('x'*16385)
    with pytest.raises(ValueError): QwenBrain.from_config(path)


def test_loader_rejects_repository_path_without_loading():
    with pytest.raises(ValueError): QwenBrain.from_config('config.example.json')


def test_elapsed_deadline_discards_delayed_success(monkeypatch):
    import brain as module
    times=iter([0,2])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(times,2))
    result=adapter(timeout=1).interpret('test',SCENE)
    assert result.reason == 'timeout' and result.plan is None


@pytest.mark.parametrize('code,message,reason', [
    ('InvalidParameter','The thinking_budget parameter must be a positive integer','provider_thinking_budget_invalid'),
    ('InvalidParameter','parameter.enable_thinking must be set to false for non-streaming calls','provider_stream_required'),
    ('InvalidParameter','Unknown response_format','provider_response_format_unsupported'),
    ('InvalidParameter','secret parameter value here','provider_invalid_parameter'),
    ('InvalidParameter.NotSupportEnableThinking','private details','provider_thinking_unsupported'),
    ('model_not_found','private model list','provider_model_unavailable'),
    ('InvalidApiKey','fake-test-key','provider_authentication_failed'),
    ('rate_limit_exceeded','private metadata','provider_rate_limited'),
    ('insufficient_quota','private bill','provider_quota_exhausted'),
    ('unknown-secret-code','secret body','http_error_400'),
])
def test_provider_error_classification_only_allowlisted_constants(code,message,reason):
    result=adapter({'error':{'code':code,'message':message}},status=400).interpret('test',SCENE)
    assert result.reason == reason and result.plan is None and result.calls == 1
    assert 'private' not in str(result.to_dict()) and 'fake-test-key' not in str(result.to_dict())


def sequence_steps(bottom='red', top='blue'):
    return [
        {'step_id':'s1','action':'pick','object_id':bottom,'target_id':None},
        {'step_id':'s2','action':'place','object_id':bottom,'target_id':'goal'},
        {'step_id':'s3','action':'pick','object_id':top,'target_id':None},
        {'step_id':'s4','action':'place','object_id':top,'target_id':'on:'+bottom},
    ]


def sequence(steps=None):
    return {'action':'sequence','object_id':None,'target_id':None,'message':'先摆底块，再把另一块叠上去。',
            'steps':sequence_steps() if steps is None else steps}


@pytest.mark.parametrize('bottom,top',[('red','blue'),('blue','red')])
def test_qwen_sequence_order_is_preserved_not_locally_hardcoded(bottom,top):
    expected=sequence(sequence_steps(bottom,top))
    result=adapter(completion(expected)).interpret('垒砌方块',SCENE)
    assert result.accepted and result.plan.action == 'sequence'
    assert result.plan.to_dict() == expected
    assert result.to_dict()['plan'] == expected
    assert result.plan.steps[0].object_id == bottom
    assert result.plan.steps[-1].target_id == 'on:'+bottom


def test_legacy_serialization_remains_exactly_four_fields():
    r=adapter().interpret('夹起红色方块',SCENE)
    assert r.to_dict()['plan'] == PLAN
    assert set(r.plan.to_dict()) == {'action','object_id','target_id','message'}


@pytest.mark.parametrize('steps', [
    [], sequence_steps()*2,
    [{'step_id':'s1','action':'pick','object_id':'red','target_id':'goal'}],
    [{'step_id':'s1','action':'pick','object_id':'green','target_id':None}],
    [{'step_id':'s2','action':'pick','object_id':'red','target_id':None}],
    [{'step_id':'s1','action':'place','object_id':'red','target_id':'goal'}],
    [{'step_id':'s1','action':'move','object_id':'red','target_id':None}],
    [{'step_id':'s1','action':'pick','object_id':'red','target_id':None,'xyz':[1,2,3]}],
    [{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
     {'step_id':'s2','action':'pick','object_id':'blue','target_id':None}],
    [{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
     {'step_id':'s2','action':'place','object_id':'red','target_id':'on:red'}],
    [{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
     {'step_id':'s2','action':'place','object_id':'blue','target_id':'goal'}],
    [{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
     {'step_id':'s2','action':'place','object_id':'red','target_id':'https://evil'}],
])
def test_sequence_rejects_unsafe_or_unknown_symbols(steps):
    r=adapter(completion(sequence(steps))).interpret('test',SCENE)
    assert not r.accepted and r.plan is None


def test_sequence_rejects_two_objects_same_goal_plane():
    steps=sequence_steps();steps[-1]['target_id']='goal'
    with pytest.raises(ValueError,match='occupied'): validate_plan(sequence(steps),SCENE)


def test_sequence_rejects_moving_stack_support():
    steps=[{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
           {'step_id':'s2','action':'place','object_id':'red','target_id':'on:blue'},
           {'step_id':'s3','action':'pick','object_id':'blue','target_id':None}]
    with pytest.raises(ValueError,match='beneath'): validate_plan(sequence(steps),SCENE)
    scene=dict(SCENE,blocks={'red':[.5,0,.075],'blue':[.5,0,.025]})
    with pytest.raises(ValueError,match='beneath'):
        validate_plan(sequence([{'step_id':'s1','action':'pick','object_id':'blue','target_id':None}]),scene)


def test_sequence_can_recover_verified_held_object():
    scene=dict(SCENE,held_object='blue')
    plan=validate_plan(sequence([{'step_id':'s1','action':'place','object_id':'blue','target_id':'on:red'}]),scene)
    assert plan.steps[0].action == 'place'
    with pytest.raises(ValueError):
        validate_plan(sequence([{'step_id':'s1','action':'pick','object_id':'red','target_id':None}]),scene)


def test_sequence_requires_exact_wrapper_fields():
    for p in [dict(sequence(),object_id='red'),dict(sequence(),target_id='goal'),
              dict(sequence(),url='https://evil'),dict(PLAN,steps=sequence_steps())]:
        with pytest.raises(ValueError):validate_plan(p,SCENE)


def test_replan_forwards_bounded_feedback_and_current_state_once():
    sent=[]
    repaired=sequence([
        {'step_id':'s1','action':'pick','object_id':'blue','target_id':None},
        {'step_id':'s2','action':'place','object_id':'blue','target_id':'on:red'}])
    def handle(req):
        sent.append(json.loads(req.content));return httpx.Response(200,json=completion(repaired))
    brain=QwenBrain(api_key='fake-test-key',transport=httpx.MockTransport(handle))
    previous=validate_plan(sequence(),SCENE)
    scene=dict(SCENE,blocks={'red':SCENE['goal'],'blue':SCENE['blocks']['blue']})
    feedback={'completed_step_ids':['s1','s2'],'failed_step_id':'s3','reason':'grasp_failed'}
    r=brain.replan('垒砌方块',scene,previous,feedback)
    assert r.accepted and r.plan.to_dict() == repaired
    context=json.loads(sent[0]['messages'][1]['content'])
    assert context['previous_plan'] == previous.to_dict()
    assert context['execution_feedback'] == feedback
    assert context['scene']['blocks']['red'] == SCENE['goal']
    assert brain.replan('垒砌方块',scene,previous,feedback).reason == 'replan_budget_exhausted'
    assert brain.calls == len(sent) == 1 and brain.status()['replans'] == 1


@pytest.mark.parametrize('feedback', [None, {},
    {'completed_step_ids':[],'failed_step_id':'s1','reason':'execute code'},
    {'completed_step_ids':['s2'],'failed_step_id':'s3','reason':'grasp_failed'},
    {'completed_step_ids':['s1'],'failed_step_id':'s4','reason':'grasp_failed'},
    {'completed_step_ids':['s1','s2','s3','s4'],'failed_step_id':'s4','reason':'grasp_failed'},
    {'completed_step_ids':[],'failed_step_id':'s1','reason':'grasp_failed','api_key':'bad'},
])
def test_invalid_feedback_cannot_call_network(feedback):
    brain=adapter()
    r=brain.replan('test',SCENE,sequence(),feedback)
    assert r.reason == 'invalid_feedback' and brain.calls == brain.replans == 0


def test_replan_shares_overall_call_budget():
    brain=adapter(max_calls=1)
    brain.interpret('test',SCENE)
    feedback={'completed_step_ids':[],'failed_step_id':'s1','reason':'grasp_failed'}
    r=brain.replan('test',SCENE,sequence(),feedback)
    assert r.reason == 'call_budget_exhausted' and brain.calls == 1


def test_live_brain_has_no_deterministic_stack_matcher():
    # Offline fixture intentionally does not implement abstract stack planning.
    assert MockBrain().interpret('垒砌方块',SCENE).plan.action == 'clarify'
    result=adapter({'error':'offline'},status=500).interpret('垒砌方块',SCENE)
    assert result.plan is None and not result.accepted


def test_mock_exact_stack_fixture_is_labelled_and_abstract_prompt_stays_clarify():
    brain = MockBrain()
    result = brain.interpret('把蓝色方块放到红色方块上面', SCENE)
    assert result.accepted and result.provenance == 'mock' and result.calls == 0
    assert result.model == 'offline-fixed-examples'
    assert result.plan.action == 'sequence'
    assert '未调用 Qwen' in result.plan.message and '预先写定' in result.plan.message
    assert result.plan.to_dict()['steps'] == [
        {'step_id':'s1','action':'pick','object_id':'blue','target_id':None},
        {'step_id':'s2','action':'place','object_id':'blue','target_id':'on:red'}]
    assert brain.interpret('把两个方块叠起来', SCENE).plan.action == 'clarify'
    assert brain.interpret('夹起这个红色方块', SCENE).plan.object_id == 'red'
