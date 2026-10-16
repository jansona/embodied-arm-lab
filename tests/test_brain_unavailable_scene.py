import pytest
from brain import scene_snapshot, request_preview, validate_plan
from brain_ipc import normalize_input
SCENE={'observation_status':'unavailable','blocks':{}}
def test_explicit_unknown_scene_without_fake_coordinates():
    assert scene_snapshot(SCENE)==SCENE
    assert normalize_input('你好',SCENE)['scene']==SCENE
    p=request_preview('你能做什么',SCENE,[{'role':'user','content':'红色方块'}])
    assert '没有有效视觉观察' in p['body']['messages'][0]['content']
    assert '"ee"' not in p['body']['messages'][1]['content']
@pytest.mark.parametrize('scene',[{}, {'blocks':{}},{**SCENE,'ee':[0,0,0]},{**SCENE,'goal':[0,0,0]}])
def test_no_implicit_or_augmented_unknown(scene):
    with pytest.raises(ValueError):scene_snapshot(scene)
@pytest.mark.parametrize('action',['clarify','status','stop'])
def test_only_conversation_actions(action):
    assert validate_plan(dict(action=action,object_id=None,target_id=None,message='需要有效图像'),SCENE).action==action
@pytest.mark.parametrize('action',['pick','place','sequence'])
def test_unknown_scene_rejects_motion_even_stateless(action):
    p=dict(action=action,object_id='red',target_id=None,message='尝试')
    if action=='sequence':p.update(object_id=None,steps=[dict(step_id='s1',action='pick',object_id='red',target_id=None)])
    with pytest.raises(ValueError):validate_plan(p,SCENE,check_state=False)
