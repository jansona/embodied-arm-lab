"""Offline prompt/wire contracts; no credentials or live model requests."""
import copy
import json
import httpx
import pytest
from embodied_arm.providers import brain, so101_jev
from embodied_arm.runtime.brain_ipc import normalize_input

SCENE = {'blocks': {'red': [.2, 0, .01], 'blue': [.3, .1, .01]},
         'goal': [.25, -.1, .01], 'ee': [.2, 0, .1], 'cube_edge_m': .02,
         'perception_source': 'rgb_only'}
CAP = {'robot_name': 'SO-ARM101', 'execution_requested': True, 'experimental': True,
       'supported_actions': ['pick', 'place', 'sequence'], 'supported_targets': ['goal'],
       'controller': 'jev', 'completion_verification': 'temporal_rgb'}
PLAN = {'action': 'pick', 'object_id': 'red', 'target_id': None, 'message': '尝试抓取红块'}
CYCLE = {'cycle_index': 1, 'completed_micro_actions': 16, 'remaining_global_calls': 184,
         'held_object': None, 'held_verified': False, 'phase': 'descend',
         'residual_xyz_m': [.01, 0, -.02], 'stop_reason': 'cycle_budget',
         'observation_status': 'fresh', 'recent_actions': [{'action': 'z_minus', 'applied': True}]}
FEEDBACK = {'completed_step_ids': [], 'failed_step_id': 's1', 'reason': 'cycle_budget', 'cycle_context': CYCLE}


def test_prompt_separates_roles_and_retains_bounded_output():
    body = brain.request_preview('拿红块', {**SCENE, 'runtime_capabilities': CAP}, previous_plan=PLAN, feedback=FEEDBACK)['body']
    prompt = body['messages'][0]['content']
    for phrase in ('Qwen大脑', 'Jev小脑', '不随每帧调用', '不重置总预算', '不重复已完成步骤', '不输出推理过程'):
        assert phrase in prompt
    assert body['max_tokens'] == 1024 and body['enable_thinking'] is False
    assert len(prompt) < 3500
    context = json.loads(body['messages'][1]['content'])
    assert context['scene']['runtime_capabilities'] == CAP
    assert context['scene']['perception_source'] == 'rgb_only'
    assert context['execution_feedback']['cycle_context'] == CYCLE


def test_context_round_trips_ipc_without_arbitrary_fields():
    scene = {**SCENE, 'runtime_capabilities': CAP, 'perception': {'untrusted': 'ignored'}}
    normalized = normalize_input('拿红块', scene, previous_plan=PLAN, feedback=FEEDBACK)
    assert normalized['feedback']['cycle_context'] == CYCLE
    assert 'perception' not in normalized['scene']
    normalized['scene']['runtime_capabilities']['supported_actions'].clear()
    assert CAP['supported_actions']


@pytest.mark.parametrize('change', [{'unexpected': 'instruction'}, {'controller': 'other'},
    {'execution_requested': 1}, {'supported_actions': ['pick', 'pick']}, {'supported_targets': ['coordinates']}])
def test_invalid_capabilities_rejected(change):
    with pytest.raises(ValueError): brain.scene_snapshot({**SCENE, 'runtime_capabilities': {**CAP, **change}})


def test_unknown_scene_remains_conversation_only_with_capabilities():
    scene = {'observation_status': 'unavailable', 'blocks': {}, 'runtime_capabilities': CAP}
    assert brain.scene_snapshot(scene) == scene
    assert normalize_input('你好', scene)['scene'] == scene
    with pytest.raises(ValueError): brain.validate_plan(PLAN, scene)
    with pytest.raises(ValueError): brain.scene_snapshot({**scene, 'ee': [0, 0, 0]})


def test_runtime_rejects_unsupported_goals_and_shorthand():
    scene = {**SCENE, 'runtime_capabilities': CAP}
    with pytest.raises(ValueError): brain.validate_plan({**PLAN, 'target_id': 'goal'}, scene)
    sequence = {'action': 'sequence', 'object_id': None, 'target_id': None, 'message': '计划', 'steps': [
        {'step_id': 's1', 'action': 'pick', 'object_id': 'red', 'target_id': None},
        {'step_id': 's2', 'action': 'place', 'object_id': 'red', 'target_id': 'on:blue'}]}
    with pytest.raises(ValueError): brain.validate_plan(sequence, scene)
    assert brain.validate_plan(sequence, {**scene, 'runtime_capabilities': {**CAP, 'execution_requested': False}})
    assert brain.validate_plan({**PLAN, 'target_id': 'goal'}, SCENE)


@pytest.mark.parametrize('change', [{'held_object': 'red'}, {'remaining_global_calls': 201},
    {'cycle_index': True}, {'phase': 'invented'}, {'residual_xyz_m': [float('nan'), 0, 0]},
    {'recent_actions': [{'action': 'close', 'applied': True}] * 9},
    {'recent_actions': [{'action': 'close', 'applied': True, 'text': 'extra'}]}, {'extra': 'no'}])
def test_cycle_context_rejects_unbounded_or_unverified_feedback(change):
    with pytest.raises(ValueError): brain.validate_cycle_context({**CYCLE, **change})


def test_cycle_feedback_keeps_prefix_guard():
    with pytest.raises(ValueError): brain.validate_feedback(PLAN, {**FEEDBACK, 'completed_step_ids': ['s1']}, SCENE)
    verified = {**CYCLE, 'held_object': 'red', 'held_verified': True}
    assert brain.validate_cycle_context(verified) == verified


def test_small_cube_support_cannot_be_picked_from_under_top_block():
    scene = copy.deepcopy(SCENE)
    scene['blocks']['blue'] = [.2, 0, .03]
    plan = {'action': 'sequence', 'object_id': None, 'target_id': None, 'message': '计划',
            'steps': [{'step_id': 's1', **{k: PLAN[k] for k in ('action', 'object_id', 'target_id')}}]}
    with pytest.raises(ValueError): brain.validate_plan(plan, scene)
    for invalid in (.03, True, '0.02'):
        with pytest.raises(ValueError): brain.scene_snapshot({**SCENE, 'cube_edge_m': invalid})


def test_configurable_replans_share_global_calls():
    def handler(_):
        return httpx.Response(200, json={'model': brain.MODEL, 'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(PLAN)}}]})
    model = brain.QwenBrain(api_key='offline-placeholder', max_calls=3, max_replans=2, transport=httpx.MockTransport(handler))
    assert model.interpret('拿红块', SCENE).accepted
    assert model.replan('拿红块', SCENE, PLAN, FEEDBACK).accepted
    assert model.replan('拿红块', SCENE, PLAN, FEEDBACK).accepted
    assert model.replan('拿红块', SCENE, PLAN, FEEDBACK).reason == 'replan_budget_exhausted'
    assert model.calls == 3 and model.status()['max_replans'] == 2
    for invalid in (-1, 9, True):
        with pytest.raises(ValueError): brain.QwenBrain(api_key='offline-placeholder', max_replans=invalid)


def test_jev_role_uses_systemone_question_not_chat_or_new_state_fields():
    robot = {'name': 'SO-ARM101', 'arm_dof': 5, 'ee': [.2, 0, .06], 'joints': [0]*5,
        'joint_limits': [[-2, 2]]*5, 'gripper_openness': .4, 'gripper_angle': .2,
        'paused': False, 'busy': False, 'epoch': 1, 'actions': 0,
        'workspace': [[.06, .38], [-.25, .25], [.006, .36]], 'cube_edge_m': .02}
    for target, axis, choices in [({'source': 'rgb_only', 'kind': 'cartesian', 'target_xyz': [.22, 0, .06],
            'tolerance_m': .005, 'step_m': .01}, 0, {'positive', 'negative', 'hold'}),
            ({'source': 'rgb_only', 'kind': 'gripper', 'target_openness': 1, 'tolerance': .02}, None, {'open', 'close', 'hold'})]:
        body = so101_jev.request_preview(so101_jev.build_observation(target, robot, axis))['body']
        assert set(body) == {'model', 'state', 'questions'}
        question = body['questions']['action']
        assert set(question) == {'type', 'instructions', 'criteria'}
        assert set(question['criteria']) == choices
        for phrase in ('cerebellum', 'one frequent fresh-state microdecision', 'confidence below .65', 'cannot reset global limits'):
            assert phrase in question['instructions']
        assert len(question['instructions']) < 2200
        assert 'runtime_capabilities' not in body['state'] and 'cycle_context' not in body['state']


def test_singleton_place_can_confirm_paused_stack_without_fabricated_holding():
    scene = {**SCENE, 'runtime_capabilities': {**CAP, 'supported_targets': ['goal', 'on:blue']}}
    plan = {'action': 'place', 'object_id': 'red', 'target_id': 'on:blue', 'message': '继续原放置检查点'}
    assert 'held_object' not in scene
    assert brain.validate_plan(plan, scene).target_id == 'on:blue'
    context = {**CYCLE, 'phase': 'release', 'held_verified': False, 'held_object': None}
    previous, feedback = brain.validate_feedback(plan, {**FEEDBACK, 'cycle_context': context}, scene)
    assert previous == plan and feedback['cycle_context']['held_object'] is None
    prompt = brain.request_preview('继续', scene, previous_plan=plan, feedback=feedback)['body']['messages'][0]['content']
    assert '继续执行器内部检查点' in prompt and '不要编造held_object' in prompt
    for target in ('on:red', 'on:unknown', None, 'on:blue:extra'):
        with pytest.raises(ValueError): brain.validate_plan({**plan, 'target_id': target}, scene)
    with pytest.raises(ValueError): brain.validate_plan(plan, {**SCENE, 'runtime_capabilities': CAP})
    with pytest.raises(ValueError): brain.validate_plan({**PLAN, 'target_id': 'on:blue'}, scene)


PROGRESS = {'phase': 'descend', 'goal_start_xyz_m': [.2, 0, .01], 'goal_end_xyz_m': [.2, 0, .01],
    'error_start_xyz_m': [.01, 0, -.04], 'error_end_xyz_m': [.005, 0, -.02],
    'object_start_xyz_m': [.2, 0, .01], 'object_end_xyz_m': [.2, 0, .01], 'observations': 5}
ROBOT_STATE = {'joints': [0, .1, -.1, .2, 0], 'gripper_openness': .4, 'holding_status': 'unknown'}


def test_progress_and_current_robot_state_survive_ipc_and_prompt():
    scene = {**SCENE, 'robot_state': ROBOT_STATE}
    feedback = {**FEEDBACK, 'cycle_context': {**CYCLE, 'progress': [PROGRESS]}}
    normalized = normalize_input('继续', scene, previous_plan=PLAN, feedback=feedback)
    assert normalized['scene']['robot_state'] == ROBOT_STATE
    assert normalized['feedback']['cycle_context']['progress'] == [PROGRESS]
    preview = brain.request_preview('继续', scene, previous_plan=PLAN, feedback=feedback)
    prompt = preview['body']['messages'][0]['content']
    for phrase in ('由你判断实际进展', '同一phase、同一局部目标', '目标移动与机械臂进展', '残差缩小不等于', 'Stop后这些是当前状态'):
        assert phrase in prompt
    normalized['feedback']['cycle_context']['progress'][0]['error_end_xyz_m'][0] = 1
    normalized['scene']['robot_state']['joints'][0] = 1
    assert PROGRESS['error_end_xyz_m'][0] == .005 and ROBOT_STATE['joints'][0] == 0
    assert brain.validate_cycle_context(CYCLE) == CYCLE


@pytest.mark.parametrize('change', [{'observations': 0}, {'observations': 1001}, {'observations': True},
    {'phase': 'step_boundary'}, {'goal_end_xyz_m': [1.001, 0, 0]}, {'error_start_xyz_m': [float('inf'), 0, 0]},
    {'object_end_xyz_m': [True, 0, 0]}, {'object_start_xyz_m': None}, {'progress_verdict': 'success'}])
def test_invalid_progress_evidence_rejected(change):
    with pytest.raises(ValueError): brain.validate_cycle_context({**CYCLE, 'progress': [{**PROGRESS, **change}]})


def test_progress_bounded_entries_and_explicit_changing_goals():
    with pytest.raises(ValueError): brain.validate_cycle_progress([PROGRESS] * 9)
    assert brain.validate_cycle_progress([]) == []
    changed = {**PROGRESS, 'goal_end_xyz_m': [.21, 0, .01], 'observations': 1}
    assert brain.validate_cycle_progress([changed]) == [changed]
    assert brain.validate_cycle_progress([{**PROGRESS, 'error_end_xyz_m': [-1, 0, 1]}])


@pytest.mark.parametrize('change', [{'joints': [0]*4}, {'joints': [0, 0, 0, 0, float('nan')]},
    {'joints': [True, 0, 0, 0, 0]}, {'gripper_openness': True}, {'gripper_openness': 1.01},
    {'holding_status': 'held'}, {'raw': 'not allowed'}])
def test_invalid_robot_state_rejected(change):
    with pytest.raises(ValueError): brain.scene_snapshot({**SCENE, 'robot_state': {**ROBOT_STATE, **change}})


def test_robot_state_never_fabricates_holding_or_visual_scene():
    unknown = {'observation_status': 'unavailable', 'blocks': {}, 'robot_state': ROBOT_STATE}
    assert brain.scene_snapshot(unknown) == unknown
    with pytest.raises(ValueError): brain.validate_plan(PLAN, unknown)
    with pytest.raises(ValueError): brain.scene_snapshot({**SCENE, 'held_object': 'red', 'robot_state': ROBOT_STATE})
    verified = {**SCENE, 'held_object': 'red', 'robot_state': {**ROBOT_STATE, 'holding_status': 'verified_held'}}
    assert brain.scene_snapshot(verified)['held_object'] == 'red'
