"""Offline grounding tests and explicitly deterministic physical baselines.

No network, Qwen, Jev, teleportation, or grasp constraint is used by these tests.
"""
from copy import deepcopy
import math
import time

import pybullet as p
import pytest

from embodied_arm.simulation.simulator import Simulation
from embodied_arm.legacy.panda.stack_plans import plan_for_symbolic_step, symbolic_step_result
from embodied_arm.legacy.panda.task_executor import TaskExecutor, validate_plan


def scene():
    return {'blocks': {'red': [.50, 0, .025], 'blue': [.58, -.19, .025]},
            'ee': [.50, 0, .27], 'goal': [.48, .21, .025],
            'gripper_width': .08, 'busy': False}


def held_scene(obj='red'):
    state = scene()
    state['blocks'][obj] = [.50, 0, .266]
    state['gripper_width'] = .05
    return state


def step(action='pick', obj='red', target=None, sid='s1'):
    return {'step_id': sid, 'action': action, 'object_id': obj, 'target_id': target}


def test_stack_geometry_uses_measured_support_top_and_cube_halfheight():
    state = held_scene()
    state['blocks']['blue'] = [.56, -.12, .026]
    plan = plan_for_symbolic_step(step('place', target='on:blue'), state)
    assert validate_plan(plan) == plan
    assert plan['steps'][2]['action'] == 'move'
    assert plan['steps'][2]['target_xyz'] == pytest.approx([.56, -.12, .083])
    assert plan['steps'][-2:] == [{'action': 'open'}, {'action': 'lift', 'height_m': .27}]
    assert state['blocks']['blue'] == [.56, -.12, .026]


def test_grounding_is_fresh_after_support_moved_between_symbolic_steps():
    state = held_scene()
    first = plan_for_symbolic_step(step('place', target='on:blue'), state)
    state['blocks']['blue'] = [.48, .21, .025]
    second = plan_for_symbolic_step(step('place', target='on:blue'), state)
    assert first['steps'][1]['target_xyz'][:2] == [.58, -.19]
    assert second['steps'][1]['target_xyz'][:2] == [.48, .21]
    assert second['steps'][2]['target_xyz'][2] == pytest.approx(.082)


def test_pick_uses_measured_height_and_vertical_clearance():
    state = scene()
    state['blocks']['red'] = [.58, -.19, .075]
    state['ee'] = [.40, 0, .13]
    plan = plan_for_symbolic_step(step(), state)
    assert plan['steps'][1]['target_xyz'][:2] == state['ee'][:2]
    assert plan['steps'][2]['target_xyz'] == [.58, -.19, .27]
    assert plan['steps'][3]['target_xyz'] == pytest.approx([.58, -.19, .079])
    assert plan['steps'][-1] == {'action': 'lift', 'height_m': .27}


@pytest.mark.parametrize('intent', [
    {}, step('execute'), step(obj='green'), step('place', target=None),
    step('place', target='on:red'), step('place', target='on:green'),
    step('pick', target='on:blue'), {**step(), 'target_xyz': [.4, 0, .3]},
    {**step(), 'object_id': []}, {**step(), 'target_id': []},
])
def test_invalid_symbols_never_become_motor_commands(intent):
    with pytest.raises(ValueError):
        plan_for_symbolic_step(intent, held_scene())


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1., True, '0.5'])
def test_invalid_measured_geometry_is_rejected(value):
    state = scene()
    state['blocks']['red'][0] = value
    with pytest.raises(ValueError):
        plan_for_symbolic_step(step(), state)


def test_place_cannot_run_without_a_verified_held_object():
    with pytest.raises(ValueError, match='held'):
        plan_for_symbolic_step(step('place', target='on:blue'), scene())
    state = held_scene()
    state['ee'][0] = .3
    with pytest.raises(ValueError, match='held'):
        plan_for_symbolic_step(step('place', target='on:blue'), state)


def test_pick_cannot_release_another_held_object():
    with pytest.raises(ValueError, match='held'):
        plan_for_symbolic_step(step(obj='blue'), held_scene())


def test_cannot_pick_support_or_place_on_occupied_goal():
    state = scene()
    state['blocks']['blue'] = [.5, 0, .075]
    with pytest.raises(ValueError, match='supporting'):
        plan_for_symbolic_step(step(), state)
    state = held_scene()
    state['blocks']['blue'] = list(state['goal'])
    with pytest.raises(ValueError, match='occupied'):
        plan_for_symbolic_step(step('place', target='goal'), state)


def test_support_must_be_stable_and_known():
    state = held_scene()
    state['blocks']['blue'][2] = .20
    with pytest.raises(ValueError, match='stable'):
        plan_for_symbolic_step(step('place', target='on:blue'), state)
    del state['blocks']['blue']
    with pytest.raises(ValueError, match='missing'):
        plan_for_symbolic_step(step('place', target='on:blue'), state)


def placed_scene(before, target='on:blue'):
    after = deepcopy(before)
    center = ([*before['blocks']['blue'][:2], before['blocks']['blue'][2] + .05]
              if target == 'on:blue' else before['goal'])
    after['blocks']['red'] = list(center)
    after['ee'] = [center[0], center[1], .27]
    after['gripper_width'] = .08
    return after


def test_outcome_requires_released_cube_and_measured_stack_alignment():
    before = held_scene()
    intent = step('place', target='on:blue')
    after = placed_scene(before)
    assert symbolic_step_result(intent, before, after)['success']
    after['blocks']['red'][2] = .025
    assert not symbolic_step_result(intent, before, after)['success']
    after = placed_scene(before)
    after['gripper_width'] = .05
    assert not symbolic_step_result(intent, before, after)['success']
    after = placed_scene(before)
    after['blocks']['red'][0] += .04
    assert not symbolic_step_result(intent, before, after)['success']


def test_support_moved_feedback_even_when_cubes_move_together():
    before = held_scene()
    after = placed_scene(before)
    after['blocks']['red'][0] += .02
    after['blocks']['blue'][0] += .02
    result = symbolic_step_result(step('place', target='on:blue'), before, after)
    assert not result['success']
    assert result['reason'] == 'support_moved'
    assert not result['support_stable']


def test_invalid_observation_cannot_pass_result_predicate():
    before = held_scene()
    after = placed_scene(before)
    after['blocks']['red'][2] = float('nan')
    assert symbolic_step_result(step('place', target='on:blue'), before, after)['reason'] == 'invalid_observation'


def test_pick_outcome_requires_actual_lift_and_proximity():
    before = scene()
    after = held_scene()
    assert symbolic_step_result(step(), before, after)['success']
    after['ee'][0] = .3
    assert symbolic_step_result(step(), before, after)['reason'] == 'grasp_not_verified'
    assert not symbolic_step_result(step(), before, before)['success']


def run_baseline_step(sim, intent):
    """Only the existing deterministic servo; no model adapter is constructed."""
    before = sim.state()
    executor = TaskExecutor(sim)
    executor.start(plan_for_symbolic_step(intent, before), max_seconds=60)
    deadline = time.monotonic() + 30
    while executor.status()['status'] == 'running' and time.monotonic() < deadline:
        sim.tick(96)
        time.sleep(.0005)
    executor.thread.join(1)
    result = executor.status()
    assert result['status'] == 'completed', result
    assert result['controller'] == 'deterministic servo'
    assert result['controller_calls'] == 0
    assert result['actions'] <= 100
    outcome = symbolic_step_result(intent, before, sim.state())
    assert outcome['success'], outcome
    assert sim.call(p.getNumConstraints) == 1
    linkage = sim.call(p.getConstraintInfo, sim.finger_constraint)
    assert linkage[:5] == (sim.robot, 9, sim.robot, 10, p.JOINT_GEAR)
    return outcome


@pytest.mark.parametrize('top,bottom,seed', [('red', 'blue', 0), ('blue', 'red', 0),
                                           ('red', 'blue', 1), ('blue', 'red', 1)])
def test_deterministic_physics_baseline_stacks_with_real_contacts(top, bottom, seed):
    sim = Simulation()
    sim.reset(seed)
    try:
        run_baseline_step(sim, step('pick', top))
        # Grasp support is by actual two-finger contact, with no attachment.
        assert all(sim.call(p.getContactPoints, sim.robot, sim.blocks[top], linkIndexA=j)
                   for j in (9, 10))
        outcome = run_baseline_step(sim, step('place', top, f'on:{bottom}', 's2'))
        assert outcome['support_stable']
        state = sim.state()
        assert state['blocks'][top][2] == pytest.approx(.075, abs=.015)
        assert math.dist(state['blocks'][top][:2], state['blocks'][bottom][:2]) <= .03
        assert sim.call(p.getContactPoints, sim.blocks[top], sim.blocks[bottom])
    finally:
        sim.close()


def test_deterministic_four_step_stack_grounds_relocated_support():
    sim = Simulation()
    try:
        for intent in [step('pick', 'blue'), step('place', 'blue', 'goal', 's2'),
                       step('pick', 'red', sid='s3'), step('place', 'red', 'on:blue', 's4')]:
            run_baseline_step(sim, intent)
        final = sim.state()
        assert math.dist(final['blocks']['blue'][:2], final['goal'][:2]) <= .03
        assert final['blocks']['red'][2] == pytest.approx(.075, abs=.015)
        assert sim.call(p.getContactPoints, sim.blocks['red'], sim.blocks['blue'])
    finally:
        sim.close()


def test_deterministic_legacy_pick_to_goal_remains_supported():
    sim = Simulation()
    try:
        outcome = run_baseline_step(sim, step('pick', 'red', 'goal'))
        assert outcome['z_error_m'] < .015
        assert sim.state()['metrics']['placed']
    finally:
        sim.close()
