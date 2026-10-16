"""Ground symbolic cube tasks in fresh measurements, without model coordinates.

Qwen chooses the symbolic objects, targets, and order. This module supplies only
geometry-specific, deterministic manipulation primitives for TaskExecutor. It
neither calls a model nor claims that deterministic servo tests exercise Jev.
All grasping and stacking in the simulator remain ordinary contact physics.
"""
from __future__ import annotations

import math
from typing import Any

from embodied_arm.simulation.simulator import WORKSPACE
from embodied_arm.legacy.panda.task_executor import validate_plan

CUBE_HALF_HEIGHT_M = .025
GRIPPER_CENTER_OFFSET_M = .004
RELEASE_CLEARANCE_M = .003
TRAVEL_HEIGHT_M = .27
STACK_XY_TOLERANCE_M = .03
PLACEMENT_Z_TOLERANCE_M = .015
SUPPORT_DISPLACEMENT_TOLERANCE_M = .015
MIN_HELD_HEIGHT_M = .13
KNOWN_OBJECTS = frozenset(('red', 'blue'))


def _xyz(value: Any, label: str) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or not all(type(v) in (int, float) and math.isfinite(v) for v in value)
            or not all(lo <= v <= hi for v, (lo, hi) in zip(value, WORKSPACE))):
        raise ValueError(f'{label} requires finite coordinates inside the workspace')
    return list(value)


def _intent(intent: Any) -> tuple[str, str, str | None]:
    if (not isinstance(intent, dict)
            or set(intent) - {'step_id', 'action', 'object_id', 'target_id', 'message'}):
        raise ValueError('Invalid symbolic step fields')
    if 'step_id' in intent and (not isinstance(intent['step_id'], str)
                              or not 1 <= len(intent['step_id']) <= 64):
        raise ValueError('Invalid symbolic step ID')
    action, obj, target = (intent.get(k) for k in ('action', 'object_id', 'target_id'))
    if not isinstance(action, str) or action not in ('pick', 'place'):
        raise ValueError('Unsupported symbolic action')
    if not isinstance(obj, str) or obj not in KNOWN_OBJECTS:
        raise ValueError('Unknown object')
    if target is not None and (not isinstance(target, str)
                              or target not in ('goal', 'on:red', 'on:blue')):
        raise ValueError('Unknown target')
    if action == 'pick' and target not in (None, 'goal'):
        raise ValueError('Stacking requires separate pick and place steps')
    if action == 'place' and target is None:
        raise ValueError('Place requires a known target')
    if target == f'on:{obj}':
        raise ValueError('Cannot place an object on itself')
    return action, obj, target


def _scene(state: Any) -> tuple[dict[str, list[float]], list[float]]:
    if not isinstance(state, dict) or not isinstance(state.get('blocks'), dict):
        raise ValueError('Invalid measured scene')
    blocks = {name: _xyz(xyz, f'{name} object') for name, xyz in state['blocks'].items()
              if name in KNOWN_OBJECTS}
    if not blocks:
        raise ValueError('Scene has no known objects')
    ee = _xyz(state.get('ee'), 'End effector')
    return blocks, ee


def _is_held(state: dict, obj: str, blocks: dict, ee: list) -> bool:
    xyz = blocks[obj]
    width = state.get('gripper_width')
    return (xyz[2] > MIN_HELD_HEIGHT_M and math.dist(xyz, ee) < .065
            and type(width) in (int, float) and math.isfinite(width)
            and 0 <= width <= .065)


def _target_center(target: str, obj: str, state: dict, blocks: dict) -> list[float]:
    if target == 'goal':
        center = _xyz(state.get('goal'), 'Goal')
        # The green region is on the ground; its Z is a cube-center coordinate.
        if abs(center[2] - CUBE_HALF_HEIGHT_M) > .005:
            raise ValueError('Goal must be the measured ground-level cube center')
        for other, xyz in blocks.items():
            if other != obj and math.dist(xyz[:2], center[:2]) < .055:
                raise ValueError('Goal is occupied; use a stacking target')
        return center
    support = target.split(':', 1)[1]
    if support not in blocks:
        raise ValueError('Support object is missing')
    xyz = blocks[support]
    # There are only two cubes and a floor in this sandbox. A third/elevated
    # support is not something the grounding layer may silently invent.
    if abs(xyz[2] - CUBE_HALF_HEIGHT_M) > .015:
        raise ValueError('Support is not stable on the floor')
    return _xyz([xyz[0], xyz[1], xyz[2] + 2 * CUBE_HALF_HEIGHT_M], 'Stack center')


def plan_for_symbolic_step(intent: dict, state: dict) -> dict:
    """Return one validated TaskExecutor plan, using state from this boundary.

    Do not pre-ground an entire Qwen sequence: after every symbolic step, obtain
    a new sim.state(), check its measured outcome, then ground the next step.
    Legacy ``pick -> goal`` remains supported; sequence picks have no target.
    No model-provided Cartesian coordinates or arbitrary body IDs are accepted.
    """
    action, obj, target = _intent(intent)
    blocks, ee = _scene(state)
    if obj not in blocks:
        raise ValueError('Object is missing')
    if state.get('busy') is True:
        raise ValueError('Wait for the previous physical motion to finish')
    block = blocks[obj]
    if action == 'pick':
        if any(_is_held(state, name, blocks, ee) for name in blocks):
            raise ValueError('Place the held object before picking another')
        for other, xyz in blocks.items():
            if (other != obj and math.dist(block[:2], xyz[:2]) < .035
                    and .035 < xyz[2] - block[2] < .07):
                raise ValueError('Cannot remove a cube supporting another cube')
    elif not _is_held(state, obj, blocks, ee):
        raise ValueError('Place requires the selected object to be measurably held')

    center = _target_center(target, obj, state, blocks) if target is not None else None
    # Horizontal travel is above every stationary cube. Never retract diagonally
    # past a support: first move vertically at the current measured TCP location.
    stationary_heights = [xyz[2] + CUBE_HALF_HEIGHT_M for name, xyz in blocks.items()
                          if action == 'pick' or name != obj]
    travel = max(TRAVEL_HEIGHT_M, max(stationary_heights, default=0) + .12,
                 (center[2] + .10) if center else 0)
    if travel > .45:
        raise ValueError('Insufficient safe vertical travel inside the workspace')
    steps = []
    if action == 'pick':
        grip_z = block[2] + GRIPPER_CENTER_OFFSET_M
        steps.extend([
            {'action': 'open'},
            {'action': 'move', 'target_xyz': [ee[0], ee[1], travel]},
            {'action': 'move', 'target_xyz': [block[0], block[1], travel]},
            {'action': 'move', 'target_xyz': [block[0], block[1], grip_z]},
            {'action': 'close'},
            {'action': 'lift', 'height_m': travel},
        ])
    else:
        steps.append({'action': 'move', 'target_xyz': [ee[0], ee[1], travel]})
    if center is not None:
        # The lifted cube can sit lower/off-center between physical fingers.
        # Measured grasp offset plus 3 mm clearance avoids driving the held cube
        # into its support. Nominal stack TCP is .075 + .004 = .079 m, but a
        # real grasp can need a slightly higher release before gravity settles it.
        offset = ([ee[i] - block[i] for i in range(3)] if action == 'place'
                  else [0., 0., GRIPPER_CENTER_OFFSET_M])
        grip_x, grip_y = center[0] + offset[0], center[1] + offset[1]
        grip_z = center[2] + max(GRIPPER_CENTER_OFFSET_M, offset[2]) + RELEASE_CLEARANCE_M
        steps.extend([
            {'action': 'move', 'target_xyz': [grip_x, grip_y, travel]},
            {'action': 'move', 'target_xyz': [grip_x, grip_y, grip_z]},
            {'action': 'open'},
            {'action': 'lift', 'height_m': travel},
        ])
    return validate_plan({'steps': steps})


def symbolic_step_result(intent: dict, before: dict, after: dict) -> dict:
    """Measure task success separately from reaching all motor waypoints.

    Stable support means its measured center stayed within 1.5 cm of its initial
    pose and remains at floor height. Call only after release and full vertical
    retraction, which gives contact physics time to settle. This is an explicit
    positional predicate, not a claim to force sensing or real-robot stability.
    """
    action, obj, target = _intent(intent)
    try:
        initial, _ = _scene(before)
        current, ee = _scene(after)
        if obj not in initial or obj not in current:
            raise ValueError('Object is missing')
        result = {'success': False, 'verified': False, 'reason': 'placement_not_verified',
                  'object_id': obj, 'target_id': target, 'object_xyz': current[obj]}
        if action == 'pick' and target is None:
            ok = _is_held(after, obj, current, ee) and current[obj][2] - initial[obj][2] > .04
            result.update(success=ok, verified=ok, reason='completed' if ok else 'grasp_not_verified')
            return result
        center = _target_center(target, obj, before, initial)
        xy_error = math.dist(current[obj][:2], center[:2])
        z_error = abs(current[obj][2] - center[2])
        released = (not _is_held(after, obj, current, ee) and ee[2] - current[obj][2] > .06
                    and type(after.get('gripper_width')) in (int, float)
                    and math.isfinite(after['gripper_width']) and .065 < after['gripper_width'] <= .2)
        ok = xy_error <= STACK_XY_TOLERANCE_M and z_error <= PLACEMENT_Z_TOLERANCE_M and released
        result.update(target_xyz=center, xy_error_m=xy_error, z_error_m=z_error,
                      released=released, success=ok, verified=ok,
                      reason='completed' if ok else 'placement_not_verified')
        if target.startswith('on:'):
            support = target.split(':', 1)[1]
            if support not in current:
                raise ValueError('Support object is missing')
            displacement = math.dist(initial[support], current[support])
            floor_error = abs(current[support][2] - CUBE_HALF_HEIGHT_M)
            result.update(support_id=support, support_displacement_m=displacement,
                          support_stable=displacement <= SUPPORT_DISPLACEMENT_TOLERANCE_M and floor_error <= .015)
            if not result['support_stable']:
                result.update(success=False, verified=False,
                              reason='support_moved' if displacement > SUPPORT_DISPLACEMENT_TOLERANCE_M else 'support_unstable')
            else:
                # Also demand actual relative alignment with the final support;
                # two cubes shifting separately cannot pass a stale target check.
                relative_xy = math.dist(current[obj][:2], current[support][:2])
                relative_z = abs(current[obj][2] - current[support][2] - 2 * CUBE_HALF_HEIGHT_M)
                result.update(relative_xy_error_m=relative_xy, relative_z_error_m=relative_z)
                if relative_xy > STACK_XY_TOLERANCE_M or relative_z > PLACEMENT_Z_TOLERANCE_M:
                    result.update(success=False, verified=False, reason='placement_not_verified')
        return result
    except (KeyError, TypeError, ValueError):
        return {'success': False, 'verified': False, 'reason': 'invalid_observation',
                'object_id': obj, 'target_id': target}
