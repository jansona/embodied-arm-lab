"""Offline contact-physics diagnosis; never imports a network/model adapter.

Compares the existing independent jaw motors with the Panda URDF's specified
mimic linkage represented by a robot-only gear constraint. No cube constraint,
teleport, or friction change. Production simulator settings are left unchanged.
"""
import json
import math
import time
import pybullet as p
from simulator import Simulation
from stack_plans import plan_for_symbolic_step
from task_executor import TaskExecutor


def couple_jaws(sim):
    if hasattr(sim, 'finger_constraint'):
        return sim.finger_constraint
    constraint = sim.call(p.createConstraint, sim.robot, 9, sim.robot, 10,
                          p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
    sim.call(p.changeConstraint, constraint, gearRatio=-1, erp=.1, maxForce=50)
    sim.finger_constraint = constraint
    return constraint


def sample(sim, elapsed):
    positions = [sim.call(p.getJointState, sim.robot, joint) for joint in (9, 10)]
    cube, orientation = sim.call(p.getBasePositionAndOrientation, sim.blocks['blue'])
    contacts = [sim.call(p.getContactPoints, sim.robot, sim.blocks['blue'], linkIndexA=joint)
                for joint in (9, 10)]
    return {'simulation_seconds': elapsed, 'ee': list(sim.ee()), 'cube': list(cube),
            'cube_velocity': list(sim.call(p.getBaseVelocity, sim.blocks['blue'])[0]),
            'cube_orientation': list(orientation),
            'jaw_positions': [j[0] for j in positions],
            'jaw_velocities': [j[1] for j in positions],
            'jaw_width': sum(j[0] for j in positions),
            'jaw_difference': positions[0][0] - positions[1][0],
            'contacts': [len(c) for c in contacts],
            'normal_forces': [sum(c[9] for c in side) for side in contacts]}


def run(case, seconds=12):
    sim = Simulation()
    try:
        if case not in ('independent_jaws', 'mimic_before_pick', 'mimic_after_pick'):
            raise ValueError('Unknown diagnostic case')
        if case != 'mimic_before_pick' and hasattr(sim, 'finger_constraint'):
            sim.call(p.removeConstraint, sim.finger_constraint)
            del sim.finger_constraint
        if case == 'mimic_before_pick':
            couple_jaws(sim)
        executor = TaskExecutor(sim)
        executor.start(plan_for_symbolic_step({'action': 'pick', 'object_id': 'blue'}, sim.state()))
        deadline = time.monotonic() + 30
        while executor.status()['status'] == 'running' and time.monotonic() < deadline:
            sim.tick(96)
            time.sleep(.0005)
        if executor.status()['status'] != 'completed':
            raise RuntimeError(executor.status())
        rows = [sample(sim, 0)]
        if case == 'mimic_after_pick':
            couple_jaws(sim)
        with sim.lock:
            sim.paused = False
        for second in range(1, seconds + 1):
            sim.tick(240)
            rows.append(sample(sim, second))
        first, last = rows[0], rows[-1]
        return {'case': case, 'controller': executor.label, 'controller_calls': 0,
                'seconds': seconds,
                'cube_displacement_m': math.dist(first['cube'], last['cube']),
                'cube_delta': [b - a for a, b in zip(first['cube'], last['cube'])],
                'ee_displacement_m': math.dist(first['ee'], last['ee']),
                'jaw_difference_delta': last['jaw_difference'] - first['jaw_difference'],
                'last_5s_cube_displacement_m': math.dist(rows[-6]['cube'], last['cube']),
                'samples': rows}
    finally:
        sim.close()


if __name__ == '__main__':
    for case in ('independent_jaws', 'mimic_before_pick', 'mimic_after_pick'):
        print(json.dumps(run(case)), flush=True)
