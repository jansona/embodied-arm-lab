"""Focused regression for the Panda's physical jaw mimic linkage, offline."""
import math
import pybullet as p
from simulator import Simulation
from gripper_stability_diagnostic import run


def test_reset_has_exactly_one_robot_only_finger_linkage():
    sim = Simulation()
    try:
        for _ in range(3):
            assert sim.call(p.getNumConstraints) == 1
            info = sim.call(p.getConstraintInfo, sim.finger_constraint)
            assert info[:5] == (sim.robot, 9, sim.robot, 10, p.JOINT_GEAR)
            assert info[0] not in sim.blocks.values()
            assert info[2] not in sim.blocks.values()
            sim.reset()
    finally:
        sim.close()


def test_physical_blue_grasp_stays_centered_for_30_simulation_seconds():
    result = run('mimic_before_pick', seconds=30)
    assert result['controller'] == 'deterministic servo'
    assert result['controller_calls'] == 0
    rows = result['samples']
    assert rows[-1]['simulation_seconds'] == 30
    assert result['cube_displacement_m'] < .0005, result
    assert result['last_5s_cube_displacement_m'] < .0001, result
    assert all(row['contacts'] == [1, 1] for row in rows)
    assert all(row['cube'][2] > .2 for row in rows)
    assert all(abs(row['jaw_difference']) < .0001 for row in rows)
    offsets = [[a - b for a, b in zip(row['cube'], row['ee'])] for row in rows]
    assert max(math.dist(offsets[0], offset) for offset in offsets) < .0001
