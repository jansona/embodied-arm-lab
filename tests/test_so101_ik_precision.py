"""Numerical IK and bounded real-servo settling, independent of scene GT."""
import numpy as np
import pytest
from so101_simulation import SO101Simulation


@pytest.mark.parametrize('xyz',[[.2,0,.06],[.19,-.05,.06],[.25,.055,.06],
                                [.18,-.05,.013],[.24,.055,.013]])
def test_ik_precision_at_grasp_and_hover_robot_targets(xyz):
    sim=SO101Simulation()
    try:
        q=sim.robot.solve_ik(xyz)
        pose=sim.robot.forward_kinematics(q)
        assert np.linalg.norm(pose[:3,3]-xyz) <= .0001
        assert sim.robot.last_ik['position_residual_m'] <= .0001
    finally:sim.close()


def test_cartesian_settling_keeps_same_target_and_bounds_runtime():
    sim=SO101Simulation()
    try:
        target=list(sim.proprioception()['ee']);target[1]-=.008
        sim.resume();sim.manual('xyz',target,origin='offline_fixture')
        assert sim.queue[0]['ticks']==144 and sim.queue[0]['settle_ticks']==120
        sim.tick(144)
        assert sim.proprioception()['busy']
        commanded=sim.robot.joint_target.copy()
        sim.tick(119)
        assert sim.proprioception()['busy']
        sim.tick(1)
        assert not sim.proprioception()['busy'] and sim.actions==1
        np.testing.assert_allclose(sim.robot.joint_target,commanded,atol=1e-12)
        assert np.linalg.norm(np.asarray(sim.proprioception()['ee'])-target)<.00005
    finally:sim.close()
