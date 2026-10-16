"""Independent known-fixture PHYSICS tests, NOT an image-only controller.

Ground-truth object coordinates below define fixture targets and score results only.
No model, production planner, perception adapter, network, or credentials are used.
Passing proves servo/contact feasibility, not RGB detection or Jev/Qwen success.
"""
import math
import numpy as np
import pybullet as p
import pytest
from embodied_arm.simulation.so101_simulation import SO101Simulation


def _settle(sim, ticks=264):
    sim.tick(ticks)
    assert sim.current is None and not sim.queue


def _fixture_move(sim, xyz):
    """Known-fixture trajectory through the public, bounded servo facade."""
    for _ in range(30):
        start = np.asarray(sim.robot.tcp_pose()[0])
        delta = np.asarray(xyz)-start
        if np.linalg.norm(delta) < .001:
            return
        target = start+delta*min(1., .015/np.linalg.norm(delta))
        sim.manual('xyz', target.tolist(), origin='offline_fixture')
        _settle(sim)
    pytest.fail('Fixture target not reached with bounded servo commands')


@pytest.mark.parametrize('seed', [0, 7, 31])
@pytest.mark.parametrize('object_name', ['red', 'blue'])
def test_known_fixture_facade_contact_grasp_lift_release(seed, object_name, monkeypatch):
    sim = SO101Simulation(seed)
    try:
        # Explicit independent fixture oracle, never sent to a production controller.
        fixture = sim.evaluation_state()['ground_truth_objects'][object_name]
        body = sim._objects[object_name]
        original_z = fixture[2]
        def forbidden(*args, **kwargs):
            pytest.fail('Physics grasp may not teleport bodies/joints or attach constraints')
        # Resets are permitted solely in construction above, never during movement.
        for name in ('resetJointState', 'resetBasePositionAndOrientation',
                     'resetBaseVelocity', 'createConstraint', 'changeConstraint'):
            monkeypatch.setattr(p, name, forbidden)
        sim.resume()
        _fixture_move(sim, [fixture[0], fixture[1], .06])
        _fixture_move(sim, [fixture[0], fixture[1], .012])
        sim.manual('close', origin='offline_fixture')
        _settle(sim, 360)
        contacts = sim.call(p.getContactPoints, sim.robot.body_id, body)
        expected_jaws = {sim.robot.link_indices['gripper_link'], sim.robot.gripper_joint_index}
        assert {c[3] for c in contacts if c[9] > 0} >= expected_jaws
        _fixture_move(sim, [fixture[0], fixture[1], .06])
        _settle(sim, 480)
        lifted = sim.evaluation_state()['ground_truth_objects'][object_name]
        assert lifted[2]-original_z > .035
        assert math.dist(lifted[:2], fixture[:2]) < .02
        assert sim.call(p.getNumConstraints) == 0
        contacts = sim.call(p.getContactPoints, sim.robot.body_id, body)
        assert {c[3] for c in contacts if c[9] > 0} >= expected_jaws
        closed_angle = sim.proprioception()['gripper_angle']
        sim.manual('open', origin='offline_fixture')
        _settle(sim, 720)
        released = sim.evaluation_state()['ground_truth_objects'][object_name]
        assert released[2] < .015
        assert sim.proprioception()['gripper_openness'] > .99
        print({'physics_fixture_only': True, 'seed': seed, 'object': object_name,
               'lift_m': lifted[2]-original_z, 'held_gripper_angle_rad': closed_angle,
               'actions': sim.actions, 'release_z_m': released[2]})
    finally:
        sim.close()
