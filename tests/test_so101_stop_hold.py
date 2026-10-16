"""Independent real-physics fixture regression for stop/resume holding force."""
import pytest
import pybullet as p
from so101_simulation import SO101Simulation
from tests.test_so101_grasp_feasibility import _fixture_move,_settle


@pytest.mark.parametrize('name',['red','blue'])
def test_idle_stop_resume_carry_retains_completed_gripper_setpoint(name):
    sim=SO101Simulation()
    try:
        # Explicit independent fixture coordinates; never controller input.
        xy=sim.evaluation_state()['ground_truth_objects'][name][:2]
        target=[*xy,.013]
        offset=sim.grasp_geometry(target)['approach_offset_world']
        target=[a+b for a,b in zip(target,offset)]
        sim.resume();_fixture_move(sim,[*target[:2],.06]);_fixture_move(sim,target)
        sim.manual('close',origin='offline_fixture');_settle(sim,480)
        _fixture_move(sim,[*target[:2],.06]);_settle(sim,480)
        before=sim.robot.gripper_target
        assert before==pytest.approx(sim.robot.gripper_limits[0])
        assert sim.evaluation_state()['ground_truth_objects'][name][2]>.04
        sim.stop()
        assert sim.paused and not sim.current and not sim.queue
        assert sim.robot.gripper_target==before
        sim.resume();_settle(sim,480)
        _fixture_move(sim,[target[0],target[1]+.04,.06]);_settle(sim,1200)
        assert sim.evaluation_state()['ground_truth_objects'][name][2]>.04
        jaws={sim.robot.link_indices['gripper_link'],sim.robot.gripper_joint_index}
        contacts=sim.call(p.getContactPoints,sim.robot.body_id,sim._objects[name])
        assert jaws <= {c[3] for c in contacts if c[9]>0}
        assert sim.call(p.getNumConstraints)==0
    finally:sim.close()


@pytest.mark.parametrize('action',['close','open'])
def test_active_gripper_stop_freezes_measured_angle(action):
    sim=SO101Simulation()
    try:
        sim.resume()
        if action=='open':
            sim.manual('close',origin='offline_fixture');_settle(sim,480)
        sim.manual(action,origin='offline_fixture');sim.tick(180)
        assert sim.current and sim.current['gripper'] is not None
        measured=sim.call(p.getJointState,sim.robot.body_id,sim.robot.gripper_joint_index)[0]
        sim.stop()
        assert sim.robot.gripper_target==pytest.approx(measured,abs=1e-10)
        assert sim.paused and not sim.current and not sim.queue
    finally:sim.close()


def test_queued_unstarted_close_is_cancelled_without_starting_it():
    sim=SO101Simulation()
    try:
        target=sim.robot.gripper_target
        sim.resume();sim.manual('close',origin='offline_fixture');sim.stop()
        assert sim.robot.gripper_target==target and not sim.queue
    finally:sim.close()
