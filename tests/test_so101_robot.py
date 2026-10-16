"""Robot-profile tests: official asset provenance, kinematics and real Bullet state."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pybullet_data
import pytest

from so101_robot import ARM_JOINT_NAMES, ASSET_DIR, SO101Robot, URDF_PATH


@pytest.fixture
def robot():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1/240, physicsClientId=client)
    instance = SO101Robot(client)
    yield instance
    p.disconnect(client)


def test_official_printed_asset_hashes_and_license():
    manifest = json.loads((ASSET_DIR / "source_manifest.json").read_text())
    assert manifest["commit"] == "a758567c3978dfeefe282ede0500085a48fe8f78"
    assert len(manifest["assets"]) == 11
    assert "Apache License" in (ASSET_DIR / "LICENSE").read_text()
    for item in manifest["assets"]:
        data = (ASSET_DIR / "assets" / item["path"].split("/")[-1]).read_bytes()
        assert len(data) == item["size"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    for item in manifest["generated_files"]:
        assert hashlib.sha256((ASSET_DIR / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    xml = ET.parse(URDF_PATH).getroot()
    assert not xml.findall(".//mimic")
    assert not any("sts3215" in mesh.attrib["filename"] for mesh in xml.findall(".//mesh"))
    for mesh in xml.findall(".//mesh"):
        assert (ASSET_DIR / mesh.attrib["filename"]).is_file()


def test_load_names_and_link_counts(robot):
    assert robot.call(p.getNumJoints, robot.body_id) == 8
    assert len(robot.arm_joint_indices) == 5
    assert len(set(robot.arm_joint_indices + [robot.gripper_joint_index])) == 6
    for name, index in zip(ARM_JOINT_NAMES, robot.arm_joint_indices):
        info = robot.call(p.getJointInfo, robot.body_id, index)
        assert info[1].decode() == name and info[2] == p.JOINT_REVOLUTE
    assert robot.call(p.getJointInfo, robot.body_id, robot.tcp_link_index)[2] == p.JOINT_FIXED
    assert robot.call(p.getNumConstraints) == 0


def test_render_and_gravity_hold(robot):
    view = p.computeViewMatrix([.5, -.5, .35], [.15, 0, .07], [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(45, 1.4, .01, 2)
    pixels = robot.call(p.getCameraImage, 280, 200, view, projection, renderer=p.ER_TINY_RENDERER)[2]
    assert np.asarray(pixels).shape == (200, 280, 4)
    assert np.std(np.asarray(pixels)[:, :, :3]) > 10
    target = robot.forward_kinematics(robot.home_joints)[:3, 3]
    for _ in range(240):
        robot.apply_motors()
        robot.call(p.stepSimulation)
    assert np.linalg.norm(np.asarray(robot.tcp_pose()[0])-target) < .003


@pytest.mark.parametrize("xyz", [[.19,-.05,.012], [.25,.055,.012], [.17,.105,.012],
                                 [.30,0,.012], [.15,-.15,.012], [.19,-.05,.06]])
def test_top_down_ik_reach_and_no_state_mutation(robot, xyz):
    before = robot.joint_positions()
    target_before = robot.joint_target.copy()
    q = robot.solve_ik(xyz)
    assert np.all(np.asarray(q) >= robot.lower_limits)
    assert np.all(np.asarray(q) <= robot.upper_limits)
    transform = robot.forward_kinematics(q)
    assert np.linalg.norm(transform[:3, 3]-xyz) < .002
    assert math.acos(np.clip(-transform[2, 2], -1, 1)) < .06
    assert robot.joint_positions() == before
    np.testing.assert_array_equal(robot.joint_target, target_before)
    # Verify independent Bullet forward kinematics using explicit test reset.
    for j, angle in zip(robot.arm_joint_indices, q):
        robot.call(p.resetJointState, robot.body_id, j, angle)
    np.testing.assert_allclose(robot.tcp_pose()[0], transform[:3, 3], atol=1e-6)


def test_ik_jacobian_matches_numeric_derivative(robot):
    q = np.asarray(robot.home_joints)
    transform, jp, ja = robot.forward_kinematics(q, True)
    for axis in range(5):
        delta = np.zeros(5); delta[axis] = 1e-6
        shifted = robot.forward_kinematics(q+delta)
        np.testing.assert_allclose((shifted[:3, 3]-transform[:3, 3])/1e-6, jp[:, axis], atol=1e-6)
        np.testing.assert_allclose((shifted[:3, 2]-transform[:3, 2])/1e-6, ja[:, axis], atol=1e-6)


def test_invalid_targets_and_unreachable_height_rejected(robot):
    for invalid in ([float("nan"),0,.02], [.2,0], [.5,0,.01], [.2,0,-.02]):
        with pytest.raises(ValueError): robot.solve_ik(invalid)
    with pytest.raises(ValueError, match="unreachable"):
        robot.solve_ik([.15, 0, .15])
    with pytest.raises(ValueError): robot.command_joints([0]*7)
    with pytest.raises(ValueError): robot.command_joints(robot.upper_limits + .01)
    with pytest.raises(ValueError, match="bounded step"):
        robot.command_cartesian([.3, .15, .02], max_step=.01)
    with pytest.raises(ValueError, match="bounded step"):
        robot.command_joints(robot.lower_limits, max_step=.01)


def test_single_revolute_gripper_and_proprioception(robot, monkeypatch):
    for opening in (0.0, .5, 1.0):
        target = robot.command_gripper(opening)
        lo, hi = robot.gripper_limits
        assert target == pytest.approx(lo+opening*(hi-lo))
        for _ in range(240):
            robot.apply_motors(); robot.call(p.stepSimulation)
        assert robot.proprioception()["gripper_openness"] == pytest.approx(opening, abs=.015)
    obs = robot.proprioception()
    assert obs["name"] == "SO-ARM101" and obs["arm_dof"] == 5
    assert len(obs["joints"]) == len(obs["joint_limits"]) == 5
    assert len(obs["ee"]) == 3
    assert not any(word in str(obs).lower() for word in ("cube", "object_id", "body_id", "contact", "force", "tactile"))
    oracle = robot.evaluation_tactile()
    assert oracle["simulation_only"] and oracle["not_for_model_input"]
    for tactile in oracle["jaws"].values():
        assert set(tactile) == {"contact_count", "total_normal_force"}
        assert tactile["contact_count"] == 0 and tactile["total_normal_force"] == 0
    def contacts_forbidden(*args, **kwargs):
        pytest.fail("Default proprioception must not even query simulated contacts")
    monkeypatch.setattr(p, "getContactPoints", contacts_forbidden)
    assert robot.proprioception()["name"] == "SO-ARM101"
    for bad in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError): robot.command_gripper(bad)


def test_physical_cube_grasp_lift_without_constraints(robot):
    """Evaluation ground truth is used only here, never as an adapter input."""
    robot.call(p.setAdditionalSearchPath, pybullet_data.getDataPath())
    robot.call(p.loadURDF, "plane.urdf")
    robot.call(p.setPhysicsEngineParameter, numSolverIterations=150)
    shape = robot.call(p.createCollisionShape, p.GEOM_BOX, halfExtents=[.01]*3)
    cube = robot.call(p.createMultiBody, baseMass=.012, baseCollisionShapeIndex=shape,
                      basePosition=[.19, -.05, .011])
    robot.call(p.changeDynamics, cube, -1, lateralFriction=1.2)
    def move(xyz):
        start = np.asarray(robot.joint_positions())
        goal = np.asarray(robot.solve_ik(xyz, seed=start))
        for t in np.linspace(0, 1, 360):
            robot.command_joints(start + (goal-start)*(t*t*(3-2*t)))
            robot.call(p.stepSimulation)
        for _ in range(120):
            robot.apply_motors(); robot.call(p.stepSimulation)
        assert np.linalg.norm(np.asarray(robot.tcp_pose()[0])-xyz) < .003
    move([.19, -.05, .06])
    move([.19, -.05, .012])
    robot.command_gripper(0)
    for _ in range(360):
        robot.apply_motors(); robot.call(p.stepSimulation)
    contacts = robot.call(p.getContactPoints, robot.body_id, cube)
    assert {c[3] for c in contacts} >= {robot.link_indices["gripper_link"], robot.gripper_joint_index}
    for tactile in robot.evaluation_tactile()["jaws"].values():
        assert tactile["contact_count"] > 0 and tactile["total_normal_force"] > 0
    move([.19, -.05, .06])
    assert robot.call(p.getBasePositionAndOrientation, cube)[0][2] > .05
    assert robot.call(p.getNumConstraints) == 0
    robot.command_gripper(1)
    for _ in range(480):
        robot.apply_motors(); robot.call(p.stepSimulation)
    assert robot.call(p.getBasePositionAndOrientation, cube)[0][2] < .015
