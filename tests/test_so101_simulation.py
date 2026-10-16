"""SO101 facade regression tests: sensor boundary, motion limits and cancellation."""
import base64
from io import BytesIO
import math

import numpy as np
from PIL import Image
import pybullet as p
import pytest

from embodied_arm.simulation.so101_simulation import CAMERAS, SO101Simulation, camera_calibration


@pytest.fixture
def sim():
    instance = SO101Simulation()
    yield instance
    instance.close()


def finish(sim):
    for _ in range(120):
        sim.tick(4)
        if sim.current is None and not sim.queue:
            sim.tick(120)
            return
    pytest.fail("SO101 motion did not finish in the bounded simulation budget")


def all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from all_keys(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from all_keys(child)


def test_proprioception_default_hardware_has_no_oracle_sensors(sim, monkeypatch):
    def oracle_forbidden():
        pytest.fail("Model-facing sensor path called the evaluation oracle")
    monkeypatch.setattr(sim, "evaluation_state", oracle_forbidden)
    obs = sim.proprioception()
    assert obs["schema"] == "so101-proprioception-v1"
    assert obs["name"] == "SO-ARM101" and obs["arm_dof"] == 5
    assert obs["simulation_only"] and obs["paused"] and not obs["busy"]
    assert len(obs["joints"]) == len(obs["joint_limits"]) == 5
    assert len(obs["ee"]) == 3
    forbidden = ("force", "tactile", "contact", "body_id", "object_id", "ground_truth",
                 "segmentation", "mask", "depth", "metric", "score")
    assert not [key for key in all_keys(obs) if any(term in key.lower() for term in forbidden)]
    assert not any(key in obs for key in ("red", "blue", "blocks", "objects", "goal"))


def test_paired_rgb_capture_calibration_and_no_depth_masks(sim, monkeypatch):
    render_calls = []
    original_call = sim.call
    def inspect_call(fn, *args, **kwargs):
        if fn is p.getCameraImage:
            render_calls.append(kwargs.copy())
        return original_call(fn, *args, **kwargs)
    monkeypatch.setattr(sim, "call", inspect_call)
    monkeypatch.setattr(sim, "evaluation_state", lambda: pytest.fail("Unexpected evaluation oracle"))
    frame = sim.capture_frames()
    assert set(frame) == {"frame_id", "captured_at", "images", "calibrations", "camera_profile", "view_metadata"}
    assert frame['camera_profile']=='legacy_fixed_stereo'
    assert len(frame['view_metadata'])==2
    for view,calibration in zip(frame['view_metadata'],frame['calibrations']):
        assert set(view)=={'camera_id','frame_id','captured_at','role','extrinsics_source','calibration'}
        assert view['camera_id']==calibration['camera_id']
        assert view['frame_id']==frame['frame_id'] and view['captured_at']==frame['captured_at']
        assert view['role']=='observer' and view['extrinsics_source']=='fixed_calibration'
        assert view['calibration']==calibration
    assert isinstance(frame["captured_at"], float) and len(frame["frame_id"]) == 32
    assert len(frame["images"]) == len(frame["calibrations"]) == 2
    assert {image["camera_id"] for image in frame["images"]} == set(CAMERAS)
    assert len(render_calls) == 2
    assert all(call["flags"] == p.ER_NO_SEGMENTATION_MASK for call in render_calls)
    decoded = []
    for record, calibration in zip(frame["images"], frame["calibrations"]):
        assert set(record) == {"camera_id", "width", "height", "data_url"}
        assert record["camera_id"] == calibration["camera_id"]
        assert record["data_url"].startswith("data:image/jpeg;base64,")
        encoded = record["data_url"].split(",", 1)[1]
        image = Image.open(BytesIO(base64.b64decode(encoded, validate=True)))
        assert image.mode == "RGB" and image.size == (640, 480)
        pixels = np.asarray(image)
        assert pixels.dtype == np.uint8 and float(pixels.std()) > 10
        decoded.append(pixels)
        assert calibration == camera_calibration(record["camera_id"])
        transform = np.asarray(calibration["world_from_camera"])
        np.testing.assert_allclose(transform[3], [0, 0, 0, 1])
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        assert np.linalg.det(rotation) == pytest.approx(1)
        eye, target = CAMERAS[record["camera_id"]]
        np.testing.assert_allclose(transform[:3, 3], eye)
        camera_target = np.linalg.inv(transform) @ np.r_[target, 1.]
        assert camera_target[2] > 0
        np.testing.assert_allclose(camera_target[:2], [0, 0], atol=1e-8)
        assert calibration["fx"] == pytest.approx(calibration["fy"])
    assert not np.array_equal(*decoded)
    assert not any(any(term in key.lower() for term in ("depth", "mask", "segmentation", "ground_truth"))
                   for key in all_keys(frame))
    again = sim.capture_frames()
    assert again["frame_id"] != frame["frame_id"] and again["captured_at"] >= frame["captured_at"]


def test_render_camera_selection_and_cache(sim):
    first = sim.render("left")
    assert first[:2] == b"\xff\xd8" and len(first) > 10000
    assert sim.render("left") == first
    assert sim.render("right") != first
    with pytest.raises(ValueError):
        sim.render("depth")


@pytest.mark.parametrize("xyz", [None, [], [.2, 0], [.2, 0, float("nan")],
                                 [.2, float("inf"), .06], [.5, 0, .06],
                                 [.2, 0, -.01], [True, 0, .06], [.24, 0, .06]])
def test_manual_xyz_bounds_reject_without_queueing(sim, xyz):
    sim.resume()
    with pytest.raises(ValueError):
        sim.manual("xyz", xyz)
    assert not sim.queue and sim.current is None and sim.actions == 0


@pytest.mark.parametrize("values", [None, [], [0]*7, [99]*5, [float("nan")]*5,
                                    [float("inf")]*5, ["bad"]*5])
def test_manual_joint_bounds_reject_without_queueing(sim, values):
    sim.resume()
    with pytest.raises((ValueError, TypeError)):
        sim.manual("joints", values)
    assert not sim.queue and sim.current is None and sim.actions == 0


def test_manual_joint_step_bound_and_physical_joint_motion(sim):
    sim.resume()
    q = sim.robot.joint_positions()
    too_far = list(q); too_far[0] += .151
    with pytest.raises(ValueError, match="0.15"):
        sim.manual("joints", too_far)
    goal = list(q); goal[0] += .05
    sim.manual("joints", goal)
    # Enqueueing does not teleport the joint.
    np.testing.assert_allclose(sim.robot.joint_positions(), q, atol=1e-8)
    finish(sim)
    assert sim.robot.joint_positions()[0] == pytest.approx(goal[0], abs=.002)
    assert sim.actions == 1 and sim.steps >= 144


def test_manual_ik_failure_does_not_queue(sim):
    # Place the test fixture at a known valid pose near the top-down reach edge.
    # This explicit test reset is not a production motion path.
    q = sim.robot.solve_ik([.17, 0, .075])
    for joint, angle in zip(sim.robot.arm_joint_indices, q):
        sim.call(p.resetJointState, sim.robot.body_id, joint, angle)
    sim.robot.command_joints(q)
    sim.resume()
    assert math.dist(sim.robot.tcp_pose()[0], [.17, 0, .09]) < .02
    with pytest.raises(ValueError, match="unreachable"):
        sim.manual("xyz", [.17, 0, .09])
    assert not sim.queue and sim.current is None and sim.actions == 0


@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)])
def test_physical_cartesian_jogs_in_world_coordinates(sim, axis, sign):
    sim.resume()
    before = np.asarray(sim.robot.tcp_pose()[0])
    target = before.copy(); target[axis] += .01*sign
    sim.manual("xyz", target.tolist())
    np.testing.assert_allclose(sim.robot.tcp_pose()[0], before, atol=1e-8)
    finish(sim)
    after = np.asarray(sim.robot.tcp_pose()[0])
    np.testing.assert_allclose(after, target, atol=.002)
    assert sign*(after[axis]-before[axis]) > .008
    assert sim.call(p.getNumConstraints) == 0


def test_paused_and_busy_commands_rejected(sim):
    with pytest.raises(ValueError, match="Resume"):
        sim.manual("open")
    sim.resume(); sim.manual("open")
    with pytest.raises(ValueError):
        sim.manual("close")
    sim.tick(1)
    with pytest.raises(ValueError):
        sim.manual("close")
    sim.stop(); sim.resume()
    with pytest.raises(ValueError, match="Unsupported"):
        sim.manual("teleport")


@pytest.mark.parametrize("start_motion", [False, True])
def test_stop_cancels_queued_and_current_arm_commands(sim, start_motion):
    sim.resume(); q = sim.robot.joint_positions(); goal = list(q); goal[0] += .1
    sim.manual("joints", goal)
    if start_motion:
        sim.tick(30)
        assert sim.current is not None
    epoch = sim.epoch
    sim.stop()
    stopped = np.asarray(sim.robot.joint_positions())
    steps = sim.steps
    assert sim.paused and not sim.queue and sim.current is None and sim.epoch == epoch+1
    sim.tick(100)
    assert sim.steps == steps
    np.testing.assert_allclose(sim.robot.joint_positions(), stopped, atol=1e-8)
    sim.resume(); sim.tick(240)
    np.testing.assert_allclose(sim.robot.joint_positions(), stopped, atol=.003)
    assert abs(sim.robot.joint_positions()[0]-goal[0]) > .05


def test_stop_freezes_in_progress_gripper_command(sim):
    sim.resume(); sim.manual("close"); sim.tick(2)
    angle_at_stop = sim.robot.proprioception()["gripper_angle"]
    assert angle_at_stop > sim.robot.gripper_limits[0] + .5
    sim.stop()
    assert not sim.queue and sim.current is None
    sim.resume(); sim.tick(240)
    assert sim.robot.proprioception()["gripper_angle"] == pytest.approx(angle_at_stop, abs=.02)


def test_reset_cancels_all_motion_and_clears_sensor_cache(sim):
    sim.capture_frames()
    assert sim._renders
    sim.resume(); sim.manual("close"); sim.tick(2)
    old_epoch = sim.epoch
    sim.reset(seed=4)
    assert sim.epoch == old_epoch+1 and sim.seed == 4
    assert sim.paused and not sim.queue and sim.current is None
    assert sim.actions == sim.steps == 0 and not sim._renders
    assert sim.proprioception()["gripper_openness"] > .98
    np.testing.assert_allclose(sim.robot.joint_positions(), sim.robot.home_joints, atol=.003)
    sim.resume(); sim.tick(240)
    assert sim.proprioception()["gripper_openness"] > .98
    assert sim.call(p.getNumConstraints) == 0


def test_close_releases_physics_client(sim):
    client = sim.cid
    sim.close()
    assert not p.isConnected(client)
    sim.close()  # Idempotent cleanup.
