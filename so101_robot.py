"""SO-ARM101 robot-only PyBullet adapter. No object state, policy, or grasp hacks.

Angles are radians; positions are meters; gripper openness is in [0, 1].
Constrained IK controls TCP position and its +Z approach axis, leaving roll free.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "so101"
URDF_PATH = ASSET_DIR / "so101.urdf"
ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT_NAME = "gripper"
TCP_LINK_NAME = "sandbox_grasp_center_link"
WORKSPACE = ((0.06, 0.38), (-0.25, 0.25), (0.006, 0.36))


def _finite_vector(value, size, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain {size} finite numbers") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


def _transform(xyz, rpy):
    result = np.eye(4)
    result[:3, :3] = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rpy))).reshape(3, 3)
    result[:3, 3] = xyz
    return result


class SO101Robot:
    """Owns one robot in an existing physics client; caller owns world and stepping."""

    def __init__(self, client_id, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):
        self.client_id = int(client_id)
        self.base_position = _finite_vector(base_position, 3, "Base position")
        orientation = _finite_vector(base_orientation, 4, "Base orientation")
        if abs(float(np.linalg.norm(orientation)) - 1) > 1e-5:
            raise ValueError("Base orientation must be a unit quaternion")
        self.base_transform = np.eye(4)
        self.base_transform[:3, :3] = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        self.base_transform[:3, 3] = self.base_position
        self.body_id = self.call(p.loadURDF, str(URDF_PATH), basePosition=self.base_position,
                                 baseOrientation=orientation, useFixedBase=True,
                                 flags=p.URDF_USE_INERTIA_FROM_FILE)
        infos = [self.call(p.getJointInfo, self.body_id, j)
                 for j in range(self.call(p.getNumJoints, self.body_id))]
        self.joint_indices = {info[1].decode(): info[0] for info in infos}
        self.link_indices = {info[12].decode(): info[0] for info in infos}
        self.link_indices["base_link"] = -1
        self.arm_joint_indices = [self.joint_indices[n] for n in ARM_JOINT_NAMES]
        self.gripper_joint_index = self.joint_indices[GRIPPER_JOINT_NAME]
        self.tcp_link_index = self.link_indices[TCP_LINK_NAME]
        self.upstream_tcp_link_index = self.link_indices["gripper_frame_link"]
        self.lower_limits = np.asarray([infos[j][8] for j in self.arm_joint_indices])
        self.upper_limits = np.asarray([infos[j][9] for j in self.arm_joint_indices])
        self.gripper_limits = (infos[self.gripper_joint_index][8], infos[self.gripper_joint_index][9])
        root = ET.parse(URDF_PATH).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        self._origins = []
        for name in (*ARM_JOINT_NAMES, "gripper_frame_joint"):
            origin = joints[name].find("origin")
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
            self._origins.append(_transform(xyz, rpy))
        self.joint_target = np.zeros(5)
        self.gripper_target = self.gripper_limits[1]
        self.last_ik = None
        home_xyz = self.base_transform[:3, :3] @ np.array([.20, 0, .06]) + self.base_position
        home_approach = self.base_transform[:3, :3] @ np.array([0, 0, -1])
        self.home_joints = self.solve_ik(home_xyz, approach=home_approach, seed=np.zeros(5))
        for j in self.arm_joint_indices + [self.gripper_joint_index]:
            self.call(p.changeDynamics, self.body_id, j, linearDamping=.04, angularDamping=.04,
                      jointDamping=.06, lateralFriction=1.2, spinningFriction=.01)
        for link in ("gripper_link", "moving_jaw_so101_v1_link"):
            self.call(p.changeDynamics, self.body_id, self.link_indices[link], lateralFriction=2.5,
                      spinningFriction=.02, rollingFriction=.001)
        self.reset_home()

    def call(self, fn, *args, **kwargs):
        return fn(*args, physicsClientId=self.client_id, **kwargs)

    def reset_home(self):
        """Explicit robot reset only; never used during motion or IK."""
        self.joint_target = np.asarray(self.home_joints, dtype=float).copy()
        self.gripper_target = self.gripper_limits[1]
        for j, q in zip(self.arm_joint_indices, self.joint_target):
            self.call(p.resetJointState, self.body_id, j, float(q))
        self.call(p.resetJointState, self.body_id, self.gripper_joint_index, self.gripper_target)
        self.apply_motors()

    def joint_positions(self):
        return [float(s[0]) for s in self.call(p.getJointStates, self.body_id, self.arm_joint_indices)]

    def tcp_pose(self):
        state = self.call(p.getLinkState, self.body_id, self.tcp_link_index, computeForwardKinematics=True)
        return list(state[4]), list(state[5])

    def proprioception(self):
        """Default SO101 sensors: encoders and FK only, with no contact queries."""
        states = self.call(p.getJointStates, self.body_id, self.arm_joint_indices + [self.gripper_joint_index])
        position, orientation = self.tcp_pose()
        lower, upper = self.gripper_limits
        return {"name": "SO-ARM101", "arm_dof": 5, "ee": position,
                "joints": [float(s[0]) for s in states[:5]],
                "joint_limits": [[float(lo), float(hi)] for lo, hi in zip(self.lower_limits, self.upper_limits)],
                "robot_model": "SO-ARM101", "joint_names": list(ARM_JOINT_NAMES) + [GRIPPER_JOINT_NAME],
                "joint_positions": [float(s[0]) for s in states],
                "joint_velocities": [float(s[1]) for s in states],
                "tcp_position": position, "tcp_orientation": orientation,
                "gripper_openness": float(np.clip((states[-1][0]-lower)/(upper-lower), 0, 1)),
                "gripper_angle": float(states[-1][0]),
                "angle_unit": "radians", "position_unit": "meters"}

    def evaluation_tactile(self):
        """Optional simulation oracle; default SO101 hardware has no tactile sensor.

        This method is for independent physical-grasp evaluation, not model input.
        It intentionally exposes no object IDs, names, positions or classifications.
        """
        jaws = {}
        for name, link in (("fixed_jaw", self.link_indices["gripper_link"]),
                           ("moving_jaw", self.gripper_joint_index)):
            contacts = self.call(p.getContactPoints, bodyA=self.body_id, linkIndexA=link)
            external = [contact for contact in contacts if contact[2] != self.body_id]
            jaws[name] = {"contact_count": len(external),
                          "total_normal_force": float(sum(max(0., c[9]) for c in external))}
        return {"schema": "so101-evaluation-tactile-v1", "simulation_only": True,
                "not_for_model_input": True, "force_unit": "newtons", "jaws": jaws}

    def forward_kinematics(self, joints, with_jacobian=False):
        """Pure kinematics, preserving the live physics state exactly."""
        q = _finite_vector(joints, 5, "Arm joints")
        t = self.base_transform.copy()
        axes, centers = [], []
        for origin, angle in zip(self._origins[:5], q):
            t = t @ origin
            axes.append(t[:3, 2].copy())
            centers.append(t[:3, 3].copy())
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            t = t @ rotation
        t = t @ self._origins[5]
        t = t @ _transform([-.012, 0, 0], [0, 0, 0])
        if not with_jacobian:
            return t
        jac_pos = np.column_stack([np.cross(axis, t[:3, 3]-center) for axis, center in zip(axes, centers)])
        jac_axis = np.column_stack([np.cross(axis, t[:3, 2]) for axis in axes])
        return t, jac_pos, jac_axis

    def validate_xyz(self, xyz):
        target = _finite_vector(xyz, 3, "XYZ")
        local = self.base_transform[:3, :3].T @ (target-self.base_position)
        if not all(lo <= v <= hi for v, (lo, hi) in zip(local, WORKSPACE)):
            raise ValueError("Target outside SO-ARM101 workspace bounds")
        return target

    def solve_ik(self, xyz, approach=(0, 0, -1), seed=None):
        """Find a bounded pose, rejecting unreachable targets instead of silent clipping.

        The 5-DOF arm cannot realize arbitrary 6D poses. Only TCP +Z direction
        is constrained alongside position; rotation about that axis remains free.
        """
        target = self.validate_xyz(xyz)
        direction = _finite_vector(approach, 3, "Approach")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise ValueError("Approach must be nonzero")
        direction /= norm
        start = _finite_vector(self.joint_target if seed is None else seed, 5, "IK seed")
        seeds = [np.clip(start, self.lower_limits, self.upper_limits)]
        # Deterministic alternate elbow branches for difficult/near-singular poses.
        seeds.extend(np.array([pan, shoulder, elbow, wrist, 0.0]) for pan in (0.0,)
                     for shoulder, elbow, wrist in ((-.6, .8, .8), (.6, -.8, -.8),
                                                   (-1.1, 1.3, 1.1), (1.1, -1.3, -1.1), (0, 0, 0)))
        best = None
        weight = .12
        for initial in seeds:
            q = np.clip(initial, self.lower_limits, self.upper_limits)
            damping = .004
            for _ in range(160):
                t, jp, ja = self.forward_kinematics(q, True)
                residual = np.r_[target-t[:3, 3], weight*(direction-t[:3, 2])]
                error = float(np.linalg.norm(residual))
                if best is None or error < best[0]:
                    best = (error, q.copy(), t.copy())
                pos_error = float(np.linalg.norm(target-t[:3, 3]))
                axis_error = math.acos(float(np.clip(direction @ t[:3, 2], -1, 1)))
                if pos_error < .0005 and axis_error < .02:
                    self.last_ik = {"position_residual_m": pos_error, "approach_residual_rad": axis_error}
                    return q.tolist()
                jac = np.vstack((jp, weight*ja))
                delta = jac.T @ np.linalg.solve(jac @ jac.T + damping*damping*np.eye(6), residual)
                delta *= min(1.0, .25/max(float(np.max(np.abs(delta))), 1e-9))
                candidate = np.clip(q+delta, self.lower_limits, self.upper_limits)
                nt = self.forward_kinematics(candidate)
                new_error = np.linalg.norm(np.r_[target-nt[:3, 3], weight*(direction-nt[:3, 2])])
                if new_error < error:
                    q = candidate
                    damping = max(.0003, damping*.75)
                else:
                    damping = min(.1, damping*2)
                if float(np.linalg.norm(delta)) < 1e-8:
                    break
        _, q, t = best
        pos_error = float(np.linalg.norm(target-t[:3, 3]))
        axis_error = math.acos(float(np.clip(direction @ t[:3, 2], -1, 1)))
        self.last_ik = {"position_residual_m": pos_error, "approach_residual_rad": axis_error}
        if pos_error > .002 or axis_error > .06:
            raise ValueError(f"SO-ARM101 target unreachable: position residual {pos_error:.4f} m, approach residual {axis_error:.3f} rad")
        return q.tolist()

    def command_joints(self, joints, max_step=None):
        target = _finite_vector(joints, 5, "Arm joints")
        if np.any(target < self.lower_limits) or np.any(target > self.upper_limits):
            raise ValueError("Arm joint target exceeds SO-ARM101 limits")
        if max_step is not None:
            if not math.isfinite(max_step) or max_step <= 0:
                raise ValueError("Joint step bound must be positive")
            if np.max(np.abs(target-np.asarray(self.joint_positions()))) > max_step:
                raise ValueError("Arm joint command exceeds the bounded step")
        self.joint_target = target.copy()
        self.apply_motors()
        return self.joint_target.tolist()

    def command_cartesian(self, xyz, max_step=.05, approach=(0, 0, -1)):
        target = self.validate_xyz(xyz)
        if not math.isfinite(max_step) or max_step <= 0:
            raise ValueError("Cartesian step bound must be positive")
        if math.dist(self.tcp_pose()[0], target) > max_step + 1e-9:
            raise ValueError("Cartesian command exceeds the bounded step")
        return self.command_joints(self.solve_ik(target, approach=approach, seed=self.joint_positions()))

    def command_gripper(self, openness):
        if not isinstance(openness, (int, float)) or not math.isfinite(openness) or not 0 <= openness <= 1:
            raise ValueError("Gripper openness must be a finite number from 0 to 1")
        lo, hi = self.gripper_limits
        self.gripper_target = lo + float(openness)*(hi-lo)
        self.apply_motors()
        return self.gripper_target

    def apply_motors(self):
        self.call(p.setJointMotorControlArray, self.body_id, self.arm_joint_indices, p.POSITION_CONTROL,
                  targetPositions=self.joint_target.tolist(), forces=[3.35]*5,
                  positionGains=[.2]*5, velocityGains=[1]*5)
        self.call(p.setJointMotorControl2, self.body_id, self.gripper_joint_index, p.POSITION_CONTROL,
                  targetPosition=float(self.gripper_target), force=1.0, positionGain=.12, velocityGain=1)
