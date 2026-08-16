from __future__ import annotations

import numpy as np

from ..input.clutch import Pose
from .base import RobotAdapter


class FrankaAdapter(RobotAdapter):
    _GRASP_THRESHOLD = 0.70
    _RELEASE_THRESHOLD = 0.20

    def __init__(self, config):
        self.config = config
        self._robot = None
        self._ee = None
        self._arm = []
        self._fingers = []
        self._last_arm = None
        self._finger_target = None
        self._grasp_closed = False

    @property
    def entity(self):
        return self._robot

    @property
    def ee_link(self):
        return self._ee

    @property
    def arm_dofs_idx(self):
        return self._arm

    @property
    def finger_dofs_idx(self):
        return self._fingers

    def build(self, scene):
        import genesis as gs

        self._robot = scene.add_entity(
            gs.morphs.MJCF(
                file=self.config["robot"]["mjcf_file"],
                pos=tuple(self.config["robot"].get("base_position", (0.0, 0.0, 0.0))),
                euler=tuple(self.config["robot"].get("base_euler_deg", (0.0, 0.0, 0.0))),
            )
        )

    def initialize_after_scene_build(self):
        r = self._robot
        rc = self.config["robot"]
        self._arm = [r.get_joint(n).dofs_idx_local[0] for n in rc["arm_joint_names"]]
        self._fingers = [
            r.get_joint(n).dofs_idx_local[0] for n in rc["finger_joint_names"]
        ]
        self._ee = r.get_link(rc["end_effector_link"])
        home = np.array(rc["home_qpos"])
        if len(home) == r.n_qs:
            r.set_qpos(home, zero_velocity=True)
        self._last_arm = r.get_dofs_position(self._arm).cpu().numpy()
        r.control_dofs_position(self._last_arm, dofs_idx_local=self._arm)
        gripper = self.config["gripper"]
        r.set_dofs_kp(
            np.full(2, gripper["kp"]),
            dofs_idx_local=self._fingers,
        )
        r.set_dofs_kv(
            np.full(2, gripper["kv"]),
            dofs_idx_local=self._fingers,
        )
        r.set_dofs_force_range(
            lower=np.full(2, -gripper["force_limit"]),
            upper=np.full(2, gripper["force_limit"]),
            dofs_idx_local=self._fingers,
        )
        for link_name in ("left_finger", "right_finger"):
            r.get_link(link_name).set_friction(gripper["finger_friction"])
        self.apply_gripper_trigger(0)

    def get_ee_pose(self):
        return Pose(self._ee.get_pos().cpu().numpy(), self._ee.get_quat().cpu().numpy())

    def apply_arm_position(self, q_arm):
        self._last_arm = np.asarray(q_arm)
        self._robot.control_dofs_position(self._last_arm, dofs_idx_local=self._arm)

    def apply_gripper_trigger(self, trigger):
        c = self.config["gripper"]
        trigger = float(trigger)
        if self._grasp_closed:
            if trigger <= self._RELEASE_THRESHOLD:
                self._grasp_closed = False
        elif trigger >= self._GRASP_THRESHOLD:
            self._grasp_closed = True

        self._finger_target = (
            c["closed_position"]
            if self._grasp_closed
            else c["open_position"]
        )
        self._robot.control_dofs_position(
            np.full(2, self._finger_target), dofs_idx_local=self._fingers
        )
        return float(self._finger_target)

    def get_finger_positions(self):
        return self._robot.get_dofs_position(self._fingers).cpu().numpy()

    def hold(self):
        if self._last_arm is not None:
            self._robot.control_dofs_position(self._last_arm, dofs_idx_local=self._arm)
        if self._finger_target is not None:
            self._robot.control_dofs_position(
                np.full(2, self._finger_target), dofs_idx_local=self._fingers
            )
