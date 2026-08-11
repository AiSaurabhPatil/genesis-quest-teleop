from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import normalize_quat_wxyz, quat_inverse_wxyz, quat_multiply_wxyz


@dataclass
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass
class ClutchOutput:
    engaged: bool
    target_pose: Pose | None
    just_engaged: bool
    just_released: bool
    requires_rearm: bool


class ClutchController:
    def __init__(self, config):
        t = config["teleop"]
        self.engage = t["clutch_engage_threshold"]
        self.release = t["clutch_release_threshold"]
        self.scale = t["translation_scale"]
        self.engaged = False
        self.requires_rearm = False
        self.controller_anchor = None
        self.robot_anchor = None

    def force_hold(self):
        self.engaged = False
        self.requires_rearm = True
        self.controller_anchor = self.robot_anchor = None

    def update(self, controller_pose, squeeze, measured_ee_pose):
        if self.requires_rearm:
            if squeeze <= self.release:
                self.requires_rearm = False
            return ClutchOutput(False, None, False, False, self.requires_rearm)
        if self.engaged and controller_pose is None:
            self.force_hold()
            return ClutchOutput(False, None, False, True, True)
        if not self.engaged:
            if controller_pose is not None and squeeze >= self.engage:
                self.engaged = True
                self.controller_anchor = controller_pose
                self.robot_anchor = measured_ee_pose
                return ClutchOutput(True, measured_ee_pose, True, False, False)
            return ClutchOutput(False, None, False, False, False)
        if squeeze <= self.release:
            self.engaged = False
            self.controller_anchor = self.robot_anchor = None
            return ClutchOutput(False, None, False, True, False)
        delta = quat_multiply_wxyz(
            controller_pose.quaternion_wxyz,
            quat_inverse_wxyz(self.controller_anchor.quaternion_wxyz),
        )
        target = Pose(
            self.robot_anchor.position
            + self.scale * (controller_pose.position - self.controller_anchor.position),
            normalize_quat_wxyz(
                quat_multiply_wxyz(delta, self.robot_anchor.quaternion_wxyz)
            ),
        )
        return ClutchOutput(True, target, False, False, False)
