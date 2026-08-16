from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from ..config import resolve_project_path
from ..input.clutch import Pose
from .base import RobotAdapter


class OpenArmAdapter(RobotAdapter):
    """One bimanual OpenArm entity with independently controllable arms."""

    _SIDES = ("left", "right")

    def __init__(self, config):
        self.config = config
        self.urdf_path = resolve_project_path(config["robot"]["urdf_file"])
        self._robot = None
        self._ee, self._arm_dofs = {}, {}
        self._gripper_driver, self._gripper_mimic = {}, {}
        self._last_arm_command, self._finger_target, self._grasp_closed = {}, {}, {}

    @property
    def entity(self):
        return self._robot

    @property
    def arm_names(self):
        return self._SIDES

    def build(self, scene):
        import genesis as gs

        package = self.urdf_path.parent / "openarm_description"
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"OpenArm URDF is missing: {self.urdf_path}")
        if not package.is_symlink() or not package.exists():
            raise FileNotFoundError(
                f"OpenArm package-resolution symlink is missing or broken: {package}"
            )
        self._robot = scene.add_entity(gs.morphs.URDF(
            file=str(self.urdf_path), fixed=True, merge_fixed_links=True,
            links_to_keep=("openarm_left_hand", "openarm_right_hand"),
            recompute_inertia=False,
        ))

    def initialize_after_scene_build(self):
        robot, rc, grip = self._robot, self.config["robot"], self.config["gripper"]
        all_arm_dofs = []
        for arm in self._SIDES:
            names = [f"openarm_{arm}_joint{i}" for i in range(1, 8)]
            dofs = [robot.get_joint(name).dofs_idx_local[0] for name in names]
            if len(dofs) != 7:
                raise RuntimeError(f"OpenArm {arm} arm must resolve exactly 7 DOFs")
            self._arm_dofs[arm] = dofs
            all_arm_dofs.extend(dofs)
            self._ee[arm] = robot.get_link(f"openarm_{arm}_hand")
            self._gripper_driver[arm] = robot.get_joint(
                f"openarm_{arm}_finger_joint1"
            ).dofs_idx_local[0]
            self._gripper_mimic[arm] = robot.get_joint(
                f"openarm_{arm}_finger_joint2"
            ).dofs_idx_local[0]
            robot.set_dofs_kp(np.asarray(rc["arm_kp"]), dofs_idx_local=dofs)
            robot.set_dofs_kv(np.asarray(rc["arm_kv"]), dofs_idx_local=dofs)
            self._preserve_effort_limits(arm, names, dofs)
            robot.set_dofs_kp(np.array([grip["kp"]]), dofs_idx_local=[self._gripper_driver[arm]])
            robot.set_dofs_kv(np.array([grip["kv"]]), dofs_idx_local=[self._gripper_driver[arm]])
            for finger in (f"openarm_{arm}_left_finger", f"openarm_{arm}_right_finger"):
                robot.get_link(finger).set_friction(grip["finger_friction"])
            robot.set_dofs_position(np.asarray(rc["home"][arm]), dofs_idx_local=dofs)
            self._grasp_closed[arm] = False
            self._finger_target[arm] = float(grip["open_position"])
            robot.control_dofs_position(np.array([self._finger_target[arm]]), dofs_idx_local=[self._gripper_driver[arm]])
        if len(set(all_arm_dofs)) != len(all_arm_dofs):
            raise RuntimeError("OpenArm left/right arm DOFs overlap")

    def capture_hold_targets(self):
        for arm in self._SIDES:
            self._last_arm_command[arm] = self._robot.get_dofs_position(self._arm_dofs[arm]).cpu().numpy()

    def _preserve_effort_limits(self, arm, names, dofs):
        lower, upper = (
            x.cpu().numpy() for x in self._robot.get_dofs_force_range(dofs)
        )
        if np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(upper > lower):
            return
        limits = {j.attrib["name"]: float(j.find("limit").attrib["effort"])
                  for j in ET.parse(self.urdf_path).getroot().findall("joint")
                  if j.find("limit") is not None and j.find("limit").get("effort")}
        magnitude = np.asarray([limits[name] for name in names])
        self._robot.set_dofs_force_range(-magnitude, magnitude, dofs_idx_local=dofs)

    def get_ee_link(self, arm): return self._ee[arm]
    def get_arm_dofs_idx(self, arm): return self._arm_dofs[arm]
    def get_finger_dofs_idx(self, arm): return [self._gripper_driver[arm], self._gripper_mimic[arm]]
    def get_ee_pose(self, arm):
        ee = self._ee[arm]
        return Pose(ee.get_pos().cpu().numpy(), ee.get_quat().cpu().numpy())
    def apply_arm_position(self, arm, q_arm):
        self._last_arm_command[arm] = np.asarray(q_arm, dtype=float)
        self._robot.control_dofs_position(self._last_arm_command[arm], dofs_idx_local=self._arm_dofs[arm])
    def apply_gripper_trigger(self, arm, trigger):
        grip = self.config["gripper"]
        if self._grasp_closed[arm]:
            if float(trigger) <= grip["grasp_release_threshold"]: self._grasp_closed[arm] = False
        elif float(trigger) >= grip["grasp_engage_threshold"]: self._grasp_closed[arm] = True
        self._finger_target[arm] = float(grip["closed_position"] if self._grasp_closed[arm] else grip["open_position"])
        self._robot.control_dofs_position(np.array([self._finger_target[arm]]), dofs_idx_local=[self._gripper_driver[arm]])
        return self._finger_target[arm]
    def get_finger_positions(self, arm):
        return self._robot.get_dofs_position(self.get_finger_dofs_idx(arm)).cpu().numpy()
    def hold_arm(self, arm):
        if arm in self._last_arm_command:
            self._robot.control_dofs_position(self._last_arm_command[arm], dofs_idx_local=self._arm_dofs[arm])
        self._robot.control_dofs_position(np.array([self._finger_target[arm]]), dofs_idx_local=[self._gripper_driver[arm]])
