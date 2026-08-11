from __future__ import annotations

import time

import numpy as np

from .control.diffik import DifferentialIKController
from .diagnostics import Diagnostics
from .input.clutch import ClutchController, Pose
from .input.frames import (
    map_webxr_position_to_genesis,
    map_webxr_quat_to_genesis_wxyz,
    quat_angle_wxyz,
    slerp_wxyz,
)
from .robots.franka import FrankaAdapter
from .state_store import LatestQuestStateStore
from .transport.local_ipc import LocalStateReceiver
from .transport.webrtc_server import WebRTCServer


class GenesisTeleopApp:
    def __init__(self, config, external_ingress=False):
        self.config = config
        self.store = LatestQuestStateStore()
        self.diagnostics = Diagnostics(config)
        self.external_ingress = external_ingress
        self.server = None
        self.local_receiver = None
        if external_ingress:
            self.local_receiver = LocalStateReceiver(
                config, self.store, self.diagnostics
            )
        else:
            self.server = WebRTCServer(config, self.store, self.diagnostics)
        self._hold_reason = None

    def setup(self):
        import genesis as gs

        if self.external_ingress:
            self.local_receiver.start()
        else:
            self.server.start_in_thread()
        s = self.config["sim"]
        gs.init(backend=gs.cuda if s["backend"] == "gpu" else gs.cpu)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=s["dt"]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.4, -1.4, 1.1),
                camera_lookat=(0.45, 0, 0.35),
                camera_fov=40,
            ),
            show_viewer=s["show_viewer"],
            show_FPS=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.grasp_cube = None
        cube = self.config.get("scene", {}).get("grasp_cube", {})
        if cube.get("enabled", False):
            self.grasp_cube = self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=tuple(cube["position"]),
                    size=tuple(cube["size"]),
                ),
                material=gs.materials.Rigid(
                    friction=cube["friction"], rho=cube["density"]
                ),
                surface=gs.surfaces.Plastic(color=tuple(cube["color"])),
            )
        self.robot = FrankaAdapter(self.config)
        self.robot.build(self.scene)
        self.scene.build()
        self.robot.initialize_after_scene_build()
        for _ in range(s["warmup_steps"]):
            self.scene.step()
        self.diffik = DifferentialIKController(
            self.robot.entity, self.robot.ee_link, self.robot.arm_dofs_idx, self.config
        )
        self.clutch = ClutchController(self.config)
        # A simulator/scene restart must never inherit an already-held squeeze.
        # Require one release before the new scene can engage the arm.
        self.clutch.force_hold()
        self.last_target = self.robot.get_ee_pose()
        self.debug_target = None
        if self.config["diagnostics"]["debug_target_frame"]:
            import genesis.utils.geom as gu

            self.debug_target = self.scene.draw_debug_frame(
                T=gu.trans_quat_to_T(
                    self.last_target.position, self.last_target.quaternion_wxyz
                ),
                axis_length=0.12,
                origin_size=0.008,
                axis_radius=0.005,
            )
        self.divisor = max(1, round((1 / s["dt"]) / s["control_hz"]))

    def _enter_hold(self, reason):
        self.clutch.force_hold()
        self.robot.hold()
        if reason != self._hold_reason:
            self.diagnostics.increment(
                {"stale": "stale_holds", "tracking": "tracking_loss_holds"}.get(
                    reason, "peer_disconnects"
                )
            )
        self._hold_reason = reason

    def _safe_target(self, p):
        t = self.config["teleop"]
        ws = t["workspace"]
        candidate = Pose(
            np.clip(p.position, [ws[x][0] for x in "xyz"], [ws[x][1] for x in "xyz"]),
            p.quaternion_wxyz,
        )
        delta = candidate.position - self.last_target.position
        dist = np.linalg.norm(delta)
        limit = t["max_target_translation_step_m"]
        if dist > limit:
            candidate.position = self.last_target.position + delta * limit / dist
            self.diagnostics.increment("safety_clamps")
        angle = quat_angle_wxyz(
            self.last_target.quaternion_wxyz, candidate.quaternion_wxyz
        )
        if angle > t["max_target_rotation_step_rad"]:
            candidate.quaternion_wxyz = slerp_wxyz(
                self.last_target.quaternion_wxyz,
                candidate.quaternion_wxyz,
                t["max_target_rotation_step_rad"] / angle,
            )
            self.diagnostics.increment("safety_clamps")
        self.last_target = candidate
        if self.debug_target is not None:
            import genesis.utils.geom as gu

            self.scene.update_debug_objects(
                (self.debug_target,),
                (gu.trans_quat_to_T(candidate.position, candidate.quaternion_wxyz),),
            )
        return candidate

    def run(self):
        idx = 0
        while True:
            if idx % self.divisor == 0:
                self._control()
            self.diagnostics.maybe_log()
            self.scene.step()
            idx += 1

    def _control(self):
        state = self.store.snapshot()
        t = self.config["teleop"]
        if state is None or not self.store.is_connected():
            return self._enter_hold("disconnected")
        age = (time.monotonic_ns() - state.receive_monotonic_ns) / 1e6
        self.diagnostics.set_value("packet_age_ms", age)
        if age > t["stale_timeout_ms"]:
            return self._enter_hold("stale")
        ctrl = state.packet.controllers.get(t["active_hand"])
        if ctrl is None or ctrl.position_xyz is None or ctrl.orientation_xyzw is None:
            return self._enter_hold("tracking")
        pose = Pose(
            map_webxr_position_to_genesis(ctrl.position_xyz),
            map_webxr_quat_to_genesis_wxyz(ctrl.orientation_xyzw),
        )
        out = self.clutch.update(pose, ctrl.squeeze, self.robot.get_ee_pose())
        finger_target = self.robot.apply_gripper_trigger(ctrl.trigger)
        self.diagnostics.set_value("trigger", round(ctrl.trigger, 3))
        self.diagnostics.set_value("finger_target", round(finger_target, 4))
        self.diagnostics.set_value(
            "finger_measured",
            np.round(self.robot.get_finger_positions(), 4).tolist(),
        )
        if out.just_engaged:
            self.diagnostics.increment("clutch_engages")
        if out.just_released:
            self.diagnostics.increment("clutch_releases")
        if out.engaged and out.target_pose:
            target = self._safe_target(out.target_pose)
            cmd = self.diffik.compute_command(target.position, target.quaternion_wxyz)
            if cmd is None:
                self.diagnostics.increment("diffik_failures")
                self.robot.hold()
            else:
                self.robot.apply_arm_position(cmd)
                self.diagnostics.increment("control_updates")
        else:
            self.robot.hold()
        self._hold_reason = None

    def shutdown(self):
        if self.local_receiver:
            self.local_receiver.stop()
        if self.server:
            self.server.stop()
