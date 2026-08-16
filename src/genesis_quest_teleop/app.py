from __future__ import annotations

import time

import gs_nyx.nyx_py_renderer as npr
import numpy as np
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

from .control.diffik import DifferentialIKController
from .diagnostics import Diagnostics
from .input.clutch import ClutchController, Pose
from .input.frames import (
    map_webxr_position_to_genesis,
    map_webxr_quat_to_genesis_wxyz,
    quat_angle_wxyz,
    slerp_wxyz,
)
from .robots import create_robot_adapter
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
        self.arm_hold_reason = {}
        self.wrist_camera = None
        self.nyx_render_divisor = None
        if external_ingress:
            self.local_receiver = LocalStateReceiver(
                config, self.store, self.diagnostics
            )
        else:
            self.server = WebRTCServer(config, self.store, self.diagnostics)

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
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
                noslip_iterations=5,
                noslip_tolerance=1e-6,
            ),
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
            pbr = cube.get("pbr", {})
            self.grasp_cube = self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=tuple(cube["position"]),
                    size=tuple(cube["size"]),
                ),
                material=gs.materials.Rigid(
                    friction=cube["friction"], rho=cube["density"]
                ),
                surface=gs.surfaces.Default(
                    color=tuple(cube["color"]),
                    metallic=float(pbr.get("metallic", 0.0)),
                    roughness=float(pbr.get("roughness", 0.5)),
                ),
            )
        self.robot = create_robot_adapter(self.config)
        self.robot.build(self.scene)
        # ---------------------------------------------------------
        # Nyx wrist camera
        # ---------------------------------------------------------
        nyx_cfg = self.config.get("nyx_camera", {})

        if nyx_cfg.get("enabled", False):
            parent_link = self.robot.entity.get_link(nyx_cfg["parent_link"])

            offset_T = np.asarray(
                nyx_cfg["offset_T"],
                dtype=np.float64,
            )

            self.wrist_camera = self.scene.add_sensor(
                NyxCameraOptions(
                    res=tuple(nyx_cfg["resolution"]),
                    fov=float(nyx_cfg["fov"]),
                    near=float(nyx_cfg["near"]),
                    far=float(nyx_cfg["far"]),

                    # Attach camera to Franka wrist.
                    entity_idx=self.robot.entity.idx,
                    link_idx_local=parent_link.idx_local,
                    offset_T=offset_T,

                    # Rendering quality.
                    spp=int(nyx_cfg["spp"]),
                    render_mode=npr.ERenderMode.FastPathTracer,

                    # Native Nyx live preview.
                    open_window=bool(nyx_cfg["open_window"]),

                    # Simple scene illumination.
                    lights=[
                        {
                            "type": "directional",
                            "dir": (-0.4, -0.4, -0.8),
                            "color": (1.0, 1.0, 1.0),
                            "intensity": 5.0,
                            "shadow": True,
                        }
                    ],
                )
            )

        self.scene.build()
        self.robot.initialize_after_scene_build()
        for _ in range(s["warmup_steps"]):
            self.scene.step()
        if hasattr(self.robot, "capture_hold_targets"):
            self.robot.capture_hold_targets()
        physics_dt = s["dt"]
        physics_hz = 1.0 / physics_dt
        self.divisor = max(1, round(physics_hz / s["control_hz"]))
        self.control_dt = self.divisor * physics_dt
        self.diffik, self.clutch, self.last_target, self.debug_target = {}, {}, {}, {}
        for arm in self.robot.arm_names:
            self.diffik[arm] = DifferentialIKController(self.robot.entity, self.robot.get_ee_link(arm), self.robot.get_arm_dofs_idx(arm), self.config, control_dt=self.control_dt, joint_velocity_limits=self.robot.get_arm_velocity_limits(arm))
            self.diffik[arm].reset_command_state()
            self.clutch[arm] = ClutchController(self.config)
            self.clutch[arm].force_hold()
            self.last_target[arm] = self.robot.get_ee_pose(arm)
            self.debug_target[arm] = None
            if self.config["diagnostics"]["debug_target_frame"]:
                import genesis.utils.geom as gu
                p = self.last_target[arm]
                self.debug_target[arm] = self.scene.draw_debug_frame(T=gu.trans_quat_to_T(p.position, p.quaternion_wxyz), axis_length=0.12, origin_size=0.008, axis_radius=0.005)
        if self.wrist_camera is not None:
            physics_hz = 1.0 / s["dt"]
            nyx_hz = float(self.config["nyx_camera"]["render_hz"])

            self.nyx_render_divisor = max(
                1,
                round(physics_hz / nyx_hz),
            )

    def _enter_arm_hold(self, arm, reason):
        self.clutch[arm].force_hold()
        if self.arm_hold_reason.get(arm) is None:
            self.robot.enter_arm_hold(arm)
        else:
            self.robot.maintain_arm_hold(arm)
        self.diffik[arm].reset_command_state()
        if reason != self.arm_hold_reason.get(arm):
            self.diagnostics.increment(f"{arm}_{'tracking_loss_holds' if reason == 'tracking' else 'diffik_failures' if reason == 'diffik_failure' else 'holds'}")
        self.arm_hold_reason[arm] = reason

    def _enter_all_holds(self, reason):
        for arm in self.robot.arm_names:
            self._enter_arm_hold(arm, reason)
        self.diagnostics.increment({"stale": "stale_holds", "disconnected": "peer_disconnects"}.get(reason, "holds"))

    def _safe_target(self, arm, p):
        t = self.config["teleop"]
        ws = t["workspace"] if "x" in t["workspace"] else t["workspace"][arm]
        candidate = Pose(
            np.clip(p.position, [ws[x][0] for x in "xyz"], [ws[x][1] for x in "xyz"]),
            p.quaternion_wxyz,
        )
        delta = candidate.position - self.last_target[arm].position
        dist = np.linalg.norm(delta)
        limit = t["max_target_translation_step_m"]
        if dist > limit:
            candidate.position = self.last_target[arm].position + delta * limit / dist
            self.diagnostics.increment("safety_clamps")
        angle = quat_angle_wxyz(
            self.last_target[arm].quaternion_wxyz, candidate.quaternion_wxyz
        )
        if angle > t["max_target_rotation_step_rad"]:
            candidate.quaternion_wxyz = slerp_wxyz(
                self.last_target[arm].quaternion_wxyz,
                candidate.quaternion_wxyz,
                t["max_target_rotation_step_rad"] / angle,
            )
            self.diagnostics.increment("safety_clamps")
        self.last_target[arm] = candidate
        if self.debug_target[arm] is not None:
            import genesis.utils.geom as gu

            self.scene.update_debug_objects(
                (self.debug_target[arm],),
                (gu.trans_quat_to_T(candidate.position, candidate.quaternion_wxyz),),
            )
        return candidate

    def run(self):
        idx = 0

        while True:
            if idx % self.divisor == 0:
                self._control()

            self.diagnostics.maybe_log()

            # Physics stays independent of Nyx rendering.
            self.scene.step()

            # Nyx uses render-on-read. Rendering only here prevents
            # the path tracer from running at the 120 Hz physics rate.
            if (
                self.wrist_camera is not None
                and self.nyx_render_divisor is not None
                and idx % self.nyx_render_divisor == 0
            ):
                self.wrist_camera.read()

            idx += 1

    def _control(self):
        state = self.store.snapshot()
        t = self.config["teleop"]
        if state is None or not self.store.is_connected():
            return self._enter_all_holds("disconnected")
        age = (time.monotonic_ns() - state.receive_monotonic_ns) / 1e6
        self.diagnostics.set_value("packet_age_ms", age)
        if age > t["stale_timeout_ms"]:
            return self._enter_all_holds("stale")
        commands = {}
        # Do not construct the Franka fallback eagerly: OpenArm configs do
        # not carry the legacy ``active_hand`` key.
        bindings = t["arm_bindings"] if "arm_bindings" in t else {
            "primary": t["active_hand"]
        }
        for arm in self.robot.arm_names:
            ctrl = state.packet.controllers.get(bindings[arm])
            if ctrl is None or ctrl.position_xyz is None or ctrl.orientation_xyzw is None:
                self._enter_arm_hold(arm, "tracking")
                continue
            pose = Pose(map_webxr_position_to_genesis(ctrl.position_xyz), map_webxr_quat_to_genesis_wxyz(ctrl.orientation_xyzw))
            finger_target = self.robot.apply_gripper_trigger(arm, ctrl.trigger)
            self.diagnostics.set_value(f"{arm}_trigger", round(ctrl.trigger, 3))
            self.diagnostics.set_value(f"{arm}_finger_target", round(finger_target, 4))
            self.diagnostics.set_value(f"{arm}_finger_measured", np.round(self.robot.get_finger_positions(arm), 4).tolist())
            out = self.clutch[arm].update(pose, ctrl.squeeze, self.robot.get_ee_pose(arm))
            if out.just_engaged:
                measured = self.robot.entity.get_dofs_position(self.robot.get_arm_dofs_idx(arm)).cpu().numpy()
                self.diffik[arm].reset_command_state(measured)
                self.last_target[arm] = self.robot.get_ee_pose(arm)
                self.diagnostics.increment(f"{arm}_clutch_engages")
            if out.just_released:
                self.robot.enter_arm_hold(arm)
                self.diffik[arm].reset_command_state()
                self.arm_hold_reason[arm] = "clutch"
                self.diagnostics.increment(f"{arm}_clutch_releases")
            if not out.engaged or not out.target_pose:
                self.robot.maintain_arm_hold(arm)
                continue
            target = self._safe_target(arm, out.target_pose)
            command = self.diffik[arm].compute_command(target.position, target.quaternion_wxyz)
            if command is None:
                self._enter_arm_hold(arm, "diffik_failure")
                continue
            commands[arm] = command
            self.diagnostics.increment(f"{arm}_control_updates")
            singular = self.diffik[arm].last_singular_values
            if singular is not None: self.diagnostics.set_value(f"{arm}_min_singular_value", float(np.min(singular)))
            raw = self.diffik[arm].last_raw_dq
            clipped = self.diffik[arm].last_clipped_dq
            if raw is not None:
                self.diagnostics.set_value(f"{arm}_max_raw_dq", float(np.max(np.abs(raw))))
            if clipped is not None:
                self.diagnostics.set_value(f"{arm}_max_clipped_dq", float(np.max(np.abs(clipped))))
            controller = self.diffik[arm]
            self.diagnostics.set_value(f"{arm}_position_error_m", float(np.linalg.norm(target.position - self.robot.get_ee_pose(arm).position)))
            self.diagnostics.set_value(f"{arm}_rotation_error_rad", float(np.linalg.norm(controller.last_error[3:] / controller.rotation_rate_gain)) if controller.mode == "resolved_rate" else float(np.linalg.norm(controller.last_error[3:])))
            if controller.mode == "resolved_rate":
                self.diagnostics.set_value(f"{arm}_max_qdot_raw", float(np.max(np.abs(controller.last_raw_dq))))
                self.diagnostics.set_value(f"{arm}_max_qdot_command", float(np.max(np.abs(controller.last_qdot_command))))
                self.diagnostics.set_value(f"{arm}_max_position_lead", float(np.max(np.abs(controller.last_position_lead))))
                self.diagnostics.set_value(f"{arm}_velocity_limited", controller.velocity_limited)
                self.diagnostics.set_value(f"{arm}_acceleration_limited", controller.acceleration_limited)
                self.diagnostics.set_value(f"{arm}_joint_limit_clamped", controller.joint_limit_clamped)
            self.arm_hold_reason[arm] = None
        for arm, command in commands.items():
            self.robot.apply_arm_command(arm, command.position, command.velocity)
            dofs = self.robot.get_arm_dofs_idx(arm)
            q = self.robot.entity.get_dofs_position(dofs).cpu().numpy()
            qdot = self.robot.entity.get_dofs_velocity(dofs).cpu().numpy()
            force = self.robot.entity.get_dofs_control_force(dofs).cpu().numpy()
            lower, upper = (x.cpu().numpy() for x in self.robot.entity.get_dofs_force_range(dofs))
            limits = np.maximum(np.abs(lower), np.abs(upper))
            utilization = np.divide(np.abs(force), limits, out=np.zeros_like(force), where=limits > 0)
            self.diagnostics.set_value(f"{arm}_max_joint_velocity_measured", float(np.max(np.abs(qdot))))
            self.diagnostics.set_value(f"{arm}_max_control_force_utilization", float(np.max(utilization)))
            self.diagnostics.set_value(f"{arm}_max_joint_position_tracking_error", float(np.max(np.abs(command.position - q))))
            if command.velocity is not None:
                self.diagnostics.set_value(f"{arm}_max_joint_velocity_tracking_error", float(np.max(np.abs(command.velocity - qdot))))

    def shutdown(self):
        if self.local_receiver:
            self.local_receiver.stop()
        if self.server:
            self.server.stop()
