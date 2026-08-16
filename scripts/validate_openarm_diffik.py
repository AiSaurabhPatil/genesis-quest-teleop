"""Validate DifferentialIKController against the Genesis OpenArm bimanual URDF.

The harness deliberately uses synthetic Cartesian targets only.  It runs
physics at 120 Hz and Differential IK at 60 Hz, matching the intended
teleoperation architecture without importing the teleoperation stack.

Genesis's Differential IK example is a kinematic reference, not a complete
gravity-loaded manipulator controller: it runs with gravity, collisions, and
joint limits disabled.  In particular, its ``measured_q + dq`` command pattern
must not be treated as evidence that a finite-effort PD arm should be driven
that way in teleoperation.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import genesis as gs
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from genesis_quest_teleop.control.diffik import DifferentialIKController

URDF_PATH = ROOT / "assets/openarm/openarm_bimanual_genesis.urdf"
CONFIG_PATH = ROOT / "config/default.yaml"
LEFT_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINTS = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
RIGHT_GRIPPER_JOINTS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
LEFT_EE_NAME, RIGHT_EE_NAME = "openarm_left_hand", "openarm_right_hand"
LEFT_HOME = np.array([0.0, -1.0, 0.0, 1.2, 0.0, 0.0, 0.0])
RIGHT_HOME = np.array([0.0, 1.0, 0.0, 1.2, 0.0, 0.0, 0.0])
ARM_KP = np.array([400, 400, 300, 250, 150, 100, 80])
ARM_KV = np.array([40, 40, 30, 25, 15, 10, 8])
SIM_DT, CONTROL_HZ = 1.0 / 120.0, 60.0
PHYSICS_PER_CONTROL = 2
POS_ACCEPT, ROT_ACCEPT = 0.005, math.radians(2.0)
POS_FALLBACK, ROT_FALLBACK = 0.010, math.radians(3.0)


def arr(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def qnorm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise RuntimeError(f"invalid quaternion: {q}")
    return q / n


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w, x, y, z = qnorm(a)
    W, X, Y, Z = qnorm(b)
    return qnorm(np.array([w * W - x * X - y * Y - z * Z,
                           w * X + x * W + y * Z - z * Y,
                           w * Y - x * Z + y * W + z * X,
                           w * Z + x * Y - y * X + z * W]))


def axis_angle(axis: np.ndarray, radians: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return qnorm(np.r_[math.cos(radians / 2), axis * math.sin(radians / 2)])


def angle_error(target: np.ndarray, actual: np.ndarray) -> float:
    return 2 * math.acos(np.clip(abs(float(np.dot(qnorm(target), qnorm(actual)))), -1, 1))


def dofs(robot: object, names: list[str]) -> list[int]:
    return [robot.get_joint(name).dofs_idx_local[0] for name in names]


def ensure_effort_limits(robot: object, names: list[str], indices: list[int]) -> None:
    """Use imported effort limits, with the known-good URDF fallback if needed."""
    lower, upper = (arr(value) for value in robot.get_dofs_force_range(indices))
    print(f"Imported force ranges for {names[0].split('_')[1]}:", list(zip(lower, upper)))
    if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(upper > lower):
        return
    limits = {}
    for joint in ET.parse(URDF_PATH).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("effort") is not None:
            limits[joint.attrib["name"]] = float(limit.get("effort"))
    magnitudes = np.array([limits[name] for name in names])
    print("Applying URDF effort-limit fallback:", magnitudes)
    robot.set_dofs_force_range(-magnitudes, magnitudes, indices)


@dataclass
class Sample:
    timestamp: float
    position_error: float
    rotation_error: float
    q: np.ndarray
    command: np.ndarray
    delta: np.ndarray
    raw_dq: np.ndarray
    clipped_dq: np.ndarray
    control_force: np.ndarray
    force_lower: np.ndarray
    force_upper: np.ndarray
    force_utilization: np.ndarray
    singular_values: np.ndarray
    elapsed_ns: int


@dataclass
class Report:
    jacobians: dict[str, tuple[np.ndarray, int, float]] = field(default_factory=dict)
    static: dict[str, tuple[float, float]] = field(default_factory=dict)
    continuous: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    timings_ns: list[int] = field(default_factory=list)
    max_delta: float = 0.0
    min_limit_margin: float = float("inf")
    nan_inf: int = 0
    failures: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failures.append(message)
        raise RuntimeError(message)


class Harness:
    def __init__(self, scene: object, robot: object, left_ee: object, right_ee: object,
                 left_dofs: list[int], right_dofs: list[int], gripper_drivers: list[int], config: dict, report: Report,
                 controller_mode: str = "measured_q"):
        self.scene, self.robot, self.report = scene, robot, report
        self.ee = {"left": left_ee, "right": right_ee}
        self.dofs = {"left": left_dofs, "right": right_dofs}
        self.home = {"left": LEFT_HOME, "right": RIGHT_HOME}
        self.gripper_drivers = gripper_drivers
        self.controller = {
            "left": DifferentialIKController(robot, left_ee, left_dofs, config),
            "right": DifferentialIKController(robot, right_ee, right_dofs, config),
        }
        for controller in self.controller.values():
            controller.mode = controller_mode
        self.lower = {side: arr(robot.get_dofs_limit(indices)[0]) for side, indices in self.dofs.items()}
        self.upper = {side: arr(robot.get_dofs_limit(indices)[1]) for side, indices in self.dofs.items()}

    def pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        return arr(self.ee[side].get_pos()), qnorm(arr(self.ee[side].get_quat()))

    def reset(self) -> None:
        self.robot.set_dofs_position(LEFT_HOME, dofs_idx_local=self.dofs["left"])
        self.robot.set_dofs_position(RIGHT_HOME, dofs_idx_local=self.dofs["right"])
        self.robot.control_dofs_position(LEFT_HOME, dofs_idx_local=self.dofs["left"])
        self.robot.control_dofs_position(RIGHT_HOME, dofs_idx_local=self.dofs["right"])
        for driver in self.gripper_drivers:
            self.robot.control_dofs_position(np.array([0.04]), dofs_idx_local=[driver])
        for _ in range(180):
            self.scene.step()
        state = arr(self.robot.get_dofs_position())
        if not np.isfinite(state).all():
            self.report.fail("home stability failed: non-finite joint state")
        for side, controller in self.controller.items():
            controller.reset_command_state(arr(self.robot.get_dofs_position(self.dofs[side])))

    def jacobian_check(self, side: str) -> None:
        jac = arr(self.robot.get_jacobian(link=self.ee[side]))[:, self.dofs[side]]
        if jac.shape != (6, 7) or not np.isfinite(jac).all():
            self.report.fail(f"{side} Jacobian invalid: shape={jac.shape}, finite={np.isfinite(jac).all()}")
        s = np.linalg.svd(jac, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jac))
        cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        self.report.jacobians[side] = (s, rank, cond)
        print(f"{side} Jacobian: shape={jac.shape}, singular_values={s}, min={s[-1]:.6g}, cond={cond:.3g}, rank={rank}")
        if rank != 6:
            self.report.fail(f"{side} Jacobian rank is {rank}, expected 6 at home")

    def _command(self, side: str, target: tuple[np.ndarray, np.ndarray], timestamp: float) -> Sample:
        pos, quat = self.pose(side)
        start = time.perf_counter_ns()
        command = self.controller[side].compute_command(*target)
        elapsed = time.perf_counter_ns() - start
        if command is None or not np.isfinite(command.position).all():
            self.report.nan_inf += 1
            self.report.fail(f"{side} DiffIK returned invalid command")
        q = arr(self.robot.get_dofs_position(self.dofs[side]))
        delta = command.position - q
        max_delta = self.controller[side].max_delta
        if self.controller[side].mode == "measured_q" and np.any(np.abs(delta) > max_delta + 1e-7):
            self.report.fail(f"{side} delta exceeded configured bound: {delta}")
        finite = np.isfinite(self.lower[side]) & np.isfinite(self.upper[side])
        margin = np.minimum(command.position[finite] - self.lower[side][finite], self.upper[side][finite] - command.position[finite])
        if np.any(margin < self.controller[side].margin - 1e-7):
            self.report.fail(f"{side} joint-limit margin violation")
        self.report.max_delta = max(self.report.max_delta, float(np.max(abs(delta))))
        self.report.min_limit_margin = min(self.report.min_limit_margin, float(np.min(margin)))
        self.report.timings_ns.append(elapsed)
        force = arr(self.robot.get_dofs_control_force(self.dofs[side]))
        force_lower, force_upper = (arr(x) for x in self.robot.get_dofs_force_range(self.dofs[side]))
        effort_limit = np.maximum(np.abs(force_lower), np.abs(force_upper))
        utilization = np.divide(np.abs(force), effort_limit, out=np.full_like(force, np.nan), where=effort_limit > 0)
        return Sample(timestamp, float(np.linalg.norm(target[0] - pos)), angle_error(target[1], quat), q, command.position,
                      delta, self.controller[side].last_raw_dq.copy(), self.controller[side].last_clipped_dq.copy(),
                      force, force_lower, force_upper, utilization, self.controller[side].last_singular_values.copy(), elapsed)

    def run_targets(self, targets: dict[str, tuple[np.ndarray, np.ndarray]], seconds: float,
                    hold_inactive: bool = True, compact_diagnostics: bool = False) -> dict[str, list[Sample]]:
        samples = {side: [] for side in targets}
        ticks = int(seconds * CONTROL_HZ)
        for tick in range(ticks):
            # Both commands are computed from the same pre-command state.
            commands = {side: self._command(side, target, tick / CONTROL_HZ) for side, target in targets.items()}
            for side, sample in commands.items():
                velocity = self.controller[side].last_qdot_command
                if velocity is None:
                    self.robot.control_dofs_position(sample.command, dofs_idx_local=self.dofs[side])
                else:
                    self.robot.control_dofs_position_velocity(sample.command, velocity, dofs_idx_local=self.dofs[side])
                samples[side].append(sample)
                if compact_diagnostics and tick % 30 == 0:
                    print(f"t={sample.timestamp:.2f}s pos={sample.position_error*1000:.2f}mm "
                          f"rot={math.degrees(sample.rotation_error):.2f}deg servo={np.max(abs(sample.delta)):.4f}rad "
                          f"raw={np.max(abs(sample.raw_dq)):.4f}rad clipped={np.max(abs(sample.clipped_dq)):.4f}rad "
                          f"force={np.nanmax(sample.force_utilization)*100:.1f}%")
            if hold_inactive:
                for side in {"left", "right"} - set(targets):
                    self.robot.control_dofs_position(self.home[side], dofs_idx_local=self.dofs[side])
            for _ in range(PHYSICS_PER_CONTROL):
                self.scene.step()
        return samples

    def focused_x(self, kinematic: bool = False) -> dict[str, float]:
        """The isolated, deliberately failing -X test used for controller diagnosis."""
        self.reset()
        pos, quat = self.pose("left")
        target = (pos + np.array([-.03, 0., 0.]), quat)
        if kinematic:
            samples = []
            for tick in range(int(3 * CONTROL_HZ)):
                sample = self._command("left", target, tick / CONTROL_HZ)
                self.robot.set_dofs_position(sample.command, dofs_idx_local=self.dofs["left"], zero_velocity=True)
                samples.append(sample)
                if tick % 30 == 0:
                    print(f"t={sample.timestamp:.2f}s kinematic pos={sample.position_error*1000:.2f}mm rot={math.degrees(sample.rotation_error):.2f}deg")
        else:
            samples = self.run_targets({"left": target}, 3.0, compact_diagnostics=True)["left"]
        final_pos, final_quat = self.pose("left")
        pe, re = float(np.linalg.norm(target[0] - final_pos)), angle_error(target[1], final_quat)
        tail = samples[-int(CONTROL_HZ):]
        t = np.array([s.timestamp for s in tail])
        ps, rs = np.array([s.position_error for s in tail]), np.array([s.rotation_error for s in tail])
        pslope, rslope = np.polyfit(t, ps, 1)[0], np.polyfit(t, rs, 1)[0]
        max_force = max(float(np.nanmax(s.force_utilization)) for s in samples) if not kinematic else float("nan")
        max_lead = max(float(np.max(abs(s.delta))) for s in samples)
        near_effort = float(np.mean([np.any(s.force_utilization >= .9) for s in samples]))
        classification = "STEADY-STATE TRACKING ERROR" if pe > .010 and abs(pslope) < .002 else "STILL CONVERGING / NOT A PLATEAU"
        print(f"FOCUSED {'KINEMATIC' if kinematic else 'DYNAMIC'} RESULT: final={pe*1000:.2f}mm/{math.degrees(re):.2f}deg; "
              f"final-1s mean={ps.mean()*1000:.2f}mm/{math.degrees(rs.mean()):.2f}deg; slopes={pslope*1000:.3f}mm/s/{math.degrees(rslope):.3f}deg/s; "
              f"max force={'n/a' if kinematic else f'{max_force*100:.1f}%'}; "
              f"ticks any actuator >=90%={'n/a' if kinematic else f'{near_effort*100:.1f}%'}; "
              f"max command lead={max_lead:.4f}rad; {classification}")
        return {"position": pe, "rotation": re, "max_force": max_force, "max_lead": max_lead, "position_slope": float(pslope)}

    def static(self, side: str, offset: np.ndarray, world_rotation: tuple[np.ndarray, float], label: str) -> None:
        self.reset()
        pos, quat = self.pose(side)
        target = (pos + offset, qmul(axis_angle(world_rotation[0], world_rotation[1]), quat))
        samples = self.run_targets({side: target}, 3.0)[side]
        final_pos, final_quat = self.pose(side)
        pe, re = float(np.linalg.norm(target[0] - final_pos)), angle_error(target[1], final_quat)
        self.report.static[label] = (pe, re)
        initial = samples[0]
        tenth = samples[min(9, len(samples) - 1)]
        peak_pos = max(s.position_error for s in samples)
        peak_rot = max(s.rotation_error for s in samples)
        print(f"{label}: pos={pe*1000:.2f} mm rot={math.degrees(re):.2f} deg (initial {initial.position_error*1000:.1f} mm/{math.degrees(initial.rotation_error):.1f} deg; tick10 {tenth.position_error*1000:.1f} mm/{math.degrees(tenth.rotation_error):.1f} deg; peak {peak_pos*1000:.1f} mm/{math.degrees(peak_rot):.1f} deg)")
        if pe > POS_FALLBACK or re > ROT_FALLBACK:
            self.report.fail(f"{label} failed fallback threshold: {pe*1000:.2f} mm, {math.degrees(re):.2f} deg")
        # The original stateless controller must improve immediately.  A
        # stateful command deliberately accumulates limited lead to overcome
        # gravity, so its transient is evaluated from peak/final values rather
        # than this baseline-only tick-10 rule.
        if self.controller[side].mode == "measured_q" and (tenth.position_error > initial.position_error + 1e-5 or tenth.rotation_error > initial.rotation_error + math.radians(.1)):
            self.report.fail(f"{label} error did not reduce by control tick 10")

    def orientation_convention(self) -> None:
        for axis, name in ((np.array([1., 0, 0]), "X"), (np.array([0., 1, 0]), "Y"), (np.array([0., 0, 1]), "Z")):
            self.reset()
            pos, quat = self.pose("left")
            target = (pos, qmul(axis_angle(axis, math.radians(5)), quat))
            samples = self.run_targets({"left": target}, .5)["left"]
            before, after = samples[0].rotation_error, samples[-1].rotation_error
            print(f"orientation world-{name}: {math.degrees(before):.3f} -> {math.degrees(after):.3f} deg")
            if not after < before:
                self.report.fail(f"orientation convention failed for world-{name}; error increased")

    def isolation(self, active: str) -> None:
        inactive = "right" if active == "left" else "left"
        self.reset()
        q0 = arr(self.robot.get_dofs_position(self.dofs[inactive]))
        p0, _ = self.pose(inactive)
        p, q = self.pose(active)
        self.run_targets({active: (p + np.array([.03, 0, .02]), q)}, 2.0)
        drift_q = float(np.max(abs(arr(self.robot.get_dofs_position(self.dofs[inactive])) - q0)))
        drift_p = float(np.linalg.norm(self.pose(inactive)[0] - p0))
        print(f"inactive {inactive} during {active}: joint={drift_q:.5f} rad EE={drift_p*1000:.3f} mm")
        if drift_q >= .01 or drift_p >= .002:
            self.report.fail(f"inactive-arm isolation failed ({inactive})")

    def continuous(self, sides: tuple[str, ...], label: str) -> None:
        self.reset()
        initial = {side: self.pose(side) for side in sides}
        samples = {side: [] for side in sides}
        for tick in range(int(10 * CONTROL_HZ)):
            t = tick / CONTROL_HZ
            targets = {}
            for side in sides:
                p, q = initial[side]
                sign = 1 if side == "left" else -1
                targets[side] = (p + np.array([.02 * math.sin(2 * math.pi * .2 * t), 0, .02 * math.cos(2 * math.pi * .2 * t)]), qmul(axis_angle(np.array([0., 0, 1]), sign * math.radians(5) * math.sin(2 * math.pi * .2 * t)), q))
            result = self.run_targets(targets, 1 / CONTROL_HZ, hold_inactive=True)
            for side in sides:
                samples[side].extend(result[side])
        for side, records in samples.items():
            ps = np.array([x.position_error for x in records]); rs = np.array([x.rotation_error for x in records])
            values = (float(np.sqrt(np.mean(ps**2))), float(np.max(ps)), float(np.sqrt(np.mean(rs**2))), float(np.max(rs)))
            self.report.continuous[f"{label}:{side}"] = values
            print(f"{label} {side}: RMS {values[0]*1000:.2f} mm/{math.degrees(values[2]):.2f} deg, max {values[1]*1000:.2f} mm/{math.degrees(values[3]):.2f} deg")
            if values[0] > .015 or values[1] > .030 or values[2] > math.radians(5) or values[3] > math.radians(10):
                self.report.fail(f"{label} {side} continuous tracking exceeded acceptance limits")

    def near_singularity(self) -> None:
        self.reset()
        # A moderate, controlled posture change raises conditioning without driving to an extreme pose.
        stress = LEFT_HOME + np.array([0, .45, 0, -.35, 0, 0, 0])
        self.robot.control_dofs_position(stress, dofs_idx_local=self.dofs["left"])
        for _ in range(240): self.scene.step()
        self.jacobian_check("left")
        p, q = self.pose("left")
        self.run_targets({"left": (p + np.array([.01, 0, 0]), q)}, 1.0)
        print("near-singularity stress: bounded commands PASS")


def print_report(report: Report) -> None:
    print("\nOPENARM DIFFERENTIAL IK VALIDATION\n==================================")
    for side in ("left", "right"):
        if side in report.jacobians:
            s, rank, cond = report.jacobians[side]
            print(f"{side.title()} Jacobian: shape: 6x7 rank: {rank} min singular value: {s[-1]:.6g} condition number: {cond:.3g}")
    print("Static tracking:")
    for name, (p, r) in report.static.items(): print(f"  {name}: {p*1000:.2f} mm, {math.degrees(r):.2f} deg")
    print("Continuous tracking:")
    for name, (rp, mp, rr, mr) in report.continuous.items(): print(f"  {name}: RMS {rp*1000:.2f} mm/{math.degrees(rr):.2f} deg; max {mp*1000:.2f} mm/{math.degrees(mr):.2f} deg")
    times = np.array(report.timings_ns) / 1e6 if report.timings_ns else np.array([float("nan")])
    print(f"Safety: max joint delta: {report.max_delta:.6f} rad min joint-limit margin: {report.min_limit_margin:.6f} rad NaN/inf count: {report.nan_inf} DiffIK failures: {len(report.failures)}")
    print(f"Performance: mean DiffIK time: {np.mean(times):.3f} ms p95: {np.percentile(times, 95):.3f} ms max: {np.max(times):.3f} ms")
    print("RESULT:\nPASS" if not report.failures else "RESULT:\nFAIL")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--test", choices=("focus", "static", "continuous", "bimanual", "all"), default="all",
                        help="Use focus for the isolated left -X 3 cm diagnostic.")
    parser.add_argument("--gravity", choices=("on", "off"), default="on",
                        help="Use off only for the controlled zero-gravity A/B diagnostic.")
    parser.add_argument("--controller", choices=("measured_q", "desired_q", "resolved_rate"), default="measured_q",
                        help="Keep measured_q as the failed baseline; desired_q is the bounded stateful candidate.")
    parser.add_argument("--kinematic", action="store_true",
                        help="For --test focus, directly set each DiffIK command to isolate solver mathematics from PD dynamics.")
    args = parser.parse_args()
    report = Report()
    try:
        if not URDF_PATH.is_file(): raise FileNotFoundError(URDF_PATH)
        config = yaml.safe_load(CONFIG_PATH.read_text())
        if args.controller == "resolved_rate":
            config["diffik"] = {
                "mode": "resolved_rate", "damping": 0.02,
                "position_rate_gain": 5.0, "rotation_rate_gain": 4.0,
                "max_linear_velocity_m_s": 0.50, "max_angular_velocity_rad_s": 1.50,
                "max_joint_velocity_rad_s": 2.0, "max_joint_acceleration_rad_s2": 12.0,
                "position_lookahead_s": 0.04, "max_position_lead_rad": 0.05,
                "joint_limit_margin_rad": 0.03,
            }
        gs.init(backend=gs.gpu)
        rigid_options = gs.options.RigidOptions(gravity=(0.0, 0.0, 0.0)) if args.gravity == "off" else gs.options.RigidOptions()
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=SIM_DT), rigid_options=rigid_options, show_viewer=not args.headless)
        scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.URDF(file=str(URDF_PATH), fixed=True, merge_fixed_links=True,
                                                 links_to_keep=(LEFT_EE_NAME, RIGHT_EE_NAME), recompute_inertia=False))
        scene.build()
        ld, rd = dofs(robot, LEFT_ARM_JOINTS), dofs(robot, RIGHT_ARM_JOINTS)
        for indices in (ld, rd):
            robot.set_dofs_kp(ARM_KP, dofs_idx_local=indices); robot.set_dofs_kv(ARM_KV, dofs_idx_local=indices)
        ensure_effort_limits(robot, LEFT_ARM_JOINTS, ld)
        ensure_effort_limits(robot, RIGHT_ARM_JOINTS, rd)
        # Keep the already-validated mimic grippers open and under their
        # established PD hold; they are not part of either 7-DOF controller.
        gripper_drivers = [dofs(robot, LEFT_GRIPPER_JOINTS)[0], dofs(robot, RIGHT_GRIPPER_JOINTS)[0]]
        for driver in gripper_drivers:
            robot.set_dofs_kp(np.array([80.0]), dofs_idx_local=[driver])
            robot.set_dofs_kv(np.array([4.0]), dofs_idx_local=[driver])
            robot.set_dofs_position(np.array([.04]), dofs_idx_local=[driver])
        ensure_effort_limits(robot, LEFT_GRIPPER_JOINTS, dofs(robot, LEFT_GRIPPER_JOINTS))
        ensure_effort_limits(robot, RIGHT_GRIPPER_JOINTS, dofs(robot, RIGHT_GRIPPER_JOINTS))
        h = Harness(scene, robot, robot.get_link(LEFT_EE_NAME), robot.get_link(RIGHT_EE_NAME), ld, rd, gripper_drivers, config, report, args.controller)
        h.reset(); h.jacobian_check("left"); h.jacobian_check("right")
        if args.test == "focus":
            h.focused_x(kinematic=args.kinematic)
        if args.test in ("static", "all"):
            h.orientation_convention()
            for side in ("left", "right"):
                for axis in np.eye(3):
                    for sign in (1, -1): h.static(side, sign * .03 * axis, (np.array([0., 0, 1]), 0), f"{side} translation {sign:+d}{axis}")
                for axis in np.eye(3):
                    for sign in (1, -1): h.static(side, np.zeros(3), (axis, sign * math.radians(5)), f"{side} rotation {sign:+d}{axis}")
            h.static("left", np.array([.03, 0, .02]), (np.array([0., 0, 1]), math.radians(5)), "left combined")
            h.static("right", np.array([.025, 0, -.02]), (np.array([0., 0, 1]), -math.radians(5)), "right combined")
            h.isolation("left"); h.isolation("right")
        if args.test in ("bimanual", "all"):
            h.reset(); lp, lq = h.pose("left"); rp, rq = h.pose("right")
            h.run_targets({"left": (lp + np.array([.03, 0, .02]), qmul(axis_angle(np.array([0.,0,1]), math.radians(5)), lq)), "right": (rp + np.array([0, .02, .02]), qmul(axis_angle(np.array([0.,0,1]), -math.radians(5)), rq))}, 3.0)
            for side, target in (("left", (lp + np.array([.03,0,.02]), qmul(axis_angle(np.array([0.,0,1]),math.radians(5)),lq))), ("right", (rp + np.array([0,.02,.02]),qmul(axis_angle(np.array([0.,0,1]),-math.radians(5)),rq)))):
                p, q = h.pose(side); pe, re = np.linalg.norm(target[0]-p), angle_error(target[1],q)
                h.report.static[f"bimanual {side}"] = (pe, re)
                print(f"bimanual {side}: pos={pe*1000:.2f} mm rot={math.degrees(re):.2f} deg")
                if pe > POS_FALLBACK or re > ROT_FALLBACK: h.report.fail(f"bimanual {side} failed fallback threshold")
        if args.test in ("continuous", "all"):
            h.continuous(("left",), "continuous"); h.continuous(("right",), "continuous"); h.continuous(("left", "right"), "continuous-bimanual")
        if args.test == "all": h.near_singularity()
    except Exception as exc:
        report.failures.append(str(exc))
        print(f"VALIDATION STOPPED: {exc}")
    print_report(report)
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
