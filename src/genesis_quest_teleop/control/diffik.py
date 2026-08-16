from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JointCommand:
    """An arm PD command; velocity is optional for legacy position control."""

    position: np.ndarray
    velocity: np.ndarray | None = None


class DifferentialIKController:
    def __init__(self, robot, ee_link, arm_dofs_idx, config, control_dt=None, joint_velocity_limits=None):
        self.robot, self.ee_link, self.arm_dofs_idx = robot, ee_link, list(arm_dofs_idx)
        c = config["diffik"]
        self.damping, self.margin = c["damping"], c["joint_limit_margin_rad"]
        self.mode = c.get("mode", "measured_q")
        self.pg, self.rg = c.get("position_gain"), c.get("rotation_gain")
        self.max_delta = c.get("max_joint_delta_rad")
        self.max_command_lead = c.get("max_command_lead_rad", 0.2)
        self.control_dt = float(control_dt or 1.0 / config["sim"]["control_hz"])
        self.position_rate_gain = c.get("position_rate_gain")
        self.rotation_rate_gain = c.get("rotation_rate_gain")
        self.max_linear_velocity = c.get("max_linear_velocity_m_s")
        self.max_angular_velocity = c.get("max_angular_velocity_rad_s")
        configured_joint_velocity = c.get("max_joint_velocity_rad_s")
        self.max_joint_velocity = (np.minimum(configured_joint_velocity, joint_velocity_limits)
                                   if joint_velocity_limits is not None else configured_joint_velocity)
        self.max_joint_acceleration = c.get("max_joint_acceleration_rad_s2")
        self.position_lookahead = c.get("position_lookahead_s")
        self.max_position_lead = c.get("max_position_lead_rad")
        self.eye = np.eye(6)
        lower_limit, upper_limit = robot.get_dofs_limit(self.arm_dofs_idx)
        self.lower, self.upper = lower_limit.cpu().numpy(), upper_limit.cpu().numpy()
        self.last_valid = self.last_error = self.last_dq = None
        self.last_raw_dq = self.last_clipped_dq = self.last_jacobian = None
        self.last_singular_values = None
        self.last_qdot_command = self.last_position_lead = None
        self.velocity_limited = self.acceleration_limited = self.joint_limit_clamped = False
        self.q_desired = None
        self.previous_qdot = np.zeros(len(self.arm_dofs_idx))

    @staticmethod
    def _limit_magnitude(vector, limit):
        magnitude = np.linalg.norm(vector)
        return vector * (limit / magnitude) if magnitude > limit else vector

    def reset_command_state(self, current_q=None):
        """Discard trajectory state; resolved-rate control never stores position lead."""
        self.previous_qdot.fill(0.0)
        if self.mode != "resolved_rate":
            if current_q is None:
                current_q = self.robot.get_dofs_position(self.arm_dofs_idx).cpu().numpy()
            self.q_desired = np.asarray(current_q, dtype=float).copy()
            self.last_valid = self.q_desired.copy()

    reset = reset_command_state

    def _clamp_position(self, position, qdot=None):
        safe_lower, safe_upper = self.lower + self.margin, self.upper - self.margin
        finite = np.isfinite(safe_lower) & np.isfinite(safe_upper)
        unclamped = position.copy()
        position[finite] = np.clip(position[finite], safe_lower[finite], safe_upper[finite])
        clamped = position != unclamped
        self.joint_limit_clamped = bool(np.any(clamped))
        if qdot is not None:
            qdot[(position <= safe_lower) & (qdot < 0)] = 0.0
            qdot[(position >= safe_upper) & (qdot > 0)] = 0.0
        return position

    def compute_command(self, target_pos, target_quat_wxyz):
        try:
            import genesis as gs

            pos = self.ee_link.get_pos().cpu().numpy()
            quat = self.ee_link.get_quat().cpu().numpy()
            errq = gs.transform_quat_by_quat(gs.inv_quat(quat), target_quat_wxyz)
            rotation_error = gs.quat_to_rotvec(errq)
            position_error = target_pos - pos
            jac = self.robot.get_jacobian(link=self.ee_link).cpu().numpy()[:, self.arm_dofs_idx]
            if jac.shape != (6, len(self.arm_dofs_idx)) or not np.isfinite(jac).all():
                return None
            current = self.robot.get_dofs_position(self.arm_dofs_idx).cpu().numpy()
            if self.mode == "resolved_rate":
                linear_velocity = self._limit_magnitude(self.position_rate_gain * position_error, self.max_linear_velocity)
                angular_velocity = self._limit_magnitude(self.rotation_rate_gain * rotation_error, self.max_angular_velocity)
                twist = np.r_[linear_velocity, angular_velocity]
                if not np.isfinite(twist).all():
                    return None
                qdot_raw = jac.T @ np.linalg.solve(jac @ jac.T + self.damping**2 * self.eye, twist)
                qdot_limited = np.clip(qdot_raw, -self.max_joint_velocity, self.max_joint_velocity)
                self.velocity_limited = bool(np.any(qdot_limited != qdot_raw))
                max_delta_velocity = self.max_joint_acceleration * self.control_dt
                qdot_command = np.clip(qdot_limited, self.previous_qdot - max_delta_velocity, self.previous_qdot + max_delta_velocity)
                self.acceleration_limited = bool(np.any(qdot_command != qdot_limited))
                lead = np.clip(qdot_command * self.position_lookahead, -self.max_position_lead, self.max_position_lead)
                command = self._clamp_position(current + lead, qdot_command)
                lead = np.clip(qdot_command * self.position_lookahead, -self.max_position_lead, self.max_position_lead)
                command = self._clamp_position(current + lead, qdot_command)
                self.previous_qdot = qdot_command.copy()
                self.last_error, self.last_dq = twist.copy(), qdot_raw.copy()
                self.last_raw_dq, self.last_clipped_dq = qdot_raw.copy(), qdot_limited.copy()
                self.last_qdot_command, self.last_position_lead = qdot_command.copy(), command - current
                self.last_valid = command.copy()
                self.last_jacobian, self.last_singular_values = jac.copy(), np.linalg.svd(jac, compute_uv=False)
                return JointCommand(command, qdot_command)

            error = np.r_[self.pg * position_error, self.rg * rotation_error]
            if not np.isfinite(error).all():
                return None
            raw_dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping**2 * self.eye, error)
            clipped_dq = np.clip(raw_dq, -self.max_delta, self.max_delta)
            if self.mode == "desired_q":
                if self.q_desired is None:
                    self.reset_command_state(current)
                self.q_desired += clipped_dq
                command = current + np.clip(self.q_desired - current, -self.max_command_lead, self.max_command_lead)
            else:
                command = current + clipped_dq
            command = self._clamp_position(command)
            if self.mode == "desired_q": self.q_desired = command.copy()
            self.last_error, self.last_dq = error.copy(), raw_dq.copy()
            self.last_raw_dq, self.last_clipped_dq = raw_dq.copy(), clipped_dq.copy()
            self.last_jacobian, self.last_singular_values, self.last_valid = jac.copy(), np.linalg.svd(jac, compute_uv=False), command.copy()
            return JointCommand(command)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            return None
