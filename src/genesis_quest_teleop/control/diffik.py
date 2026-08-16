from __future__ import annotations

import numpy as np


class DifferentialIKController:
    def __init__(self, robot, ee_link, arm_dofs_idx, config):
        self.robot, self.ee_link, self.arm_dofs_idx = robot, ee_link, list(arm_dofs_idx)
        c = config["diffik"]
        self.damping = c["damping"]
        self.pg = c["position_gain"]
        self.rg = c["rotation_gain"]
        self.max_delta = c["max_joint_delta_rad"]
        self.margin = c["joint_limit_margin_rad"]
        self.eye = np.eye(6)
        lower_limit, upper_limit = robot.get_dofs_limit(self.arm_dofs_idx)
        self.lower = lower_limit.cpu().numpy()
        self.upper = upper_limit.cpu().numpy()
        self.last_valid = None
        # Read-only diagnostics used by the standalone OpenArm validation.  They
        # intentionally mirror values used for the command and do not affect
        # control behavior or the existing Franka interface.
        self.last_error = None
        self.last_dq = None
        self.last_jacobian = None
        self.last_singular_values = None

    def compute_command(self, target_pos, target_quat_wxyz):
        try:
            import genesis as gs

            pos = self.ee_link.get_pos().cpu().numpy()
            q = self.ee_link.get_quat().cpu().numpy()
            # Genesis's angular Jacobian rows use the end-effector/body frame,
            # so use the body-frame residual taking current to target.
            errq = gs.transform_quat_by_quat(gs.inv_quat(q), target_quat_wxyz)
            rot = gs.quat_to_rotvec(errq)
            error = np.r_[self.pg * (target_pos - pos), self.rg * rot]
            jac = (
                self.robot.get_jacobian(link=self.ee_link)
                .cpu()
                .numpy()[:, self.arm_dofs_idx]
            )
            if (
                jac.shape != (6, len(self.arm_dofs_idx))
                or not np.isfinite(error).all()
                or not np.isfinite(jac).all()
            ):
                return None
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + self.damping**2 * self.eye, error
            )
            current = self.robot.get_dofs_position(self.arm_dofs_idx).cpu().numpy()
            cmd = current + np.clip(dq, -self.max_delta, self.max_delta)
            finite = np.isfinite(self.lower) & np.isfinite(self.upper)
            cmd[finite] = np.clip(
                cmd[finite],
                self.lower[finite] + self.margin,
                self.upper[finite] - self.margin,
            )
            if not np.isfinite(cmd).all():
                return None
            self.last_error = error.copy()
            self.last_dq = dq.copy()
            self.last_jacobian = jac.copy()
            self.last_singular_values = np.linalg.svd(jac, compute_uv=False)
            self.last_valid = cmd
            return cmd
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            return None
