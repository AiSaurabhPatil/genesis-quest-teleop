from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

R_G_XR = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def normalize_quat_xyzw(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(q).all() or n < 1e-6:
        raise ValueError("invalid xyzw quaternion")
    return q / n


def normalize_quat_wxyz(q):
    return xyzw_to_wxyz(normalize_quat_xyzw(wxyz_to_xyzw(q)))


def xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def map_webxr_position_to_genesis(position_xyz):
    return R_G_XR @ np.asarray(position_xyz, dtype=float)


def map_webxr_quat_to_genesis_wxyz(quat_xyzw):
    r = Rotation.from_quat(normalize_quat_xyzw(quat_xyzw)).as_matrix()
    return normalize_quat_wxyz(
        xyzw_to_wxyz(Rotation.from_matrix(R_G_XR @ r @ R_G_XR.T).as_quat())
    )


def quat_multiply_wxyz(a, b):
    return normalize_quat_wxyz(
        xyzw_to_wxyz(
            (
                Rotation.from_quat(wxyz_to_xyzw(a))
                * Rotation.from_quat(wxyz_to_xyzw(b))
            ).as_quat()
        )
    )


def quat_inverse_wxyz(q):
    return normalize_quat_wxyz(
        xyzw_to_wxyz(Rotation.from_quat(wxyz_to_xyzw(q)).inv().as_quat())
    )


def quat_angle_wxyz(a, b):
    return float(
        (
            Rotation.from_quat(wxyz_to_xyzw(a)).inv()
            * Rotation.from_quat(wxyz_to_xyzw(b))
        ).magnitude()
    )


def slerp_wxyz(a, b, t):
    return normalize_quat_wxyz(
        xyzw_to_wxyz(
            Slerp([0, 1], Rotation.from_quat([wxyz_to_xyzw(a), wxyz_to_xyzw(b)]))(
                [float(t)]
            ).as_quat()[0]
        )
    )
