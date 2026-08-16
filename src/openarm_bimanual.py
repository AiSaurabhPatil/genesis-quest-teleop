"""Standalone Genesis validation for the TeleSim OpenArm bimanual URDF.

Run with ``uv run python src/openarm_bimanual.py`` from this repository.
The script intentionally contains no teleoperation or application integration.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import genesis as gs
import numpy as np

# Robot-specific configuration -------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TELESIM_ROOT = WORKSPACE_ROOT.parent / "telesim"
URDF_PATH = TELESIM_ROOT / "robot_configs/openarm_config/openarm_bimanual.urdf"
OPENARM_DESCRIPTION_PATH = WORKSPACE_ROOT.parent / "openarm_description"
URDF_PACKAGE_LINK = URDF_PATH.parent / "openarm_description"

LEFT_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINTS = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
RIGHT_GRIPPER_JOINTS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
LEFT_EE_NAME = "openarm_left_hand"
RIGHT_EE_NAME = "openarm_right_hand"

LEFT_HOME = np.array([0.0, -1.0, 0.0, 1.2, 0.0, 0.0, 0.0])
RIGHT_HOME = np.array([0.0, 1.0, 0.0, 1.2, 0.0, 0.0, 0.0])
ARM_KP = np.array([400, 400, 300, 250, 150, 100, 80])
ARM_KV = np.array([40, 40, 30, 25, 15, 10, 8])
# The supplied finger tensors violate Genesis's physical-inertia triangle inequality.
# Recompute only because the current Genesis loader rejects the original values.
RECOMPUTE_INERTIA = True


def as_numpy(value: object) -> np.ndarray:
    """Convert Genesis/Torch values to a detached host array for validation."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def require_assets() -> None:
    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"OpenArm URDF not found: {URDF_PATH}")
    if not OPENARM_DESCRIPTION_PATH.is_dir():
        raise FileNotFoundError(
            "openarm_description package not found: "
            f"{OPENARM_DESCRIPTION_PATH}. Expected the original package here so "
            f"'{URDF_PACKAGE_LINK}' can resolve package:// mesh paths."
        )
    mesh_root = OPENARM_DESCRIPTION_PATH / "meshes/arm/v10"
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"openarm_description mesh directory not found: {mesh_root}")
    if not URDF_PACKAGE_LINK.is_symlink() or URDF_PACKAGE_LINK.resolve() != OPENARM_DESCRIPTION_PATH.resolve():
        raise RuntimeError(
            "OpenArm package symlink is missing or points elsewhere: "
            f"{URDF_PACKAGE_LINK} -> {OPENARM_DESCRIPTION_PATH}. Create it with: "
            f"ln -s {OPENARM_DESCRIPTION_PATH} {URDF_PACKAGE_LINK}"
        )


def resolve_dofs(robot: object, names: list[str]) -> list[int]:
    imported_names = [joint.name for joint in robot.joints]
    dofs: list[int] = []
    for name in names:
        try:
            joint = robot.get_joint(name)
        except Exception as exc:
            raise RuntimeError(
                f"Missing joint '{name}'. All imported joint names: {imported_names}"
            ) from exc
        if not joint.dofs_idx_local:
            raise RuntimeError(f"Joint '{name}' has no local DOF after import.")
        dofs.append(joint.dofs_idx_local[0])
    return dofs


def resolve_link(robot: object, name: str) -> object:
    imported_names = [link.name for link in robot.links]
    try:
        return robot.get_link(name)
    except Exception as exc:
        raise RuntimeError(f"Missing EE link '{name}'. All imported links: {imported_names}") from exc


def urdf_effort_limits() -> dict[str, float]:
    root = ET.parse(URDF_PATH).getroot()
    limits: dict[str, float] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        effort = None if limit is None else limit.get("effort")
        if effort is not None:
            limits[joint.attrib["name"]] = float(effort)
    return limits


def ensure_effort_limits(robot: object, names: list[str], dofs: list[int]) -> None:
    """Preserve imported limits, falling back only when they are invalid."""
    lower, upper = (as_numpy(x) for x in robot.get_dofs_force_range(dofs))
    print(f"Imported force ranges for {names[0].split('_')[1]} arm:", list(zip(lower, upper)))
    if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(upper > lower):
        return
    effort = urdf_effort_limits()
    try:
        magnitudes = np.array([effort[name] for name in names])
    except KeyError as exc:
        raise RuntimeError(f"URDF has no effort limit for joint '{exc.args[0]}'.") from exc
    print("Imported force limits are invalid; applying URDF effort limits:", magnitudes)
    robot.set_dofs_force_range(-magnitudes, magnitudes, dofs)


def print_state(robot: object, left_dofs: list[int], right_dofs: list[int], label: str) -> None:
    print(label)
    print("  left positions:", as_numpy(robot.get_dofs_position(left_dofs)))
    print("  right positions:", as_numpy(robot.get_dofs_position(right_dofs)))
    print("  left control forces:", as_numpy(robot.get_dofs_control_force(left_dofs)))
    print("  right control forces:", as_numpy(robot.get_dofs_control_force(right_dofs)))


def step_control(scene: object, robot: object, left_dofs: list[int], right_dofs: list[int], steps: int, label: str) -> None:
    for step in range(steps):
        if step % 40 == 0:
            print_state(robot, left_dofs, right_dofs, f"{label}, step {step}")
        scene.step()


def validate_ik_solution(robot: object, qpos: object, dofs: list[int], current_qpos: np.ndarray, label: str) -> np.ndarray:
    solution = as_numpy(qpos).reshape(-1)
    if solution.shape != current_qpos.shape:
        raise RuntimeError(f"{label} IK returned shape {solution.shape}; expected {current_qpos.shape}.")
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(f"{label} IK output contains NaN or inf: {solution}")
    lower, upper = (as_numpy(x) for x in robot.get_dofs_limit(dofs))
    arm_solution = solution[dofs]
    if np.any(arm_solution < lower - 1e-5) or np.any(arm_solution > upper + 1e-5):
        raise RuntimeError(f"{label} IK violates arm joint limits: {arm_solution}")
    change = arm_solution - current_qpos[dofs]
    print(f"{label} IK full solution:", solution)
    print(f"{label} IK arm delta:", change, "max abs:", np.max(np.abs(change)))
    if np.max(np.abs(change)) > math.pi:
        raise RuntimeError(f"{label} IK has an implausibly large arm jump: {change}")
    return solution


def run_single_arm_ik(
    scene: object,
    robot: object,
    ee: object,
    dofs: list[int],
    left_dofs: list[int],
    right_dofs: list[int],
    label: str,
) -> None:
    current_qpos = as_numpy(robot.get_dofs_position())
    target_pos = as_numpy(ee.get_pos()) + np.array([0.05, 0.0, 0.0])
    print(f"{label} EE target position:", target_pos)
    solution = robot.inverse_kinematics(
        link=ee,
        pos=target_pos,
        init_qpos=current_qpos,
        respect_joint_limit=True,
        dofs_idx_local=dofs,
    )
    solution = validate_ik_solution(robot, solution, dofs, current_qpos, label)
    robot.control_dofs_position(solution[dofs], dofs_idx_local=dofs)
    step_control(scene, robot, left_dofs, right_dofs, 160, f"{label} IK motion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run validation without the Genesis viewer.")
    parser.add_argument("--no-hold", action="store_true", help="Exit after the validation sequence.")
    args = parser.parse_args()

    require_assets()
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, -3.2, 1.8), camera_lookat=(0.0, 0.0, 0.65), camera_fov=40
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=not args.headless,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(URDF_PATH), fixed=True, merge_fixed_links=True,
            links_to_keep=(LEFT_EE_NAME, RIGHT_EE_NAME), recompute_inertia=RECOMPUTE_INERTIA,
        )
    )
    scene.build()

    print(f"OpenArm structure: links={robot.n_links}, joints={robot.n_joints}, dofs={robot.n_dofs}")
    for link in robot.links:
        print("link", link.idx_local, link.name)
    for joint in robot.joints:
        print("joint", joint.name, joint.dofs_idx_local)

    left_dofs = resolve_dofs(robot, LEFT_ARM_JOINTS)
    right_dofs = resolve_dofs(robot, RIGHT_ARM_JOINTS)
    left_gripper_dofs = resolve_dofs(robot, LEFT_GRIPPER_JOINTS)
    right_gripper_dofs = resolve_dofs(robot, RIGHT_GRIPPER_JOINTS)
    print("Left arm local DOFs:", left_dofs)
    print("Right arm local DOFs:", right_dofs)
    print("Left gripper local DOFs (open ~= 0.04, closed ~= 0.0):", left_gripper_dofs)
    print("Right gripper local DOFs (open ~= 0.04, closed ~= 0.0):", right_gripper_dofs)

    left_ee, right_ee = resolve_link(robot, LEFT_EE_NAME), resolve_link(robot, RIGHT_EE_NAME)
    for ee in (left_ee, right_ee):
        print(f"EE {ee.name}: idx={ee.idx_local}, pos={as_numpy(ee.get_pos())}, quat={as_numpy(ee.get_quat())}")

    robot.set_dofs_kp(ARM_KP, dofs_idx_local=left_dofs)
    robot.set_dofs_kp(ARM_KP, dofs_idx_local=right_dofs)
    robot.set_dofs_kv(ARM_KV, dofs_idx_local=left_dofs)
    robot.set_dofs_kv(ARM_KV, dofs_idx_local=right_dofs)
    ensure_effort_limits(robot, LEFT_ARM_JOINTS, left_dofs)
    ensure_effort_limits(robot, RIGHT_ARM_JOINTS, right_dofs)
    robot.set_dofs_position(LEFT_HOME, dofs_idx_local=left_dofs)
    robot.set_dofs_position(RIGHT_HOME, dofs_idx_local=right_dofs)

    print("Passive stability test (200 steps)")
    for _ in range(200):
        scene.step()
    state = as_numpy(robot.get_dofs_position())
    if not np.all(np.isfinite(state)):
        raise RuntimeError("Passive stability test failed: robot joint state contains NaN or inf.")

    left_target = LEFT_HOME.copy()
    left_target[1] += 0.15
    left_target[3] -= 0.15
    robot.control_dofs_position(left_target, dofs_idx_local=left_dofs)
    step_control(scene, robot, left_dofs, right_dofs, 300, "left joint-space motion")
    robot.control_dofs_position(LEFT_HOME, dofs_idx_local=left_dofs)
    step_control(scene, robot, left_dofs, right_dofs, 200, "left return home")

    right_target = RIGHT_HOME.copy()
    right_target[1] -= 0.15
    right_target[3] -= 0.15
    robot.control_dofs_position(right_target, dofs_idx_local=right_dofs)
    step_control(scene, robot, left_dofs, right_dofs, 300, "right joint-space motion")
    robot.control_dofs_position(RIGHT_HOME, dofs_idx_local=right_dofs)
    step_control(scene, robot, left_dofs, right_dofs, 200, "right return home")

    run_single_arm_ik(scene, robot, left_ee, left_dofs, left_dofs, right_dofs, "left")
    robot.control_dofs_position(LEFT_HOME, dofs_idx_local=left_dofs)
    step_control(scene, robot, left_dofs, right_dofs, 160, "left IK return home")
    run_single_arm_ik(scene, robot, right_ee, right_dofs, left_dofs, right_dofs, "right")

    current_qpos = as_numpy(robot.get_dofs_position())
    left_target = as_numpy(left_ee.get_pos()) + np.array([0.02, 0.0, 0.0])
    right_target = as_numpy(right_ee.get_pos()) + np.array([0.02, 0.0, 0.0])
    bimanual_dofs = left_dofs + right_dofs
    bimanual_solution = robot.inverse_kinematics_multilink(
        links=[left_ee, right_ee], poss=[left_target, right_target], init_qpos=current_qpos,
        respect_joint_limit=True, dofs_idx_local=bimanual_dofs,
    )
    bimanual_solution = validate_ik_solution(robot, bimanual_solution, bimanual_dofs, current_qpos, "bimanual")
    robot.control_dofs_position(bimanual_solution[left_dofs], dofs_idx_local=left_dofs)
    robot.control_dofs_position(bimanual_solution[right_dofs], dofs_idx_local=right_dofs)
    step_control(scene, robot, left_dofs, right_dofs, 200, "bimanual IK motion")

    print("OpenArm validation complete.")
    if not args.no_hold:
        print("Viewer remains active; press Ctrl+C to exit.")
        while True:
            scene.step()


if __name__ == "__main__":
    main()
