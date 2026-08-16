from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root() / path).resolve()


def apply_env_overrides(config: dict) -> dict:
    config = copy.deepcopy(config)
    mapping = {
        "GENESIS_TELEOP_CERT": ("tls", "cert_file"),
        "GENESIS_TELEOP_KEY": ("tls", "key_file"),
        "GENESIS_TELEOP_HOST": ("server", "host"),
        "GENESIS_TELEOP_PORT": ("server", "port"),
    }
    for env, (section, key) in mapping.items():
        if value := os.getenv(env):
            config[section][key] = int(value) if env.endswith("PORT") else value
    return config


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML object")
    config = apply_env_overrides(config)
    try:
        config["web"]["directory"] = str(
            resolve_project_path(config["web"]["directory"])
        )
        for key in ("cert_file", "key_file"):
            config["tls"][key] = str(resolve_project_path(config["tls"][key]))
        if not 1 <= int(config["ipc"]["port"]) <= 65535:
            raise ValueError("ipc.port must be between 1 and 65535")
        if config["ipc"]["host"] not in ("127.0.0.1", "localhost"):
            raise ValueError("ipc.host must remain localhost-only")
        if config["sim"]["dt"] <= 0 or config["sim"]["control_hz"] <= 0:
            raise ValueError("sim.dt and sim.control_hz must be positive")
        cube = config.get("scene", {}).get("grasp_cube")
        if cube and cube.get("enabled", False):
            if len(cube["position"]) != 3 or len(cube["size"]) != 3:
                raise ValueError("scene.grasp_cube position and size must have 3 values")
            if any(float(value) <= 0 for value in cube["size"]):
                raise ValueError("scene.grasp_cube size values must be positive")
            if float(cube["friction"]) <= 0 or float(cube["density"]) <= 0:
                raise ValueError("scene.grasp_cube friction and density must be positive")
            if len(cube["color"]) != 4:
                raise ValueError("scene.grasp_cube color must be RGBA")
            pbr = cube.get("pbr", {})
            for key in ("metallic", "roughness"):
                if key in pbr and not 0.0 <= float(pbr[key]) <= 1.0:
                    raise ValueError(f"scene.grasp_cube.pbr.{key} must be in [0, 1]")
        if not 0.01 <= float(config["gripper"]["finger_friction"]) <= 5.0:
            raise ValueError("gripper.finger_friction must be between 0.01 and 5.0")
        robot = config["robot"]
        if robot["type"] not in ("franka", "openarm"):
            raise ValueError("robot.type must be franka or openarm")
        for key in ("base_position", "base_euler_deg"):
            if key in robot and len(robot[key]) != 3:
                raise ValueError(f"robot.{key} must contain [x, y, z]")
        t = config["teleop"]
        if not 0 <= t["clutch_release_threshold"] < t["clutch_engage_threshold"] <= 1:
            raise ValueError("invalid clutch thresholds")
        if robot["type"] == "openarm":
            urdf = resolve_project_path(robot["urdf_file"])
            if not urdf.is_file():
                raise ValueError(f"OpenArm URDF does not exist: {urdf}")
            for arm in ("left", "right"):
                if len(robot["home"][arm]) != 7:
                    raise ValueError(f"robot.home.{arm} must contain 7 values")
                if t["arm_bindings"].get(arm) not in ("left", "right"):
                    raise ValueError(f"teleop.arm_bindings.{arm} must be left or right")
                for axis in "xyz":
                    bounds = t["workspace"][arm][axis]
                    if len(bounds) != 2 or bounds[0] >= bounds[1]:
                        raise ValueError(f"workspace.{arm}.{axis} min must be less than max")
            if len(set(t["arm_bindings"].values())) != 2:
                raise ValueError("OpenArm arm bindings must be unique")
            if len(robot["arm_kp"]) != 7 or len(robot["arm_kv"]) != 7:
                raise ValueError("OpenArm arm_kp and arm_kv must contain 7 values")
            if config["diffik"].get("mode") not in ("measured_q", "desired_q"):
                raise ValueError("diffik.mode must be measured_q or desired_q")
            if float(config["diffik"].get("max_command_lead_rad", 0)) <= 0:
                raise ValueError("diffik.max_command_lead_rad must be positive")
            if not float(config["gripper"]["grasp_release_threshold"]) < float(config["gripper"]["grasp_engage_threshold"]):
                raise ValueError("invalid gripper grasp thresholds")
        else:
            for axis in "xyz":
                if t["workspace"][axis][0] >= t["workspace"][axis][1]:
                    raise ValueError(f"workspace.{axis} min must be less than max")
        nyx = config.get("nyx_camera", {})

        if nyx.get("enabled", False):
            resolution = nyx["resolution"]

            if len(resolution) != 2:
                raise ValueError(
                    "nyx_camera.resolution must contain [width, height]"
                )

            if any(int(value) <= 0 for value in resolution):
                raise ValueError(
                    "nyx_camera.resolution values must be positive"
                )

            if not 1.0 <= float(nyx["fov"]) < 180.0:
                raise ValueError(
                    "nyx_camera.fov must be between 1 and 180 degrees"
                )

            if float(nyx["near"]) <= 0:
                raise ValueError(
                    "nyx_camera.near must be positive"
                )

            if float(nyx["far"]) <= float(nyx["near"]):
                raise ValueError(
                    "nyx_camera.far must be greater than near"
                )

            if int(nyx["spp"]) <= 0:
                raise ValueError(
                    "nyx_camera.spp must be positive"
                )

            if float(nyx["render_hz"]) <= 0:
                raise ValueError(
                    "nyx_camera.render_hz must be positive"
                )

            offset_T = nyx["offset_T"]

            if (
                len(offset_T) != 4
                or any(len(row) != 4 for row in offset_T)
            ):
                raise ValueError(
                    "nyx_camera.offset_T must be a 4x4 matrix"
                )

            if not str(nyx["parent_link"]):
                raise ValueError(
                    "nyx_camera.parent_link must not be empty"
                )

        if nyx.get("enabled", False):
            physics_hz = 1.0 / float(config["sim"]["dt"])
            if float(nyx["render_hz"]) > physics_hz:
                raise ValueError(
                    "nyx_camera.render_hz must not exceed physics frequency"
                )
    except KeyError as exc:
        raise ValueError(f"missing configuration key: {exc}") from exc
    return config
