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
        if not 0.01 <= float(config["gripper"]["finger_friction"]) <= 5.0:
            raise ValueError("gripper.finger_friction must be between 0.01 and 5.0")
        t = config["teleop"]
        if not 0 <= t["clutch_release_threshold"] < t["clutch_engage_threshold"] <= 1:
            raise ValueError("invalid clutch thresholds")
        for axis in "xyz":
            if t["workspace"][axis][0] >= t["workspace"][axis][1]:
                raise ValueError(f"workspace.{axis} min must be less than max")
    except KeyError as exc:
        raise ValueError(f"missing configuration key: {exc}") from exc
    return config
