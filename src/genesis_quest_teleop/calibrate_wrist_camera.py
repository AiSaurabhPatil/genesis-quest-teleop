from __future__ import annotations

import argparse
import logging

import numpy as np

from .config import load_config, project_root
from .robots.franka import FrankaAdapter


def _format_offset_yaml(offset_T: np.ndarray) -> str:
    rows = [
        "    - [" + ", ".join(f"{value: .8f}" for value in row) + "]"
        for row in offset_T
    ]
    return "nyx_camera:\n  offset_T:\n" + "\n".join(rows)


def _rotation_matrix(axis: int, angle_rad: float) -> np.ndarray:
    rotation = np.eye(3)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    first = (axis + 1) % 3
    second = (axis + 2) % 3
    rotation[first, first] = c
    rotation[second, second] = c
    rotation[first, second] = -s
    rotation[second, first] = s
    return rotation


def _build_scene(config: dict):
    import genesis as gs

    sim = config["sim"]
    gs.init(backend=gs.cuda if sim["backend"] == "gpu" else gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim["dt"]),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            noslip_iterations=5,
            noslip_tolerance=1e-6,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, -1.4, 1.1),
            camera_lookat=(0.45, 0.0, 0.35),
            camera_fov=float(config["nyx_camera"]["fov"]),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=True,
        ),
        show_viewer=True,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = config.get("scene", {}).get("grasp_cube", {})
    if cube.get("enabled", False):
        pbr = cube.get("pbr", {})
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=tuple(cube["position"]),
                size=tuple(cube["size"]),
            ),
            material=gs.materials.Rigid(
                friction=cube["friction"],
                rho=cube["density"],
            ),
            surface=gs.surfaces.Default(
                color=tuple(cube["color"]),
                metallic=float(pbr.get("metallic", 0.0)),
                roughness=float(pbr.get("roughness", 0.5)),
            ),
        )
    robot = FrankaAdapter(config)
    robot.build(scene)
    scene.build()
    robot.initialize_after_scene_build()
    for _ in range(sim["warmup_steps"]):
        scene.step()
    return scene, robot


def main():
    parser = argparse.ArgumentParser(
        description="Interactively calibrate a Nyx camera transform relative to a robot link."
    )
    parser.add_argument("--config", default=str(project_root() / "config/default.yaml"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    if not config.get("nyx_camera", {}).get("enabled", False):
        raise ValueError("nyx_camera.enabled must be true for calibration")
    if args.cpu:
        config["sim"]["backend"] = "cpu"

    scene, robot = _build_scene(config)
    parent_link = robot.entity.get_link(config["nyx_camera"]["parent_link"])
    running = True

    import genesis.utils.geom as gu
    import genesis.vis.keybindings as kb

    def hand_transform() -> np.ndarray:
        return gu.trans_quat_to_T(
            parent_link.get_pos().cpu().numpy(),
            parent_link.get_quat().cpu().numpy(),
        )

    def current_offset() -> np.ndarray:
        return np.linalg.inv(hand_transform()) @ scene.viewer.camera_pose

    def set_offset(offset_T: np.ndarray) -> None:
        scene.viewer.set_camera_pose(pose=hand_transform() @ offset_T)

    def nudge(local_translation=(0.0, 0.0, 0.0), rotation_axis=None, sign=1):
        offset_T = current_offset()
        offset_T[:3, 3] += offset_T[:3, :3] @ np.asarray(local_translation)
        if rotation_axis is not None:
            offset_T[:3, :3] = offset_T[:3, :3] @ _rotation_matrix(
                rotation_axis,
                sign * np.deg2rad(2.5),
            )
        set_offset(offset_T)

    def export_offset():
        print("\nPaste this into config/default.yaml:\n")
        print(_format_offset_yaml(current_offset()))
        print()

    def stop():
        nonlocal running
        running = False

    initial_offset_T = np.asarray(config["nyx_camera"]["offset_T"], dtype=np.float64)
    set_offset(initial_offset_T)
    scene.viewer.register_keybinds(
        kb.Keybind(
            "camera local x -",
            kb.Key.LEFT,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.CTRL,),
            callback=lambda: nudge(local_translation=(-0.005, 0.0, 0.0)),
        ),
        kb.Keybind(
            "camera local x +",
            kb.Key.RIGHT,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.CTRL,),
            callback=lambda: nudge(local_translation=(0.005, 0.0, 0.0)),
        ),
        kb.Keybind(
            "camera local y +",
            kb.Key.UP,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.CTRL,),
            callback=lambda: nudge(local_translation=(0.0, 0.005, 0.0)),
        ),
        kb.Keybind(
            "camera local y -",
            kb.Key.DOWN,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.CTRL,),
            callback=lambda: nudge(local_translation=(0.0, -0.005, 0.0)),
        ),
        kb.Keybind(
            "camera local z +",
            kb.Key.PAGEUP,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.CTRL,),
            callback=lambda: nudge(local_translation=(0.0, 0.0, 0.005)),
        ),
        kb.Keybind(
            "camera local z -",
            kb.Key.PAGEDOWN,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.CTRL,),
            callback=lambda: nudge(local_translation=(0.0, 0.0, -0.005)),
        ),
        kb.Keybind(
            "camera yaw +",
            kb.Key.LEFT,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.ALT,),
            callback=lambda: nudge(rotation_axis=1),
        ),
        kb.Keybind(
            "camera yaw -",
            kb.Key.RIGHT,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.ALT,),
            callback=lambda: nudge(rotation_axis=1, sign=-1),
        ),
        kb.Keybind(
            "camera pitch +",
            kb.Key.UP,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.ALT,),
            callback=lambda: nudge(rotation_axis=0),
        ),
        kb.Keybind(
            "camera pitch -",
            kb.Key.DOWN,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.ALT,),
            callback=lambda: nudge(rotation_axis=0, sign=-1),
        ),
        kb.Keybind(
            "camera roll +",
            kb.Key.PAGEUP,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.ALT,),
            callback=lambda: nudge(rotation_axis=2),
        ),
        kb.Keybind(
            "camera roll -",
            kb.Key.PAGEDOWN,
            kb.KeyAction.PRESS,
            key_mods=(kb.KeyMod.ALT,),
            callback=lambda: nudge(rotation_axis=2, sign=-1),
        ),
        kb.Keybind(
            "print wrist-camera offset",
            kb.Key.C,
            kb.KeyAction.PRESS,
            callback=export_offset,
        ),
        kb.Keybind(
            "quit calibration",
            kb.Key.ESCAPE,
            kb.KeyAction.RELEASE,
            callback=stop,
        ),
    )
    print(
        "Calibration viewer is ready. Ctrl+arrow/PageUp/PageDown translates the "
        "camera by 5 mm in its local frame. Alt+arrow/PageUp/PageDown changes "
        "yaw/pitch/roll by 2.5 degrees. Press C to print offset_T; Esc quits."
    )

    while running and scene.viewer.is_alive():
        scene.step()


if __name__ == "__main__":
    main()
