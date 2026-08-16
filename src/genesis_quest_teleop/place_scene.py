from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from .config import load_config, project_root
from .robots import create_robot_adapter


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Interactively place the grasp cube and Franka base."
    )
    parser.add_argument("--config", default=str(project_root() / "config/default.yaml"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cpu:
        config["sim"]["backend"] = "cpu"

    import genesis as gs
    import genesis.vis.keybindings as kb

    sim = config["sim"]
    gs.init(backend=gs.cuda if sim["backend"] == "gpu" else gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim["dt"]),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, -1.4, 1.1),
            camera_lookat=(0.45, 0.0, 0.35),
            camera_fov=40,
            enable_gui=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=True,
        ),
        show_viewer=True,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube_config = config.get("scene", {}).get("grasp_cube", {})
    cube = None
    if cube_config.get("enabled", False):
        pbr = cube_config.get("pbr", {})
        cube = scene.add_entity(
            morph=gs.morphs.Box(
                pos=tuple(cube_config["position"]),
                size=tuple(cube_config["size"]),
            ),
            material=gs.materials.Rigid(
                friction=cube_config["friction"],
                rho=cube_config["density"],
            ),
            surface=gs.surfaces.Default(
                color=tuple(cube_config["color"]),
                metallic=float(pbr.get("metallic", 0.0)),
                roughness=float(pbr.get("roughness", 0.5)),
            ),
        )
    # Use the same adapter factory as the teleop runtime so this utility works
    # with every robot type supported by the project (currently Franka/OpenArm).
    robot = create_robot_adapter(config)
    robot.build(scene)
    scene.build()
    robot.initialize_after_scene_build()

    def print_placement():
        robot_position = robot.entity.get_pos(relative=False).cpu().numpy()
        robot_quat_wxyz = robot.entity.get_quat(relative=False).cpu().numpy()
        robot_qpos = robot.entity.get_qpos().cpu().numpy()
        robot_euler = Rotation.from_quat(robot_quat_wxyz[[1, 2, 3, 0]]).as_euler(
            "xyz",
            degrees=True,
        )
        print("\nPaste these values into config/default.yaml:\n")
        print("robot:")
        print(f"  base_position: {_format_vector(robot_position)}")
        print(f"  base_euler_deg: {_format_vector(robot_euler)}")
        print(f"  home_qpos: {_format_vector(robot_qpos)}")
        if cube is not None:
            cube_position = cube.get_pos(relative=False).cpu().numpy()
            print("scene:")
            print("  grasp_cube:")
            print(f"    position: {_format_vector(cube_position)}")
        print()

    running = True

    def stop():
        nonlocal running
        running = False

    scene.viewer.register_keybinds(
        kb.Keybind("print scene placement", kb.Key.C, kb.KeyAction.PRESS, callback=print_placement),
        kb.Keybind("quit placement", kb.Key.ESCAPE, kb.KeyAction.RELEASE, callback=stop),
    )
    print(
        "Open the Entity Browser in the Genesis GUI, expand the cube or Franka, "
        "enable Gizmo, and choose Translate or Rotate. Press C to print YAML; "
        "press Esc to quit."
    )
    while running and scene.viewer.is_alive():
        scene.step()


if __name__ == "__main__":
    main()
