"""Print the legacy and MuJoCo URDF parser link order for OpenArm.

This intentionally calls the same parser functions as RigidEntity._parse_scene;
it does not reproduce either parser's ordering logic.
"""

from __future__ import annotations

from pathlib import Path

import genesis as gs
from genesis.utils import mjcf as mju
from genesis.utils import urdf as uu


URDF_PATH = Path(__file__).resolve().parents[1] / "assets/openarm/openarm_bimanual_genesis.urdf"
FINGER_NAMES = {
    "openarm_left_left_finger",
    "openarm_left_right_finger",
    "openarm_right_left_finger",
    "openarm_right_right_finger",
}


def main() -> None:
    gs.init(backend=gs.cpu)
    morph = gs.morphs.URDF(
        file=str(URDF_PATH),
        fixed=True,
        merge_fixed_links=True,
        links_to_keep=("openarm_left_hand", "openarm_right_hand"),
        recompute_inertia=False,
    )
    surface = gs.surfaces.Default()
    legacy_l_infos, _, _, _ = uu.parse_urdf(morph, surface)
    mj_l_infos, _, _, _ = mju.parse_xml(morph.model_copy(update=dict(visualization=False)), surface)

    legacy_names = [l_info["name"] for l_info in legacy_l_infos]
    mj_names = [l_info["name"] for l_info in mj_l_infos]
    print(f"{'INDEX':<7} | {'LEGACY LINK':<38} | MUJOCO LINK")
    print("-" * 90)
    for index in range(max(len(legacy_names), len(mj_names))):
        legacy = legacy_names[index] if index < len(legacy_names) else "<missing>"
        mujoco = mj_names[index] if index < len(mj_names) else "<missing>"
        marker = "  <== FINGER" if legacy in FINGER_NAMES or mujoco in FINGER_NAMES else ""
        print(f"{index:<7} | {legacy:<38} | {mujoco}{marker}")

    legacy_set, mj_set = set(legacy_names), set(mj_names)
    print("\nlegacy-only:", sorted(legacy_set - mj_set))
    print("MuJoCo-only:", sorted(mj_set - legacy_set))
    print("same semantic link names (except permitted virtual world):", legacy_set - {"world"} == mj_set - {"world"})
    for finger_name in sorted(FINGER_NAMES):
        print(
            f"{finger_name}: legacy index={legacy_names.index(finger_name)}, "
            f"MuJoCo index={mj_names.index(finger_name)}"
        )


if __name__ == "__main__":
    main()
