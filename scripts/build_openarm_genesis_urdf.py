"""Generate a Genesis-compatible OpenArm URDF with mesh-derived finger inertia."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF = REPOSITORY_ROOT.parent / "telesim/robot_configs/openarm_config/openarm_bimanual.urdf"
DESCRIPTION_ROOT = REPOSITORY_ROOT.parent / "openarm_description"
OUTPUT_URDF = REPOSITORY_ROOT / "assets/openarm/openarm_bimanual_genesis.urdf"
FINGER_LINKS = (
    "openarm_left_left_finger",
    "openarm_left_right_finger",
    "openarm_right_left_finger",
    "openarm_right_right_finger",
)
EXPECTED_FINGER_TOPOLOGY = {
    "openarm_left_finger_joint1": ("openarm_left_hand", "openarm_left_right_finger"),
    "openarm_left_finger_joint2": ("openarm_left_hand", "openarm_left_left_finger"),
    "openarm_right_finger_joint1": ("openarm_right_hand", "openarm_right_right_finger"),
    "openarm_right_finger_joint2": ("openarm_right_hand", "openarm_right_left_finger"),
}
TOLERANCE = 1e-12


def parse_floats(value: str | None, count: int, default: str) -> np.ndarray:
    values = (value or default).split()
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {values!r}.")
    return np.array([float(item) for item in values], dtype=float)


def origin_transform(origin: ET.Element | None) -> np.ndarray:
    xyz = parse_floats(None if origin is None else origin.get("xyz"), 3, "0 0 0")
    rpy = parse_floats(None if origin is None else origin.get("rpy"), 3, "0 0 0")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def mesh_path(filename: str) -> Path:
    prefix = "package://openarm_description/"
    if not filename.startswith(prefix):
        raise ValueError(f"Unsupported finger mesh URI: {filename}")
    path = DESCRIPTION_ROOT / filename.removeprefix(prefix)
    if not path.is_file():
        raise FileNotFoundError(f"Finger collision mesh not found: {path}")
    return path


def validate_inertia(mass: float, com: np.ndarray, inertia: np.ndarray, link_name: str) -> np.ndarray:
    if not np.isfinite(mass) or mass <= 0 or not np.all(np.isfinite(com)) or not np.all(np.isfinite(inertia)):
        raise ValueError(f"{link_name}: non-finite or non-positive mass property.")
    if not np.allclose(inertia, inertia.T, atol=TOLERANCE):
        raise ValueError(f"{link_name}: inertia matrix is not symmetric.")
    principal = np.linalg.eigvalsh(inertia)
    if np.any(principal <= TOLERANCE):
        raise ValueError(f"{link_name}: inertia is not positive definite: {principal}")
    if principal[0] + principal[1] < principal[2] - TOLERANCE:
        raise ValueError(f"{link_name}: inertia violates the rigid-body triangle inequality: {principal}")
    return principal


def derived_properties(link: ET.Element) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    link_name = link.attrib["name"]
    inertial = link.find("inertial")
    collision = link.find("collision")
    if inertial is None or collision is None:
        raise ValueError(f"{link_name}: expected inertial and collision elements.")
    mass_element = inertial.find("mass")
    mesh = collision.find("geometry/mesh")
    if mass_element is None or mesh is None:
        raise ValueError(f"{link_name}: expected source mass and collision mesh.")

    mass = float(mass_element.attrib["value"])
    collision_mesh = trimesh.load_mesh(mesh_path(mesh.attrib["filename"]), process=False)
    if isinstance(collision_mesh, trimesh.Scene):
        collision_mesh = collision_mesh.dump(concatenate=True)
    if not isinstance(collision_mesh, trimesh.Trimesh):
        raise TypeError(f"{link_name}: collision mesh did not load as Trimesh.")
    collision_mesh = collision_mesh.copy()
    scale = parse_floats(mesh.get("scale"), 3, "1 1 1")
    scale_transform = np.eye(4)
    scale_transform[:3, :3] = np.diag(scale)
    collision_mesh.apply_transform(scale_transform)
    collision_mesh.apply_transform(origin_transform(collision.find("origin")))
    if not collision_mesh.is_watertight:
        # The supplied finger STL is an open collision shell. Its convex hull is a
        # deterministic, watertight volume made solely from that collision geometry.
        collision_mesh = collision_mesh.convex_hull
        print("  collision mesh is open; using its watertight convex hull for volume properties")
    if collision_mesh.volume < 0:
        collision_mesh.invert()
    if collision_mesh.volume <= 0:
        raise ValueError(f"{link_name}: collision geometry has non-positive volume {collision_mesh.volume}.")

    properties = collision_mesh.mass_properties
    mesh_mass = float(properties["mass"])
    if not np.isfinite(mesh_mass) or mesh_mass <= 0:
        raise ValueError(f"{link_name}: collision mesh has invalid temporary mass {mesh_mass}.")
    com = np.asarray(properties["center_mass"], dtype=float)
    inertia = np.asarray(properties["inertia"], dtype=float) * (mass / mesh_mass)
    principal = validate_inertia(mass, com, inertia, link_name)
    return mass, com, inertia, principal


def inertial_xml(mass: float, com: np.ndarray, inertia: np.ndarray) -> str:
    values = {
        "mass": f"{mass:.17g}",
        "xyz": " ".join(f"{value:.17g}" for value in com),
        "ixx": f"{inertia[0, 0]:.17g}",
        "ixy": f"{inertia[0, 1]:.17g}",
        "ixz": f"{inertia[0, 2]:.17g}",
        "iyy": f"{inertia[1, 1]:.17g}",
        "iyz": f"{inertia[1, 2]:.17g}",
        "izz": f"{inertia[2, 2]:.17g}",
    }
    return (
        "    <inertial>\n"
        f"      <origin xyz=\"{values['xyz']}\" rpy=\"0 0 0\"/>\n"
        f"      <mass value=\"{values['mass']}\"/>\n"
        "      <inertia "
        f"ixx=\"{values['ixx']}\" ixy=\"{values['ixy']}\" ixz=\"{values['ixz']}\" "
        f"iyy=\"{values['iyy']}\" iyz=\"{values['iyz']}\" izz=\"{values['izz']}\"/>\n"
        "    </inertial>"
    )


def replace_link_inertial(source: str, link_name: str, replacement: str) -> str:
    link_pattern = rf'(<link name="{re.escape(link_name)}">.*?</link>)'
    link_match = re.search(link_pattern, source, flags=re.DOTALL)
    if link_match is None:
        raise ValueError(f"Could not find link '{link_name}' in source URDF.")
    link_text = link_match.group(1)
    changed_link, count = re.subn(r"    <inertial>.*?    </inertial>", replacement, link_text, count=1, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"{link_name}: expected exactly one inertial block.")
    return source[: link_match.start(1)] + changed_link + source[link_match.end(1) :]


def remove_inertial(link: ET.Element) -> ET.Element:
    clone = ET.fromstring(ET.tostring(link, encoding="unicode"))
    inertial = clone.find("inertial")
    if inertial is not None:
        clone.remove(inertial)
    return clone


def xml_equal(lhs: ET.Element, rhs: ET.Element) -> bool:
    """Compare XML content while ignoring formatting-only whitespace."""
    def normalized(element: ET.Element) -> ET.Element:
        clone = ET.fromstring(ET.tostring(element, encoding="unicode"))
        for node in clone.iter():
            if node.text is not None and not node.text.strip():
                node.text = None
            if node.tail is not None and not node.tail.strip():
                node.tail = None
        return clone

    return ET.tostring(normalized(lhs)) == ET.tostring(normalized(rhs))


def verify_only_finger_inertials_changed(source_text: str, generated_text: str) -> None:
    source = ET.fromstring(source_text)
    generated = ET.fromstring(generated_text)
    source_links = {link.attrib["name"]: link for link in source.findall("link")}
    generated_links = {link.attrib["name"]: link for link in generated.findall("link")}
    if source_links.keys() != generated_links.keys():
        raise ValueError("Generated URDF changed the link set.")
    for name, source_link in source_links.items():
        generated_link = generated_links[name]
        if name in FINGER_LINKS:
            if not xml_equal(remove_inertial(source_link), remove_inertial(generated_link)):
                raise ValueError(f"{name}: generated URDF changed more than its inertial block.")
        elif not xml_equal(source_link, generated_link):
            raise ValueError(f"{name}: non-finger link was modified.")
    source_joints = {joint.attrib["name"]: joint for joint in source.findall("joint")}
    generated_joints = {joint.attrib["name"]: joint for joint in generated.findall("joint")}
    if source_joints.keys() != generated_joints.keys():
        raise ValueError("Generated URDF changed the joint set.")
    for name, source_joint in source_joints.items():
        generated_joint = generated_joints[name]
        if not xml_equal(source_joint, generated_joint):
            raise ValueError(f"{name}: generated URDF changed joint semantics.")


def validate_finger_topology(source_root: ET.Element) -> None:
    joints = {joint.attrib["name"]: joint for joint in source_root.findall("joint")}
    for joint_name, (parent_name, child_name) in EXPECTED_FINGER_TOPOLOGY.items():
        joint = joints.get(joint_name)
        if joint is None:
            raise ValueError(f"Expected finger joint missing from source URDF: {joint_name}")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None or (parent.get("link"), child.get("link")) != (parent_name, child_name):
            raise ValueError(
                f"{joint_name}: source topology is not canonical; expected "
                f"{parent_name} -> {child_name}."
            )


def main() -> None:
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(f"OpenArm source URDF not found: {SOURCE_URDF}")
    if not DESCRIPTION_ROOT.is_dir():
        raise FileNotFoundError(f"openarm_description package not found: {DESCRIPTION_ROOT}")
    source_text = SOURCE_URDF.read_text()
    source_root = ET.fromstring(source_text)
    validate_finger_topology(source_root)
    links = {link.attrib["name"]: link for link in source_root.findall("link")}
    if set(FINGER_LINKS) - links.keys():
        raise ValueError(f"Expected finger links missing from source URDF: {set(FINGER_LINKS) - links.keys()}")

    generated_text = source_text
    for link_name in FINGER_LINKS:
        mass, com, inertia, principal = derived_properties(links[link_name])
        print(f"{link_name}: source mass={mass:.17g} kg")
        print("  computed COM:", com)
        print("  inertia matrix:\n", inertia)
        print("  principal moments:", principal)
        print("  physical validation: PASS")
        generated_text = replace_link_inertial(generated_text, link_name, inertial_xml(mass, com, inertia))
    ET.fromstring(generated_text)
    verify_only_finger_inertials_changed(source_text, generated_text)
    OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_URDF.write_text(generated_text)
    print(f"Wrote Genesis-compatible URDF: {OUTPUT_URDF}")


if __name__ == "__main__":
    main()
