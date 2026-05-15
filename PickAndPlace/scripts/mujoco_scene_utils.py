#!/usr/bin/env python3

"""Utilities for building a MuJoCo pick-and-place scene that matches the Isaac Lab task."""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_URDF_PATH = Path("assets/mujoco/franka_panda/franka_description/robots/panda_arm_hand.urdf")
DEFAULT_SCENE_XML_PATH = Path("assets/mujoco/franka_pick_place_scene.xml")

GROUND_Z = -1.05
TABLE_CENTER = (0.5, 0.0, 0.0)
TABLE_TOP_HALF_SIZE = (0.55, 0.45, 0.025)
TABLE_TOP_CENTER_Z = -TABLE_TOP_HALF_SIZE[2]
TABLE_LEG_HALF_SIZE = (0.03, 0.03, 0.5)
TABLE_LEG_CENTER_Z = -0.55
TABLE_LEG_X_INSET = 0.08
TABLE_LEG_Y_INSET = 0.08

CUBE_SIZE = 0.05
CUBE_HALF_SIZE = CUBE_SIZE * 0.5
CUBE_START_POS = (0.45, 0.0, CUBE_HALF_SIZE)
CUBE_START_QUAT = (1.0, 0.0, 0.0, 0.0)
CUBE_SLIDING_FRICTION = 1.0

TARGET_BOX_CENTER_X = 0.58
TARGET_BOX_CENTER_Y = -0.36
TARGET_BOX_INNER_SIZE = 0.12
TARGET_BOX_WALL_THICKNESS = 0.01
TARGET_BOX_WALL_HEIGHT = 0.06
TARGET_BOX_FLOOR_THICKNESS = 0.01
TARGET_BOX_FLOOR_CENTER_Z = TARGET_BOX_FLOOR_THICKNESS * 0.5
TARGET_BOX_WALL_CENTER_Z = TARGET_BOX_FLOOR_THICKNESS + TARGET_BOX_WALL_HEIGHT * 0.5
TARGET_BOX_OUTER_SPAN = TARGET_BOX_INNER_SIZE + 2.0 * TARGET_BOX_WALL_THICKNESS
TARGET_BOX_PLACE_Z = TARGET_BOX_FLOOR_THICKNESS + CUBE_HALF_SIZE
TARGET_BOX_SLIDING_FRICTION = 1.0

WAYPOINT_POSE = (0.50, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0)
BOX_POSE = (TARGET_BOX_CENTER_X, TARGET_BOX_CENTER_Y, TARGET_BOX_PLACE_Z + 0.08, 1.0, 0.0, 0.0, 0.0)

DEFAULT_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
DEFAULT_GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
ROBOT_SELF_COLLISION_BODIES = [
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_leftfinger",
    "panda_rightfinger",
]
ALT_ROBOT_SELF_COLLISION_BODIES = [
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "left_finger",
    "right_finger",
]

# ARM_HOME_QPOS = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
ARM_HOME_QPOS = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
GRIPPER_OPEN_QPOS = (0.04, 0.04)
FULL_HOME_QPOS = ARM_HOME_QPOS + GRIPPER_OPEN_QPOS

ARM_JOINT_DAMPING = 4.0
ARM_JOINT_ARMATURE = 0.0
GRIPPER_JOINT_DAMPING = 100.0
GRIPPER_JOINT_ARMATURE = 0.0
ARM_ACTUATOR_KP = 80.0
GRIPPER_ACTUATOR_KP = 2000.0
ARM_JOINT_FORCE_LIMIT = 87.0
FOREARM_JOINT_FORCE_LIMIT = 12.0
GRIPPER_JOINT_FORCE_LIMIT = 200.0


def _import_mujoco():
    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'mujoco' Python package is required. Install it in the Python environment used to run this script."
        ) from exc
    return mujoco


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _vec_str(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _get_or_create(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _find_package_root(src_urdf: Path) -> Path | None:
    for candidate in (src_urdf.parent, *src_urdf.parents):
        if (candidate / "package.xml").exists():
            return candidate
    return None


def _sanitize_urdf_text_for_mujoco(urdf_text: str, src_urdf: Path) -> str:
    if (src_urdf.parent / "franka_description").exists():
        replacement_prefix = "franka_description/"
        package_name = "franka_description"
    else:
        package_root = _find_package_root(src_urdf)
        if package_root is not None:
            package_name = package_root.name
            replacement_prefix = os.path.relpath(package_root, src_urdf.parent).replace("\\", "/").rstrip("/") + "/"
        else:
            franka_description_root = next((parent for parent in src_urdf.parents if parent.name == "franka_description"), None)
            if franka_description_root is not None:
                package_name = "franka_description"
                replacement_prefix = (
                    os.path.relpath(franka_description_root, src_urdf.parent).replace("\\", "/").rstrip("/") + "/"
                )
            else:
                package_name = "franka_description"
                replacement_prefix = "franka_description/"

    urdf_text = urdf_text.replace(f"package://{package_name}/", replacement_prefix)
    if package_name != "franka_description":
        urdf_text = urdf_text.replace("package://franka_description/", replacement_prefix)
    if "package://" in urdf_text:
        raise ValueError(
            "Unsupported package:// URI remains in URDF after path sanitization. "
            f"Expected package '{package_name}' rooted at '{src_urdf.parent}'."
        )
    urdf_text = re.sub(
        re.escape(replacement_prefix) + r"meshes/visual/([A-Za-z0-9_]+)\.dae",
        replacement_prefix + r"meshes/collision/\1.stl",
        urdf_text,
    )
    return urdf_text


def _resolve_mjcf_mesh_root(src_xml_path: Path) -> Path:
    root = ET.parse(src_xml_path).getroot()
    compiler_elem = root.find("compiler")
    meshdir = compiler_elem.get("meshdir") if compiler_elem is not None else None
    if meshdir:
        return (src_xml_path.parent / meshdir).resolve()
    return src_xml_path.parent.resolve()


def _rewrite_compiled_mesh_paths(root: ET.Element, mesh_root: Path, output_xml_path: Path) -> None:
    xml_parent = output_xml_path.parent
    for mesh_elem in root.findall(".//mesh"):
        mesh_file = mesh_elem.get("file")
        if not mesh_file:
            continue
        mesh_path = Path(mesh_file)
        if mesh_path.is_absolute():
            continue
        target_mesh_path = (mesh_root / mesh_path).resolve()
        relative_mesh_path = os.path.relpath(target_mesh_path, xml_parent).replace("\\", "/")
        mesh_elem.set("file", relative_mesh_path)
    compiler_elem = root.find("compiler")
    if compiler_elem is not None and "meshdir" in compiler_elem.attrib:
        del compiler_elem.attrib["meshdir"]


def _stabilize_robot_dynamics(
    root: ET.Element,
    arm_joint_names: list[str] | tuple[str, ...],
    gripper_joint_names: list[str] | tuple[str, ...],
) -> None:
    arm_joint_name_set = set(arm_joint_names)
    gripper_joint_name_set = set(gripper_joint_names)
    for joint_elem in root.findall(".//joint"):
        joint_name = joint_elem.get("name", "")
        if joint_name in arm_joint_name_set:
            joint_elem.set("damping", f"{ARM_JOINT_DAMPING:.9g}")
            joint_elem.set("armature", f"{ARM_JOINT_ARMATURE:.9g}")
        elif joint_name in gripper_joint_name_set:
            joint_elem.set("damping", f"{GRIPPER_JOINT_DAMPING:.9g}")
            joint_elem.set("armature", f"{GRIPPER_JOINT_ARMATURE:.9g}")


def _remove_gripper_coupling_constraints(root: ET.Element) -> None:
    tendon_elem = root.find("tendon")
    if tendon_elem is not None:
        root.remove(tendon_elem)

    equality_elem = root.find("equality")
    if equality_elem is not None:
        root.remove(equality_elem)


def _normalize_parallel_gripper_axes(root: ET.Element, gripper_joint_names: list[str] | tuple[str, ...]) -> None:
    if len(gripper_joint_names) != 2:
        return
    joint_by_name = {joint_elem.get("name", ""): joint_elem for joint_elem in root.findall(".//joint")}
    left_joint = joint_by_name.get(gripper_joint_names[0])
    right_joint = joint_by_name.get(gripper_joint_names[1])
    if left_joint is not None:
        left_joint.set("axis", "0 1 0")
    if right_joint is not None:
        right_joint.set("axis", "0 -1 0")


def _mirror_right_gripper_collision_geoms(root: ET.Element) -> None:
    for right_body_name in ("right_finger", "panda_rightfinger"):
        for body_elem in root.findall(f".//body[@name='{right_body_name}']"):
            for geom_elem in body_elem.findall("geom"):
                pos_attr = geom_elem.get("pos")
                if not pos_attr:
                    continue
                values = pos_attr.split()
                if len(values) != 3:
                    continue
                try:
                    x, y, z = (float(value) for value in values)
                except ValueError:
                    continue
                if y > 0.0:
                    geom_elem.set("pos", _vec_str((x, -y, z)))


def _disable_base_collision(worldbody: ET.Element) -> None:
    for geom_elem in worldbody.findall("./geom"):
        if geom_elem.get("mesh") == "link0":
            geom_elem.set("contype", "0")
            geom_elem.set("conaffinity", "0")
            break


def _robot_self_collision_bodies_for_root(root: ET.Element) -> list[str]:
    existing_body_names = {body_elem.get("name", "") for body_elem in root.findall(".//body")}
    if set(ROBOT_SELF_COLLISION_BODIES).issubset(existing_body_names):
        return ROBOT_SELF_COLLISION_BODIES
    if set(ALT_ROBOT_SELF_COLLISION_BODIES).issubset(existing_body_names):
        return ALT_ROBOT_SELF_COLLISION_BODIES
    return []


def _add_robot_self_collision_excludes(root: ET.Element) -> None:
    collision_bodies = _robot_self_collision_bodies_for_root(root)
    if not collision_bodies:
        return
    contact_elem = _get_or_create(root, "contact")
    known_robot_bodies = set(ROBOT_SELF_COLLISION_BODIES) | set(ALT_ROBOT_SELF_COLLISION_BODIES) | {"link0"}
    for exclude_elem in list(contact_elem.findall("exclude")):
        body1 = exclude_elem.get("body1", "")
        body2 = exclude_elem.get("body2", "")
        if body1 in known_robot_bodies or body2 in known_robot_bodies:
            contact_elem.remove(exclude_elem)

    existing_body_names = {body_elem.get("name", "") for body_elem in root.findall(".//body")}
    if "link0" in existing_body_names and collision_bodies and collision_bodies[0] in existing_body_names:
        ET.SubElement(contact_elem, "exclude", {"body1": "link0", "body2": collision_bodies[0]})

    for i, body1 in enumerate(collision_bodies):
        for body2 in collision_bodies[i + 1 :]:
            ET.SubElement(contact_elem, "exclude", {"body1": body1, "body2": body2})


def _write_temp_urdf(src_urdf: Path) -> Path:
    sanitized_text = _sanitize_urdf_text_for_mujoco(src_urdf.read_text(encoding="utf-8"), src_urdf=src_urdf)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".mujoco_tmp.urdf",
        prefix=src_urdf.stem + "_",
        dir=src_urdf.parent,
        delete=False,
        encoding="utf-8",
    )
    with temp_file:
        temp_file.write(sanitized_text)
    return Path(temp_file.name)


def _compile_robot_tree_from_urdf(mujoco, src_urdf_path: Path, compiled_xml_path: Path) -> tuple[object, ET.ElementTree, Path]:
    temp_urdf_path = _write_temp_urdf(src_urdf_path)
    try:
        model = mujoco.MjModel.from_xml_path(str(temp_urdf_path))
        if not hasattr(mujoco, "mj_saveLastXML"):
            raise RuntimeError("This MuJoCo build does not expose mj_saveLastXML(), so URDF -> MJCF export is unavailable.")
        mujoco.mj_saveLastXML(str(compiled_xml_path), model)
        tree = ET.parse(compiled_xml_path)
    finally:
        temp_urdf_path.unlink(missing_ok=True)
        compiled_xml_path.unlink(missing_ok=True)
    return model, tree, src_urdf_path.parent.resolve()


def _compile_robot_tree_from_mjcf(mujoco, src_mjcf_path: Path, compiled_xml_path: Path) -> tuple[object, ET.ElementTree, Path]:
    if not hasattr(mujoco, "mj_saveLastXML"):
        raise RuntimeError("This MuJoCo build does not expose mj_saveLastXML(), so MJCF recompilation is unavailable.")
    model = mujoco.MjModel.from_xml_path(str(src_mjcf_path))
    mujoco.mj_saveLastXML(str(compiled_xml_path), model)
    tree = ET.parse(compiled_xml_path)
    compiled_xml_path.unlink(missing_ok=True)
    return model, tree, _resolve_mjcf_mesh_root(src_mjcf_path)


def _append_static_box(
    worldbody: ET.Element,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    friction: tuple[float, float, float] = (1.0, 0.01, 0.001),
) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": _vec_str(pos),
            "size": _vec_str(size),
            "rgba": _vec_str(rgba),
            "friction": _vec_str(friction),
            "contype": "1",
            "conaffinity": "1",
        },
    )


def _arm_joint_force_limit(joint_name: str) -> float:
    if joint_name.endswith(("5", "6", "7")):
        return FOREARM_JOINT_FORCE_LIMIT
    return ARM_JOINT_FORCE_LIMIT


def build_pick_place_mujoco_scene(
    urdf_path: str | Path = DEFAULT_URDF_PATH,
    output_xml_path: str | Path = DEFAULT_SCENE_XML_PATH,
    timestep: float = 0.01,
    input_format: str = "urdf",
    arm_joint_names: list[str] | tuple[str, ...] = DEFAULT_ARM_JOINT_NAMES,
    gripper_joint_names: list[str] | tuple[str, ...] = DEFAULT_GRIPPER_JOINT_NAMES,
) -> Path:
    mujoco = _import_mujoco()

    input_path = resolve_repo_path(urdf_path)
    output_xml_path = resolve_repo_path(output_xml_path)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)

    compiled_xml_path = output_xml_path.with_name(output_xml_path.stem + "_compiled_tmp.xml")
    if not input_path.exists():
        raise FileNotFoundError(f"Robot model was not found: {input_path}")

    input_format = input_format.lower()
    if input_format == "urdf":
        model, tree, mesh_root = _compile_robot_tree_from_urdf(mujoco, input_path, compiled_xml_path)
    elif input_format == "mjcf":
        model, tree, mesh_root = _compile_robot_tree_from_mjcf(mujoco, input_path, compiled_xml_path)
    else:
        raise ValueError(f"Unsupported robot input format: {input_format}")

    root = tree.getroot()
    root.set("model", "franka_pick_place")
    _rewrite_compiled_mesh_paths(root, mesh_root=mesh_root, output_xml_path=output_xml_path)
    _stabilize_robot_dynamics(root, arm_joint_names=arm_joint_names, gripper_joint_names=gripper_joint_names)
    _remove_gripper_coupling_constraints(root)
    _normalize_parallel_gripper_axes(root, gripper_joint_names=gripper_joint_names)
    _mirror_right_gripper_collision_geoms(root)
    _add_robot_self_collision_excludes(root)

    option = _get_or_create(root, "option")
    option.set("timestep", f"{timestep:.9g}")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    option.set("iterations", "50")

    visual = _get_or_create(root, "visual")
    map_elem = _get_or_create(visual, "map")
    map_elem.set("znear", "0.01")
    map_elem.set("zfar", "10")

    asset = _get_or_create(root, "asset")
    ET.SubElement(asset, "material", {"name": "table_mat", "rgba": "0.38 0.30 0.18 1"})
    ET.SubElement(asset, "material", {"name": "cube_mat", "rgba": "0.0 1.0 0.0 1"})
    ET.SubElement(asset, "material", {"name": "target_floor_mat", "rgba": "0.72 0.70 0.62 1"})
    ET.SubElement(asset, "material", {"name": "target_wall_mat", "rgba": "0.55 0.43 0.28 1"})

    worldbody = _get_or_create(root, "worldbody")
    _disable_base_collision(worldbody)
    ET.SubElement(worldbody, "light", {"name": "sun", "mode": "fixed", "pos": "1.0 -1.0 2.0", "dir": "-0.3 0.2 -1.0"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.0 -1.0 0.8", "xyaxes": "1 0 0 0 0 1"})

    _append_static_box(
        worldbody,
        name="ground",
        pos=(0.0, 0.0, GROUND_Z - 0.01),
        size=(2.0, 2.0, 0.01),
        rgba=(0.20, 0.20, 0.20, 1.0),
        friction=(1.0, 0.01, 0.001),
    )
    _append_static_box(
        worldbody,
        name="table_top",
        pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_CENTER_Z),
        size=TABLE_TOP_HALF_SIZE,
        rgba=(0.38, 0.30, 0.18, 1.0),
        friction=(1.0, 0.01, 0.001),
    )
    leg_x = TABLE_TOP_HALF_SIZE[0] - TABLE_LEG_X_INSET
    leg_y = TABLE_TOP_HALF_SIZE[1] - TABLE_LEG_Y_INSET
    for name, offset_x, offset_y in [
        ("table_leg_front_left", -leg_x, leg_y),
        ("table_leg_front_right", -leg_x, -leg_y),
        ("table_leg_back_left", leg_x, leg_y),
        ("table_leg_back_right", leg_x, -leg_y),
    ]:
        _append_static_box(
            worldbody,
            name=name,
            pos=(TABLE_CENTER[0] + offset_x, TABLE_CENTER[1] + offset_y, TABLE_LEG_CENTER_Z),
            size=TABLE_LEG_HALF_SIZE,
            rgba=(0.25, 0.25, 0.25, 1.0),
            friction=(1.0, 0.01, 0.001),
        )

    cube_body = ET.SubElement(worldbody, "body", {"name": "pick_cube", "pos": _vec_str(CUBE_START_POS)})
    ET.SubElement(cube_body, "freejoint", {"name": "pick_cube_freejoint"})
    ET.SubElement(
        cube_body,
        "geom",
        {
            "name": "pick_cube_geom",
            "type": "box",
            "size": _vec_str((CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE)),
            "rgba": "0.0 1.0 0.0 1.0",
            "friction": f"{CUBE_SLIDING_FRICTION:.9g} 0.01 0.001",
            "contype": "1",
            "conaffinity": "1",
        },
    )

    _append_static_box(
        worldbody,
        name="target_box_floor",
        pos=(TARGET_BOX_CENTER_X, TARGET_BOX_CENTER_Y, TARGET_BOX_FLOOR_CENTER_Z),
        size=(TARGET_BOX_OUTER_SPAN * 0.5, TARGET_BOX_OUTER_SPAN * 0.5, TARGET_BOX_FLOOR_THICKNESS * 0.5),
        rgba=(0.72, 0.70, 0.62, 1.0),
        friction=(TARGET_BOX_SLIDING_FRICTION, 0.01, 0.001),
    )
    _append_static_box(
        worldbody,
        name="target_box_left_wall",
        pos=(
            TARGET_BOX_CENTER_X - (TARGET_BOX_INNER_SIZE * 0.5 + TARGET_BOX_WALL_THICKNESS * 0.5),
            TARGET_BOX_CENTER_Y,
            TARGET_BOX_WALL_CENTER_Z,
        ),
        size=(TARGET_BOX_WALL_THICKNESS * 0.5, TARGET_BOX_OUTER_SPAN * 0.5, TARGET_BOX_WALL_HEIGHT * 0.5),
        rgba=(0.55, 0.43, 0.28, 1.0),
        friction=(TARGET_BOX_SLIDING_FRICTION, 0.01, 0.001),
    )
    _append_static_box(
        worldbody,
        name="target_box_right_wall",
        pos=(
            TARGET_BOX_CENTER_X + (TARGET_BOX_INNER_SIZE * 0.5 + TARGET_BOX_WALL_THICKNESS * 0.5),
            TARGET_BOX_CENTER_Y,
            TARGET_BOX_WALL_CENTER_Z,
        ),
        size=(TARGET_BOX_WALL_THICKNESS * 0.5, TARGET_BOX_OUTER_SPAN * 0.5, TARGET_BOX_WALL_HEIGHT * 0.5),
        rgba=(0.55, 0.43, 0.28, 1.0),
        friction=(TARGET_BOX_SLIDING_FRICTION, 0.01, 0.001),
    )
    _append_static_box(
        worldbody,
        name="target_box_front_wall",
        pos=(
            TARGET_BOX_CENTER_X,
            TARGET_BOX_CENTER_Y + (TARGET_BOX_INNER_SIZE * 0.5 + TARGET_BOX_WALL_THICKNESS * 0.5),
            TARGET_BOX_WALL_CENTER_Z,
        ),
        size=(TARGET_BOX_OUTER_SPAN * 0.5, TARGET_BOX_WALL_THICKNESS * 0.5, TARGET_BOX_WALL_HEIGHT * 0.5),
        rgba=(0.55, 0.43, 0.28, 1.0),
        friction=(TARGET_BOX_SLIDING_FRICTION, 0.01, 0.001),
    )
    _append_static_box(
        worldbody,
        name="target_box_back_wall",
        pos=(
            TARGET_BOX_CENTER_X,
            TARGET_BOX_CENTER_Y - (TARGET_BOX_INNER_SIZE * 0.5 + TARGET_BOX_WALL_THICKNESS * 0.5),
            TARGET_BOX_WALL_CENTER_Z,
        ),
        size=(TARGET_BOX_OUTER_SPAN * 0.5, TARGET_BOX_WALL_THICKNESS * 0.5, TARGET_BOX_WALL_HEIGHT * 0.5),
        rgba=(0.55, 0.43, 0.28, 1.0),
        friction=(TARGET_BOX_SLIDING_FRICTION, 0.01, 0.001),
    )

    ET.SubElement(worldbody, "site", {"name": "waypoint_marker", "type": "sphere", "pos": _vec_str(WAYPOINT_POSE[:3]), "size": "0.012", "rgba": "0.2 0.6 1.0 0.4"})
    ET.SubElement(worldbody, "site", {"name": "goal_marker", "type": "sphere", "pos": _vec_str(BOX_POSE[:3]), "size": "0.012", "rgba": "1.0 0.4 0.2 0.4"})

    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")

    for joint_name in arm_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' was not found in the compiled Franka model.")
        lower, upper = model.jnt_range[joint_id]
        effort_limit = _arm_joint_force_limit(joint_name)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": joint_name,
                "joint": joint_name,
                "kp": f"{ARM_ACTUATOR_KP:.9g}",
                "ctrllimited": "true",
                "ctrlrange": f"{lower:.9g} {upper:.9g}",
                "forcelimited": "true",
                "forcerange": f"-{effort_limit:.9g} {effort_limit:.9g}",
            },
        )

    for joint_name in gripper_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' was not found in the compiled Franka model.")
        lower, upper = model.jnt_range[joint_id]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": joint_name,
                "joint": joint_name,
                "kp": f"{GRIPPER_ACTUATOR_KP:.9g}",
                "ctrllimited": "true",
                "ctrlrange": f"{lower:.9g} {upper:.9g}",
                "forcelimited": "true",
                "forcerange": f"-{GRIPPER_JOINT_FORCE_LIMIT:.9g} {GRIPPER_JOINT_FORCE_LIMIT:.9g}",
            },
        )

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    return output_xml_path
