#!/usr/bin/env python3

"""Run an Isaac Lab teacher ONNX policy in MuJoCo with a simple GUI controller."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mujoco_scene_utils import (
    ARM_HOME_QPOS,
    BOX_POSE,
    CUBE_START_POS,
    CUBE_START_QUAT,
    DEFAULT_GRIPPER_JOINT_NAMES,
    DEFAULT_SCENE_XML_PATH,
    DEFAULT_URDF_PATH,
    DEFAULT_ARM_JOINT_NAMES,
    FULL_HOME_QPOS,
    GRIPPER_OPEN_QPOS,
    WAYPOINT_POSE,
    build_pick_place_mujoco_scene,
    resolve_repo_path,
)


def _import_mujoco():
    try:
        import mujoco  # type: ignore
        import mujoco.viewer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'mujoco' Python package is required. Install it in the Python environment used to run this script."
        ) from exc
    return mujoco


def _import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'onnxruntime' Python package is required. Install it in the Python environment used to run this script."
        ) from exc
    return ort


def _import_tkinter():
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError("tkinter is required for the GUI control window.") from exc
    return tk


DEFAULT_ARM_ACTUATOR_NAMES = DEFAULT_ARM_JOINT_NAMES
DEFAULT_GRIPPER_ACTUATOR_NAMES = DEFAULT_GRIPPER_JOINT_NAMES
DEFAULT_BOX_POSE = np.array(BOX_POSE, dtype=np.float32)
DEFAULT_WAYPOINT_POSE = np.array(WAYPOINT_POSE, dtype=np.float32)
DEFAULT_ARM_OFFSET = np.array(ARM_HOME_QPOS, dtype=np.float32)
DEFAULT_FULL_QPOS_OFFSET = np.array(FULL_HOME_QPOS, dtype=np.float32)
DEFAULT_GRIPPER_OPEN = float(GRIPPER_OPEN_QPOS[0])
DEFAULT_GRIPPER_CLOSE = 0.0
DEFAULT_OBJECT_QPOS = np.array(CUBE_START_POS + CUBE_START_QUAT, dtype=np.float32)
DEFAULT_OBJECT_RESET_X_RANGE = (-0.1, 0.1)
DEFAULT_OBJECT_RESET_Y_RANGE = (-0.25, 0.25)
DEFAULT_OBJECT_RESET_Z_RANGE = (0.0, 0.0)
FRANKA_EMIKA_PANDA_MJCF_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
FRANKA_EMIKA_PANDA_MJCF_GRIPPER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]
FRANKA_EMIKA_PANDA_MJCF_ARM_ACTUATOR_NAMES = FRANKA_EMIKA_PANDA_MJCF_ARM_JOINT_NAMES
FRANKA_EMIKA_PANDA_MJCF_GRIPPER_ACTUATOR_NAMES = FRANKA_EMIKA_PANDA_MJCF_GRIPPER_JOINT_NAMES
# Match Isaac Lab FRANKA_PANDA_CFG initial joint positions used by the PickAndPlace task.
FRANKA_EMIKA_PANDA_MJCF_ARM_HOME_QPOS = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], dtype=np.float32)
FRANKA_EMIKA_PANDA_MJCF_FULL_HOME_QPOS = np.array(
    [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04], dtype=np.float32
)


def parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True, help="Path to exported Isaac Lab PPO policy.onnx.")
    parser.add_argument(
        "--mjcf_path",
        type=str,
        default=None,
        help=(
            "Optional robot MJCF XML path, for example '/home/.../franka_emika_panda/panda.xml'. "
            "When set, it is used instead of --robot_description or --urdf_path."
        ),
    )
    parser.add_argument(
        "--robot_description",
        type=str,
        default=None,
        help=(
            "Optional robot_descriptions module name, for example 'panda_description'. "
            "When set, its URDF path is used instead of --urdf_path."
        ),
    )
    parser.add_argument(
        "--robot_description_attr",
        type=str,
        default="URDF_PATH",
        help="Attribute to read from --robot_description. Defaults to URDF_PATH.",
    )
    parser.add_argument(
        "--robot_preset",
        type=str,
        default="default",
        choices=["default", "franka_emika_panda_mjcf"],
        help="Apply robot-specific defaults for joint names, actuator names, root body, and home pose.",
    )
    parser.add_argument("--urdf_path", type=str, default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--scene_xml", type=str, default=str(DEFAULT_SCENE_XML_PATH))
    parser.add_argument("--rebuild_scene", action="store_true", help="Rebuild the MuJoCo scene XML before launching.")
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--robot_root_body", type=str, default="panda_link0")
    parser.add_argument("--object_body", type=str, default="pick_cube")
    parser.add_argument("--object_freejoint", type=str, default="pick_cube_freejoint")
    parser.add_argument("--arm_joint_names", type=parse_name_list, default=DEFAULT_ARM_JOINT_NAMES)
    parser.add_argument("--gripper_joint_names", type=parse_name_list, default=DEFAULT_GRIPPER_JOINT_NAMES)
    parser.add_argument("--arm_actuator_names", type=parse_name_list, default=DEFAULT_ARM_ACTUATOR_NAMES)
    parser.add_argument("--gripper_actuator_names", type=parse_name_list, default=DEFAULT_GRIPPER_ACTUATOR_NAMES)
    parser.add_argument("--arm_default_offset", type=float, nargs=7, default=DEFAULT_ARM_OFFSET.tolist())
    parser.add_argument("--full_qpos_default_offset", type=float, nargs=9, default=DEFAULT_FULL_QPOS_OFFSET.tolist())
    parser.add_argument("--waypoint_pose", type=float, nargs=7, default=DEFAULT_WAYPOINT_POSE.tolist())
    parser.add_argument("--box_pose", type=float, nargs=7, default=DEFAULT_BOX_POSE.tolist())
    parser.add_argument("--arm_action_scale", type=float, default=0.5)
    parser.add_argument("--gripper_open_pos", type=float, default=DEFAULT_GRIPPER_OPEN)
    parser.add_argument("--gripper_close_pos", type=float, default=DEFAULT_GRIPPER_CLOSE)
    parser.add_argument("--minimal_height", type=float, default=0.04)
    parser.add_argument("--waypoint_xy_threshold", type=float, default=0.03)
    parser.add_argument("--waypoint_z_threshold", type=float, default=0.03)
    parser.add_argument("--waypoint_max_speed", type=float, default=0.05)
    parser.add_argument("--control_rate_hz", type=float, default=50.0)
    parser.add_argument(
        "--object_reset_x_range",
        type=float,
        nargs=2,
        default=list(DEFAULT_OBJECT_RESET_X_RANGE),
        help="Object reset x-offset range relative to the default cube pose. Matches Isaac pick-and-place by default.",
    )
    parser.add_argument(
        "--object_reset_y_range",
        type=float,
        nargs=2,
        default=list(DEFAULT_OBJECT_RESET_Y_RANGE),
        help="Object reset y-offset range relative to the default cube pose. Matches Isaac pick-and-place by default.",
    )
    parser.add_argument(
        "--object_reset_z_range",
        type=float,
        nargs=2,
        default=list(DEFAULT_OBJECT_RESET_Z_RANGE),
        help="Object reset z-offset range relative to the default cube pose. Matches Isaac pick-and-place by default.",
    )
    parser.add_argument(
        "--force_close_gripper_in_policy",
        action="store_true",
        help="Override the policy gripper output and keep the gripper closed during policy mode.",
    )
    parser.add_argument("--home_tolerance", type=float, default=0.03)
    return parser.parse_args()


def apply_robot_preset(args):
    if args.robot_preset != "franka_emika_panda_mjcf":
        return args
    args.robot_root_body = "link0"
    args.arm_joint_names = FRANKA_EMIKA_PANDA_MJCF_ARM_JOINT_NAMES.copy()
    args.gripper_joint_names = FRANKA_EMIKA_PANDA_MJCF_GRIPPER_JOINT_NAMES.copy()
    args.arm_actuator_names = FRANKA_EMIKA_PANDA_MJCF_ARM_ACTUATOR_NAMES.copy()
    args.gripper_actuator_names = FRANKA_EMIKA_PANDA_MJCF_GRIPPER_ACTUATOR_NAMES.copy()
    args.arm_default_offset = FRANKA_EMIKA_PANDA_MJCF_ARM_HOME_QPOS.tolist()
    args.full_qpos_default_offset = FRANKA_EMIKA_PANDA_MJCF_FULL_HOME_QPOS.tolist()
    args.gripper_open_pos = 0.04
    args.gripper_close_pos = 0.0
    return args


def resolve_robot_urdf_path(robot_description: str | None, robot_description_attr: str, urdf_path: str) -> Path:
    if not robot_description:
        return resolve_repo_path(urdf_path)

    try:
        description_module = importlib.import_module(f"robot_descriptions.{robot_description}")
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import the requested robot description. "
            "Install 'robot_descriptions' in the Python environment running this script."
        ) from exc

    if not hasattr(description_module, robot_description_attr):
        raise AttributeError(
            f"robot_descriptions.{robot_description} does not define '{robot_description_attr}'."
        )

    description_path = Path(getattr(description_module, robot_description_attr)).resolve()
    if not description_path.exists():
        raise FileNotFoundError(
            f"Resolved robot description path does not exist: {description_path}"
        )
    return description_path


def resolve_robot_model_path(
    mjcf_path: str | None,
    robot_description: str | None,
    robot_description_attr: str,
    urdf_path: str,
) -> tuple[Path, str]:
    if mjcf_path:
        return resolve_repo_path(mjcf_path), "mjcf"
    return resolve_robot_urdf_path(robot_description, robot_description_attr, urdf_path), "urdf"


def name_to_joint_qpos_addr(mujoco, model, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise ValueError(f"Joint '{name}' was not found in the MuJoCo model.")
    return int(model.jnt_qposadr[joint_id])


def name_to_joint_dof_addr(mujoco, model, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise ValueError(f"Joint '{name}' was not found in the MuJoCo model.")
    return int(model.jnt_dofadr[joint_id])


def name_to_actuator_id(mujoco, model, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id == -1:
        raise ValueError(f"Actuator '{name}' was not found in the MuJoCo model.")
    return actuator_id


def name_to_body_id(mujoco, model, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise ValueError(f"Body '{name}' was not found in the MuJoCo model.")
    return body_id


def maybe_name_to_body_id(mujoco, model, name: str) -> int | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        return None
    return body_id


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def rotate_vector_by_quat(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    vec_quat = np.array([0.0, vec_xyz[0], vec_xyz[1], vec_xyz[2]], dtype=np.float32)
    rotated = quat_multiply(quat_multiply(quat_wxyz, vec_quat), quat_conjugate(quat_wxyz))
    return rotated[1:]


def world_to_body_position(root_pos_w: np.ndarray, root_quat_wxyz: np.ndarray, pos_w: np.ndarray) -> np.ndarray:
    return rotate_vector_by_quat(quat_conjugate(root_quat_wxyz), pos_w - root_pos_w)


class TransportCommandState:
    def __init__(
        self,
        waypoint_pose: np.ndarray,
        box_pose: np.ndarray,
        minimal_height: float,
        waypoint_xy_threshold: float,
        waypoint_z_threshold: float,
        waypoint_max_speed: float,
    ):
        self.waypoint_pose = waypoint_pose.astype(np.float32)
        self.box_pose = box_pose.astype(np.float32)
        self.minimal_height = minimal_height
        self.waypoint_xy_threshold = waypoint_xy_threshold
        self.waypoint_z_threshold = waypoint_z_threshold
        self.waypoint_max_speed = waypoint_max_speed
        self.stage_complete = False

    @property
    def command(self) -> np.ndarray:
        return self.box_pose if self.stage_complete else self.waypoint_pose

    @property
    def phase(self) -> np.ndarray:
        return np.array([1.0 if self.stage_complete else 0.0], dtype=np.float32)

    def reset(self):
        self.stage_complete = False

    def maybe_reset_for_transport(self, object_height: float):
        if object_height <= self.minimal_height:
            self.stage_complete = False

    def update(self, object_pos_b: np.ndarray, object_lin_vel_w: np.ndarray):
        if self.stage_complete:
            return
        xy_distance = np.linalg.norm(object_pos_b[:2] - self.waypoint_pose[:2])
        z_distance = abs(object_pos_b[2] - self.waypoint_pose[2])
        object_speed = np.linalg.norm(object_lin_vel_w)
        if (
            object_pos_b[2] > self.minimal_height
            and xy_distance < self.waypoint_xy_threshold
            and z_distance < self.waypoint_z_threshold
            and object_speed < self.waypoint_max_speed
        ):
            self.stage_complete = True


def build_teacher_observation(
    mujoco,
    data,
    robot_root_body_id: int | None,
    object_body_id: int,
    qpos_addrs: list[int],
    qvel_addrs: list[int],
    qpos_default_offset: np.ndarray,
    transport_state: TransportCommandState,
    last_action: np.ndarray,
) -> np.ndarray:
    if robot_root_body_id is None:
        robot_root_pos_w = np.zeros(3, dtype=np.float32)
        robot_root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        robot_root_pos_w = np.asarray(data.xpos[robot_root_body_id], dtype=np.float32)
        robot_root_quat_wxyz = np.asarray(data.xquat[robot_root_body_id], dtype=np.float32)
    object_pos_w = np.asarray(data.xpos[object_body_id], dtype=np.float32)
    object_lin_vel_w = np.asarray(data.cvel[object_body_id][3:], dtype=np.float32)
    object_pos_b = world_to_body_position(robot_root_pos_w, robot_root_quat_wxyz, object_pos_w)
    transport_state.update(object_pos_b, object_lin_vel_w)

    joint_pos = np.asarray(data.qpos[qpos_addrs], dtype=np.float32)
    joint_vel = np.asarray(data.qvel[qvel_addrs], dtype=np.float32)
    joint_pos_rel = joint_pos - qpos_default_offset

    obs_parts = [
        joint_pos_rel,
        joint_vel,
        object_pos_b,
        transport_state.command,
        transport_state.phase,
        last_action,
    ]
    return np.concatenate(obs_parts, axis=0).astype(np.float32)


def sample_reset_object_qpos(
    default_object_qpos: np.ndarray,
    object_reset_x_range: tuple[float, float],
    object_reset_y_range: tuple[float, float],
    object_reset_z_range: tuple[float, float],
) -> np.ndarray:
    randomized_qpos = default_object_qpos.copy()
    randomized_qpos[0] += np.random.uniform(*object_reset_x_range)
    randomized_qpos[1] += np.random.uniform(*object_reset_y_range)
    randomized_qpos[2] += np.random.uniform(*object_reset_z_range)
    return randomized_qpos


@dataclass
class SharedState:
    mode: str = "home"
    request_reset: bool = False
    request_quit: bool = False

    def key_callback(self, keycode: int):
        if keycode in (ord("h"), ord("H")):
            self.mode = "home"
            print("[INFO] Mode -> home")
        elif keycode == ord("1"):
            self.mode = "policy"
            print("[INFO] Mode -> policy")
        elif keycode == ord("2"):
            self.mode = "open"
            print("[INFO] Mode -> open")
        elif keycode == ord("0"):
            self.mode = "hold"
            print("[INFO] Mode -> hold")
        elif keycode in (ord("r"), ord("R")):
            self.request_reset = True
            print("[INFO] Reset requested")
        elif keycode in (ord("q"), ord("Q")):
            self.request_quit = True
            print("[INFO] Quit requested")


class ControlPanel:
    def __init__(self, shared_state: SharedState):
        tk = _import_tkinter()
        self._tk = tk
        self._shared_state = shared_state
        self.root = tk.Tk()
        self.root.title("Franka PickPlace MuJoCo Control")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.mode_var = tk.StringVar(value=f"Mode: {shared_state.mode}")
        self.stage_var = tk.StringVar(value="Stage complete: False")
        self.object_var = tk.StringVar(value="Object z: 0.000")

        tk.Label(self.root, text="Teacher PPO ONNX Control", font=("Arial", 12, "bold")).pack(padx=12, pady=(12, 8))
        tk.Label(self.root, textvariable=self.mode_var).pack(anchor="w", padx=12)
        tk.Label(self.root, textvariable=self.stage_var).pack(anchor="w", padx=12)
        tk.Label(self.root, textvariable=self.object_var).pack(anchor="w", padx=12, pady=(0, 8))

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", padx=12, pady=(0, 12))

        tk.Button(button_frame, text="Home", width=16, command=lambda: self._set_mode("home")).grid(row=0, column=0, padx=4, pady=4)
        tk.Button(button_frame, text="Policy", width=16, command=lambda: self._set_mode("policy")).grid(row=0, column=1, padx=4, pady=4)
        tk.Button(button_frame, text="Open Gripper", width=16, command=lambda: self._set_mode("open")).grid(row=1, column=0, padx=4, pady=4)
        tk.Button(button_frame, text="Hold", width=16, command=lambda: self._set_mode("hold")).grid(row=1, column=1, padx=4, pady=4)
        tk.Button(button_frame, text="Reset Scene", width=16, command=self._request_reset).grid(row=2, column=0, padx=4, pady=4)
        tk.Button(button_frame, text="Quit", width=16, command=self._on_close).grid(row=2, column=1, padx=4, pady=4)

        help_text = "Keys: H=Home, 1=Policy, 2=Open, 0=Hold, R=Reset, Q=Quit"
        tk.Label(self.root, text=help_text, justify="left").pack(anchor="w", padx=12, pady=(0, 12))

    def _set_mode(self, mode: str):
        self._shared_state.mode = mode

    def _request_reset(self):
        self._shared_state.request_reset = True

    def _on_close(self):
        self._shared_state.request_quit = True

    def sync(self, mode: str, stage_complete: bool, object_height: float):
        self.mode_var.set(f"Mode: {mode}")
        self.stage_var.set(f"Stage complete: {stage_complete}")
        self.object_var.set(f"Object z: {object_height:.3f}")

    def update(self):
        self.root.update_idletasks()
        self.root.update()


def main():
    mujoco = _import_mujoco()
    ort = _import_onnxruntime()
    args = apply_robot_preset(parse_args())

    if len(args.arm_joint_names) != 7 or len(args.arm_actuator_names) != 7:
        raise ValueError("Expected exactly 7 Franka arm joints and 7 Franka arm actuators.")
    if len(args.full_qpos_default_offset) != 9:
        raise ValueError("--full_qpos_default_offset must contain 9 values (7 arm + 2 gripper).")
    robot_model_path, robot_input_format = resolve_robot_model_path(
        mjcf_path=args.mjcf_path,
        robot_description=args.robot_description,
        robot_description_attr=args.robot_description_attr,
        urdf_path=args.urdf_path,
    )

    scene_xml_path = resolve_repo_path(args.scene_xml)
    if args.rebuild_scene or not scene_xml_path.exists():
        build_pick_place_mujoco_scene(
            urdf_path=robot_model_path,
            output_xml_path=args.scene_xml,
            timestep=args.timestep,
            input_format=robot_input_format,
            arm_joint_names=args.arm_joint_names,
            gripper_joint_names=args.gripper_joint_names,
        )
        scene_xml_path = resolve_repo_path(args.scene_xml)

    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)

    session = ort.InferenceSession(str(Path(args.policy).resolve()), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    qpos_addrs = [name_to_joint_qpos_addr(mujoco, model, name) for name in args.arm_joint_names + args.gripper_joint_names]
    qvel_addrs = [name_to_joint_dof_addr(mujoco, model, name) for name in args.arm_joint_names + args.gripper_joint_names]
    arm_actuator_ids = [name_to_actuator_id(mujoco, model, name) for name in args.arm_actuator_names]
    gripper_actuator_ids = [name_to_actuator_id(mujoco, model, name) for name in args.gripper_actuator_names]
    robot_root_body_id = maybe_name_to_body_id(mujoco, model, args.robot_root_body)
    object_body_id = name_to_body_id(mujoco, model, args.object_body)
    object_freejoint_qpos_addr = name_to_joint_qpos_addr(mujoco, model, args.object_freejoint)

    if robot_root_body_id is None:
        print(f"[WARN] Body '{args.robot_root_body}' was not found. Using world origin as the robot root frame.")

    arm_default_offset = np.asarray(args.arm_default_offset, dtype=np.float32)
    qpos_default_offset = np.asarray(args.full_qpos_default_offset, dtype=np.float32)
    object_reset_x_range = (float(args.object_reset_x_range[0]), float(args.object_reset_x_range[1]))
    object_reset_y_range = (float(args.object_reset_y_range[0]), float(args.object_reset_y_range[1]))
    object_reset_z_range = (float(args.object_reset_z_range[0]), float(args.object_reset_z_range[1]))
    transport_state = TransportCommandState(
        waypoint_pose=np.asarray(args.waypoint_pose, dtype=np.float32),
        box_pose=np.asarray(args.box_pose, dtype=np.float32),
        minimal_height=float(args.minimal_height),
        waypoint_xy_threshold=float(args.waypoint_xy_threshold),
        waypoint_z_threshold=float(args.waypoint_z_threshold),
        waypoint_max_speed=float(args.waypoint_max_speed),
    )
    shared_state = SharedState(mode="home")
    panel = ControlPanel(shared_state)

    last_action = np.zeros(8, dtype=np.float32)
    last_arm_target = arm_default_offset.copy()
    last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)

    def reset_scene():
        nonlocal last_action, last_arm_target, last_gripper_target
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_addrs] = qpos_default_offset
        data.qvel[:] = 0.0
        randomized_object_qpos = sample_reset_object_qpos(
            default_object_qpos=DEFAULT_OBJECT_QPOS,
            object_reset_x_range=object_reset_x_range,
            object_reset_y_range=object_reset_y_range,
            object_reset_z_range=object_reset_z_range,
        )
        data.qpos[object_freejoint_qpos_addr : object_freejoint_qpos_addr + 7] = randomized_object_qpos
        last_action = np.zeros(8, dtype=np.float32)
        last_arm_target = arm_default_offset.copy()
        last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)
        transport_state.reset()
        shared_state.mode = "home"
        shared_state.request_reset = False
        for actuator_id, target in zip(arm_actuator_ids, last_arm_target, strict=True):
            data.ctrl[actuator_id] = float(target)
        for actuator_id, target in zip(gripper_actuator_ids, last_gripper_target, strict=True):
            data.ctrl[actuator_id] = float(target)
        mujoco.mj_forward(model, data)

    reset_scene()

    control_dt = 1.0 / args.control_rate_hz
    sim_dt = float(model.opt.timestep)
    steps_per_control = max(1, int(round(control_dt / sim_dt)))
    step_counter = 0

    print(f"[INFO] Scene: {scene_xml_path}")
    print(f"[INFO] Robot model ({robot_input_format}): {robot_model_path}")
    print(f"[INFO] ONNX policy: {Path(args.policy).resolve()}")
    print(
        "[INFO] Object reset offset ranges (Isaac-matched): "
        f"x={object_reset_x_range}, y={object_reset_y_range}, z={object_reset_z_range}"
    )
    print("[INFO] Key H -> home")
    print("[INFO] Key 1 -> policy")
    print("[INFO] Key 2 -> open gripper")
    print("[INFO] Key 0 -> hold")
    print("[INFO] Key R -> reset scene")
    print("[INFO] Key Q -> quit")

    with mujoco.viewer.launch_passive(model, data, key_callback=shared_state.key_callback) as viewer:
        while viewer.is_running() and not shared_state.request_quit:
            panel.update()

            if shared_state.request_reset:
                reset_scene()

            if step_counter % steps_per_control == 0:
                object_height = float(data.xpos[object_body_id][2])
                current_arm_qpos = np.asarray(data.qpos[qpos_addrs[:7]], dtype=np.float32)

                if shared_state.mode == "home":
                    last_arm_target = arm_default_offset.copy()
                    last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)
                    last_action = np.zeros(8, dtype=np.float32)
                    if np.max(np.abs(current_arm_qpos - arm_default_offset)) < args.home_tolerance:
                        transport_state.reset()

                elif shared_state.mode == "policy":
                    transport_state.maybe_reset_for_transport(object_height)
                    obs = build_teacher_observation(
                        mujoco=mujoco,
                        data=data,
                        robot_root_body_id=robot_root_body_id,
                        object_body_id=object_body_id,
                        qpos_addrs=qpos_addrs,
                        qvel_addrs=qvel_addrs,
                        qpos_default_offset=qpos_default_offset,
                        transport_state=transport_state,
                        last_action=last_action,
                    )
                    policy_action = session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0][0].astype(np.float32)

                    arm_action = np.clip(policy_action[:7], -1.0, 1.0)
                    last_arm_target = arm_default_offset + args.arm_action_scale * arm_action

                    if args.force_close_gripper_in_policy:
                        last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_close_pos, dtype=np.float32)
                        last_action = np.concatenate([policy_action[:7], np.array([-1.0], dtype=np.float32)])
                    else:
                        gripper_open = policy_action[7] >= 0.0
                        gripper_target = args.gripper_open_pos if gripper_open else args.gripper_close_pos
                        last_gripper_target = np.full(len(gripper_actuator_ids), gripper_target, dtype=np.float32)
                        last_action = policy_action

                elif shared_state.mode == "open":
                    last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)
                    last_action = np.concatenate([last_action[:7], np.array([1.0], dtype=np.float32)])

                elif shared_state.mode == "hold":
                    pass

                for actuator_id, target in zip(arm_actuator_ids, last_arm_target, strict=True):
                    data.ctrl[actuator_id] = float(target)
                for actuator_id, target in zip(gripper_actuator_ids, last_gripper_target, strict=True):
                    data.ctrl[actuator_id] = float(target)

                panel.sync(shared_state.mode, transport_state.stage_complete, object_height)

            mujoco.mj_step(model, data)
            viewer.sync()
            step_counter += 1
            time.sleep(sim_dt)


if __name__ == "__main__":
    main()
