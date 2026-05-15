#!/usr/bin/env python3

"""Run an Isaac Lab teacher ONNX policy in MuJoCo using the robot_descriptions scene XML."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mujoco_scene_utils import (
    ARM_HOME_QPOS,
    BOX_POSE,
    CUBE_START_POS,
    CUBE_START_QUAT,
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_GRIPPER_JOINT_NAMES,
    FULL_HOME_QPOS,
    GRIPPER_OPEN_QPOS,
    resolve_repo_path,
)
from play_teacher_policy_onnx_mujoco_gui import (
    ControlPanel,
    SharedState,
    TransportCommandState,
    _import_mujoco,
    _import_onnxruntime,
    build_teacher_observation,
    maybe_name_to_body_id,
    name_to_actuator_id,
    name_to_body_id,
    name_to_joint_dof_addr,
    name_to_joint_qpos_addr,
    sample_reset_object_qpos,
)
from policy_trace_utils import CsvTraceWriter


# ROBOT_DESCRIPTIONS_SCENE_XML = Path("assets/mujoco/franka_pick_place_scene_robot_descriptions.xml")
ROBOT_DESCRIPTIONS_SCENE_XML = Path("assets/mujoco/franka_panda/franka_description/(cube&box)_(franka_emika_panda)_(pickandplace)/franka_panda_pick_place.xml")

DEFAULT_ARM_ACTUATOR_NAMES = DEFAULT_ARM_JOINT_NAMES
DEFAULT_GRIPPER_ACTUATOR_NAMES = DEFAULT_GRIPPER_JOINT_NAMES
DEFAULT_BOX_POSE = np.array(BOX_POSE, dtype=np.float32)
DEFAULT_WAYPOINT_POSE = np.array([0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_ARM_OFFSET = np.array(ARM_HOME_QPOS, dtype=np.float32)
DEFAULT_FULL_QPOS_OFFSET = np.array(FULL_HOME_QPOS, dtype=np.float32)
DEFAULT_GRIPPER_OPEN = float(GRIPPER_OPEN_QPOS[0])
DEFAULT_GRIPPER_CLOSE = 0.0
DEFAULT_OBJECT_QPOS = np.array(CUBE_START_POS + CUBE_START_QUAT, dtype=np.float32)
DEFAULT_OBJECT_RESET_X_RANGE = (-0.1, 0.1)
DEFAULT_OBJECT_RESET_Y_RANGE = (-0.25, 0.25)
DEFAULT_OBJECT_RESET_Z_RANGE = (0.0, 0.0)


def parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True, help="Path to exported Isaac Lab PPO policy.onnx.")
    parser.add_argument("--scene_xml", type=str, default=str(ROBOT_DESCRIPTIONS_SCENE_XML))
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
        "--fixed_object_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional fixed object XY position in world frame. Overrides random object XY reset.",
    )
    parser.add_argument(
        "--force_close_gripper_in_policy",
        action="store_true",
        help="Override the policy gripper output and keep the gripper closed during policy mode.",
    )
    parser.add_argument("--home_tolerance", type=float, default=0.03)
    parser.add_argument("--trace_csv", type=str, default=None, help="Optional CSV path for policy observation/action traces.")
    parser.add_argument("--trace_every", type=int, default=1, help="Write one CSV row every N control steps in policy mode.")
    return parser.parse_args()


def main():
    mujoco = _import_mujoco()
    ort = _import_onnxruntime()
    args = parse_args()

    if len(args.arm_joint_names) != 7 or len(args.arm_actuator_names) != 7:
        raise ValueError("Expected exactly 7 Franka arm joints and 7 Franka arm actuators.")
    if len(args.full_qpos_default_offset) != 9:
        raise ValueError("--full_qpos_default_offset must contain 9 values (7 arm + 2 gripper).")

    scene_xml_path = resolve_repo_path(args.scene_xml)
    if not scene_xml_path.exists():
        raise FileNotFoundError(f"Scene XML does not exist: {scene_xml_path}")

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
    left_finger_body_id = maybe_name_to_body_id(mujoco, model, "panda_leftfinger")
    right_finger_body_id = maybe_name_to_body_id(mujoco, model, "panda_rightfinger")
    finger_body_ids = {
        body_id for body_id in (left_finger_body_id, right_finger_body_id) if body_id is not None
    }

    if robot_root_body_id is None:
        print(f"[WARN] Body '{args.robot_root_body}' was not found. Using world origin as the robot root frame.")
    if not finger_body_ids:
        print("[WARN] Finger bodies 'panda_leftfinger' and 'panda_rightfinger' were not found. Contact logging is disabled.")

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
    prev_finger_cube_contact = False

    def check_finger_cube_contact() -> None:
        nonlocal prev_finger_cube_contact

        if not finger_body_ids:
            return

        contact_now = False
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])

            if (body1 in finger_body_ids and body2 == object_body_id) or (
                body2 in finger_body_ids and body1 == object_body_id
            ):
                contact_now = True
                break

        if contact_now and not prev_finger_cube_contact:
            print("[CONTACT] finger <-> cube")
        elif prev_finger_cube_contact and not contact_now:
            print("[CONTACT END] finger <-> cube")

        prev_finger_cube_contact = contact_now

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
        if args.fixed_object_xy is not None:
            randomized_object_qpos[0] = float(args.fixed_object_xy[0])
            randomized_object_qpos[1] = float(args.fixed_object_xy[1])
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
    control_step_counter = 0
    trace_writer = CsvTraceWriter(args.trace_csv) if args.trace_csv is not None else None

    print(f"[INFO] Scene: {scene_xml_path}")
    print(f"[INFO] ONNX policy: {Path(args.policy).resolve()}")
    print(
        "[INFO] Object reset offset ranges (Isaac-matched): "
        f"x={object_reset_x_range}, y={object_reset_y_range}, z={object_reset_z_range}"
    )
    if args.fixed_object_xy is not None:
        print(f"[INFO] Fixed object XY override: x={float(args.fixed_object_xy[0])}, y={float(args.fixed_object_xy[1])}")
    if trace_writer is not None:
        print(f"[INFO] Writing MuJoCo trace CSV to: {trace_writer.path}")
    print("[INFO] Key H -> home")
    print("[INFO] Key 1 -> policy")
    print("[INFO] Key 2 -> open gripper")
    print("[INFO] Key 0 -> hold")
    print("[INFO] Key R -> reset scene")
    print("[INFO] Key Q -> quit")

    try:
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
                        arm_action = policy_action[:7]
                        last_arm_target = arm_default_offset + args.arm_action_scale * arm_action

                        if args.force_close_gripper_in_policy:
                            last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_close_pos, dtype=np.float32)
                            last_action = np.concatenate([policy_action[:7], np.array([-1.0], dtype=np.float32)])
                        else:
                            gripper_open = policy_action[7] >= 0.0
                            gripper_target = args.gripper_open_pos if gripper_open else args.gripper_close_pos
                            last_gripper_target = np.full(len(gripper_actuator_ids), gripper_target, dtype=np.float32)
                            last_action = policy_action

                        if trace_writer is not None and control_step_counter % args.trace_every == 0:
                            trace_writer.write_row(
                                source="mujoco",
                                mode=shared_state.mode,
                                step=control_step_counter,
                                sim_time=control_step_counter * control_dt,
                                policy_obs=obs,
                                action=policy_action,
                                arm_target=last_arm_target,
                                gripper_target=last_gripper_target,
                            )
                            trace_writer.flush()

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
                    control_step_counter += 1

                mujoco.mj_step(model, data)
                check_finger_cube_contact()
                viewer.sync()
                step_counter += 1
                time.sleep(sim_dt)
    finally:
        if trace_writer is not None:
            trace_writer.close()


if __name__ == "__main__":
    main()
