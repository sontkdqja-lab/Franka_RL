import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas.schemas_cfg import (
    ConvexDecompositionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from . import mdp


OBJECT_RADIUS = 0.02
OBJECT_HEIGHT = 0.06
OBJECT_START_Z = OBJECT_HEIGHT * 0.5
ROBOT_START_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
}
GRIPPER_GRASP_FINGER_POS = 0.012
OBJECT_GRASP_OFFSET_B = (0.0, 0.0, 0.025)
OBJECT_GRASP_ROT_EULER_DEG = (0.0, 0.0, 0.0)
LIFT_MINIMAL_HEIGHT = 0.04

TARGET_PLATE_CENTER_X = 0.58
TARGET_PLATE_CENTER_Y = -0.36
TARGET_PLATE_SIZE = 0.14
TARGET_PLATE_THICKNESS = 0.01
TARGET_PLATE_CENTER_Z = TARGET_PLATE_THICKNESS * 0.5
TARGET_HOLE_RADIUS = OBJECT_RADIUS + 0.002
TARGET_OBJECT_REST_Z = OBJECT_HEIGHT * 0.5

TARGET_BOX_CENTER_X = TARGET_PLATE_CENTER_X
TARGET_BOX_CENTER_Y = TARGET_PLATE_CENTER_Y
TARGET_BOX_INNER_SIZE = TARGET_PLATE_SIZE
TARGET_BOX_FLOOR_THICKNESS = TARGET_PLATE_THICKNESS
TARGET_BOX_FLOOR_CENTER_Z = TARGET_PLATE_CENTER_Z
TARGET_BOX_OUTER_SPAN = TARGET_PLATE_SIZE
TARGET_BOX_PLACE_Z = TARGET_OBJECT_REST_Z


def _write_square_plate_with_hole_mesh(
    file_path: Path,
    plate_size: float,
    hole_radius: float,
    thickness: float,
    inner_sides: int = 128,
) -> None:
    """Create a square plate mesh with a circular through-hole as a Wavefront OBJ."""
    outer_sides = 4
    if inner_sides % outer_sides != 0:
        raise ValueError("inner_sides must be divisible by outer_sides for sector triangulation.")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    top_z = thickness * 0.5
    bottom_z = -top_z
    sector_ratio = inner_sides // outer_sides
    half_extent = plate_size * 0.5
    outer_xy = [
        (-half_extent, -half_extent),
        (half_extent, -half_extent),
        (half_extent, half_extent),
        (-half_extent, half_extent),
    ]

    vertices: list[tuple[float, float, float]] = []
    outer_top: list[int] = []
    outer_bottom: list[int] = []
    inner_top: list[int] = []
    inner_bottom: list[int] = []

    def add_vertex(x: float, y: float, z: float) -> int:
        vertices.append((x, y, z))
        return len(vertices) - 1

    for x, y in outer_xy:
        outer_top.append(add_vertex(x, y, top_z))
        outer_bottom.append(add_vertex(x, y, bottom_z))

    for i in range(inner_sides):
        angle = 2.0 * math.pi * i / inner_sides
        x = hole_radius * math.cos(angle)
        y = hole_radius * math.sin(angle)
        inner_top.append(add_vertex(x, y, top_z))
        inner_bottom.append(add_vertex(x, y, bottom_z))

    faces: list[tuple[int, int, int]] = []

    def add_triangle(a: int, b: int, c: int) -> None:
        faces.append((a, b, c))

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        add_triangle(a, b, c)
        add_triangle(a, c, d)

    for i in range(outer_sides):
        i_next = (i + 1) % outer_sides
        add_quad(outer_bottom[i], outer_bottom[i_next], outer_top[i_next], outer_top[i])

    for i in range(inner_sides):
        i_next = (i + 1) % inner_sides
        add_quad(inner_bottom[i], inner_top[i], inner_top[i_next], inner_bottom[i_next])

    for i in range(outer_sides):
        i_next = (i + 1) % outer_sides
        inner_start = i * sector_ratio
        inner_end = inner_start + sector_ratio

        top_sector = [outer_top[i], outer_top[i_next]]
        top_sector.extend(inner_top[j % inner_sides] for j in range(inner_end, inner_start - 1, -1))
        for j in range(1, len(top_sector) - 1):
            add_triangle(top_sector[0], top_sector[j], top_sector[j + 1])

        bottom_sector = [outer_bottom[i]]
        bottom_sector.extend(inner_bottom[j % inner_sides] for j in range(inner_start, inner_end + 1))
        bottom_sector.append(outer_bottom[i_next])
        for j in range(1, len(bottom_sector) - 1):
            add_triangle(bottom_sector[0], bottom_sector[j], bottom_sector[j + 1])

    with file_path.open("w", encoding="ascii") as f:
        f.write("# Auto-generated square plate with circular hole mesh for Frankapeginhole_new\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _get_target_plate_usd_path() -> str:
    asset_root = Path(__file__).resolve().parent / "assets" / "generated"
    obj_path = asset_root / "target_plate_with_hole.obj"
    usd_dir = asset_root / "usd"

    _write_square_plate_with_hole_mesh(
        obj_path,
        plate_size=TARGET_PLATE_SIZE,
        hole_radius=TARGET_HOLE_RADIUS,
        thickness=TARGET_PLATE_THICKNESS,
    )

    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(obj_path),
            usd_dir=str(usd_dir),
            usd_file_name="target_plate_with_hole.usd",
            make_instanceable=True,
            force_usd_conversion=True,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mesh_collision_props=ConvexDecompositionPropertiesCfg(),
        )
    )
    return converter.usd_path


TARGET_PLATE_USD_PATH = _get_target_plate_usd_path()


def make_main_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/main_camera",
        update_period=4,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
            semantic_tags=[("class", "main_camera")],
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.04),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )


def make_sub_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/SubCamera",
        update_period=4,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
            semantic_tags=[("class", "sub_camera")],
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.90, -0.65, 0.55),
            rot=(0.45568907, -0.28168112, 0.15462769, 0.83011655),
            convention="world",
        ),
    )


##
# Scene definition
##


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # robots
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.joint_pos.update(ROBOT_START_JOINT_POS)
    robot.init_state.joint_pos["panda_finger_joint.*"] = GRIPPER_GRASP_FINGER_POS

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # objects
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0, OBJECT_START_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=OBJECT_RADIUS,
            height=OBJECT_HEIGHT,
            axis="Z",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.11, 0.68, 0.46), metallic=0.2),
        ),
    )

    target_box_floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPlate",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[TARGET_BOX_CENTER_X, TARGET_BOX_CENTER_Y, TARGET_BOX_FLOOR_CENTER_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=TARGET_PLATE_USD_PATH,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.70, 0.62), metallic=0.05),
        ),
    )

    main_camera: CameraCfg | None = make_main_camera_cfg()

    sub_camera: CameraCfg | None = make_sub_camera_cfg()

    # Listens to the required transforms, adding visualization markers to end-effector frame
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  # root of robot (base_link)
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.copy().replace(
            markers={
                "frame": FRAME_MARKER_CFG.markers["frame"].replace(
                    scale=(0.1, 0.1, 0.1)
                )
            },
            prim_path="/Visuals/FrameTransformer",
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # end-effector link
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    drop_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.50, 0.50),
            pos_y=(0.00, 0.00),
            pos_z=(0.30, 0.30),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    box_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(TARGET_BOX_CENTER_X, TARGET_BOX_CENTER_X),
            pos_y=(TARGET_BOX_CENTER_Y, TARGET_BOX_CENTER_Y),
            pos_z=(TARGET_BOX_PLACE_Z, TARGET_BOX_PLACE_Z),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    transport_target = mdp.StagedTransportPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        object_name="object",
        waypoint_command_name="drop_pose",
        drop_command_name="box_pose",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        hold_duration_s=0.0,
        minimal_height=LIFT_MINIMAL_HEIGHT,
        waypoint_xy_threshold=0.03,
        waypoint_z_threshold=0.03,
        waypoint_max_speed=0.05,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_normal = ObsTerm(func=mdp.object_normal_in_robot_root_frame)

        place_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "transport_target"})
        transport_phase = ObsTerm(
            func=mdp.staged_command_phase, params={"command_name": "transport_target"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material_object = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_grasped = EventTerm(
        func=mdp.reset_cylinder_in_gripper,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "hand_body_name": "panda_hand",
            "left_finger_body_name": "panda_leftfinger",
            "right_finger_body_name": "panda_rightfinger",
            "gripper_joint_expr": "panda_finger_joint.*",
            "gripper_joint_pos": GRIPPER_GRASP_FINGER_POS,
            "object_offset_b": OBJECT_GRASP_OFFSET_B,
            "object_rot_euler_xyz_deg": OBJECT_GRASP_ROT_EULER_DEG,
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    grasping_object = RewTerm(
        func=mdp.grasp_reward,
        params={"distance_threshold": 0.04},
        weight=5.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": LIFT_MINIMAL_HEIGHT}, weight=10.0
    )

    object_goal_tracking_coarse = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.5, "minimal_height": LIFT_MINIMAL_HEIGHT, "command_name": "transport_target"},
        weight=8.0,
    )

    object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": LIFT_MINIMAL_HEIGHT, "command_name": "transport_target"},
        weight=8.0,
    )

    target_stability = RewTerm(
        func=mdp.target_stability_reward,
        params={
            "command_name": "transport_target",
            "proximity_std": 0.04,
            "joint_vel_std": 0.6,
            "action_rate_std": 0.08,
        },
        weight=6.0,
    )

    waypoint_hold_reward = RewTerm(
        func=mdp.waypoint_hold_bonus,
        params={"command_name": "transport_target"},
        weight=12.0,
    )

    premature_box_penalty = RewTerm(
        func=mdp.premature_goal_penalty,
        params={
            "command_name": "box_pose",
            "staged_command_name": "transport_target",
            "std": 0.06,
            "minimal_height": LIFT_MINIMAL_HEIGHT,
        },
        weight=-8.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_reaching_goal = DoneTerm(
        func=mdp.object_placed_success,
        params={
            "command_name": "transport_target",
            "xy_threshold": 0.02,
            "z_threshold": 0.02,
            "linear_vel_threshold": 0.05,
            "angular_vel_threshold": 0.10,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-2, "num_steps": 20000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -5e-2, "num_steps": 20000},
    )


##
# Environment configuration
##
@configclass
class FrankaPegInHoleNewEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka pick-and-place environment."""

    # Scene settings
    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    use_cameras: bool = False

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = (
            8.0  # Allow pick, 2-second waypoint hold, and final placement.
        )
        # simulation settings
        self.sim.dt = 0.01
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        if not self.use_cameras:
            self.scene.main_camera = None
            self.scene.sub_camera = None


@configclass
class FrankaPegInHoleNewEnvCfg_PLAY(FrankaPegInHoleNewEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
