# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip

from . import mdp


PEG_RADIUS = 0.015
PEG_HEIGHT = 0.12
PEG_HALF_LENGTH = PEG_HEIGHT * 0.5
TABLE_SURFACE_Z = 0.03
PEG_START_Z = TABLE_SURFACE_Z + PEG_HALF_LENGTH
HOLE_CENTER_X = 0.60
HOLE_CENTER_Y = 0.00
HOLE_TOP_Z = 0.11
SOCKET_WALL_HEIGHT = 0.08
SOCKET_WALL_Z = TABLE_SURFACE_Z + SOCKET_WALL_HEIGHT * 0.5
SOCKET_INNER_HALF_EXTENT = 0.022
SOCKET_WALL_THICKNESS = 0.015
SOCKET_OUTER_SPAN = 2 * (SOCKET_INNER_HALF_EXTENT + SOCKET_WALL_THICKNESS)


@configclass
class FrankaPegInHoleSceneCfg(InteractiveSceneCfg):
    """Scene with a Franka arm, a graspable peg, and a four-wall socket."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, -0.10, PEG_START_Z], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CylinderCfg(
            radius=PEG_RADIUS,
            height=PEG_HEIGHT,
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
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.70, 0.95), metallic=0.1),
        ),
    )

    socket_left_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SocketLeftWall",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[HOLE_CENTER_X - (SOCKET_INNER_HALF_EXTENT + SOCKET_WALL_THICKNESS * 0.5), HOLE_CENTER_Y, SOCKET_WALL_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=[SOCKET_WALL_THICKNESS, SOCKET_OUTER_SPAN, SOCKET_WALL_HEIGHT],
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), metallic=0.3),
        ),
    )

    socket_right_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SocketRightWall",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[HOLE_CENTER_X + (SOCKET_INNER_HALF_EXTENT + SOCKET_WALL_THICKNESS * 0.5), HOLE_CENTER_Y, SOCKET_WALL_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=[SOCKET_WALL_THICKNESS, SOCKET_OUTER_SPAN, SOCKET_WALL_HEIGHT],
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), metallic=0.3),
        ),
    )

    socket_front_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SocketFrontWall",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[HOLE_CENTER_X, HOLE_CENTER_Y + (SOCKET_INNER_HALF_EXTENT + SOCKET_WALL_THICKNESS * 0.5), SOCKET_WALL_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=[SOCKET_OUTER_SPAN, SOCKET_WALL_THICKNESS, SOCKET_WALL_HEIGHT],
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), metallic=0.3),
        ),
    )

    socket_back_wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SocketBackWall",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[HOLE_CENTER_X, HOLE_CENTER_Y - (SOCKET_INNER_HALF_EXTENT + SOCKET_WALL_THICKNESS * 0.5), SOCKET_WALL_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=[SOCKET_OUTER_SPAN, SOCKET_WALL_THICKNESS, SOCKET_WALL_HEIGHT],
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), metallic=0.3),
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.copy().replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.1, 0.1, 0.1))},
            prim_path="/Visuals/FrameTransformer",
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    """Hole target pose used for visual debugging and reward computation."""

    hole_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(HOLE_CENTER_X, HOLE_CENTER_X),
            pos_y=(HOLE_CENTER_Y, HOLE_CENTER_Y),
            pos_z=(HOLE_TOP_Z, HOLE_TOP_Z),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the task."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: ActionTerm = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation terms for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        peg_position = ObsTerm(func=mdp.peg_position_in_robot_root_frame)
        ee_to_peg = ObsTerm(func=mdp.ee_to_peg_vector)
        peg_tip_to_hole = ObsTerm(
            func=mdp.peg_tip_to_hole_vector,
            params={"command_name": "hole_pose", "peg_half_length": PEG_HALF_LENGTH},
        )
        hole_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hole_pose"})
        peg_upright = ObsTerm(func=mdp.peg_upright_projection)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Environment reset configuration."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_peg_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.03), "y": (-0.12, 0.12), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("peg", body_names="Peg"),
        },
    )


@configclass
class RewardsCfg:
    """Stage-shaped rewards for peg insertion."""

    reaching_peg = RewTerm(func=mdp.peg_ee_distance, params={"std": 0.10}, weight=1.5)

    grasping_peg = RewTerm(
        func=mdp.grasp_peg_reward,
        params={"distance_threshold": 0.045},
        weight=6.0,
    )

    lifting_peg = RewTerm(
        func=mdp.peg_is_lifted,
        params={"minimal_height": 0.14},
        weight=8.0,
    )

    upright_peg = RewTerm(func=mdp.peg_upright_reward, weight=2.0)

    align_hole_coarse = RewTerm(
        func=mdp.peg_hole_xy_alignment_reward,
        params={"std": 0.10, "minimal_height": 0.14, "command_name": "hole_pose", "peg_half_length": PEG_HALF_LENGTH},
        weight=10.0,
    )

    align_hole_fine = RewTerm(
        func=mdp.peg_hole_xy_alignment_reward,
        params={"std": 0.03, "minimal_height": 0.14, "command_name": "hole_pose", "peg_half_length": PEG_HALF_LENGTH},
        weight=12.0,
    )

    pre_insert = RewTerm(
        func=mdp.peg_pre_insertion_reward,
        params={
            "xy_threshold": 0.025,
            "target_height_offset": 0.015,
            "command_name": "hole_pose",
            "peg_half_length": PEG_HALF_LENGTH,
        },
        weight=14.0,
    )

    insertion = RewTerm(
        func=mdp.peg_insertion_reward,
        params={
            "xy_threshold": 0.020,
            "desired_depth": 0.060,
            "command_name": "hole_pose",
            "peg_half_length": PEG_HALF_LENGTH,
        },
        weight=24.0,
    )

    success_bonus = RewTerm(
        func=mdp.peg_insertion_success_reward,
        params={
            "xy_threshold": 0.018,
            "success_depth": 0.055,
            "command_name": "hole_pose",
            "peg_half_length": PEG_HALF_LENGTH,
        },
        weight=40.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    peg_fell = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.02, "asset_cfg": SceneEntityCfg("peg")},
    )

    peg_inserted = DoneTerm(
        func=mdp.peg_inserted,
        params={
            "xy_threshold": 0.018,
            "success_depth": 0.055,
            "command_name": "hole_pose",
            "peg_half_length": PEG_HALF_LENGTH,
        },
    )


@configclass
class CurriculumCfg:
    """Simple curriculum to increase precision-related rewards over time."""

    pre_insert_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "pre_insert", "weight": 20.0, "num_steps": 15000},
    )

    insertion_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "insertion", "weight": 35.0, "num_steps": 25000},
    )

    success_bonus_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "success_bonus", "weight": 60.0, "num_steps": 25000},
    )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 20000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 20000},
    )


@configclass
class FrankaPegInHoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka peg-in-hole environment."""

    scene: FrankaPegInHoleSceneCfg = FrankaPegInHoleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 6.0
        self.sim.dt = 0.01
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class FrankaPegInHoleEnvCfg_PLAY(FrankaPegInHoleEnvCfg):
    """Smaller, deterministic play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
