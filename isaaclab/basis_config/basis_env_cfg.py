
import argparse

import math

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.app import AppLauncher

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import mdp

import basis_config.custom_metrics as custom_metrics

"""Rest everything follows."""
BASIS_HEIGHT = 0.33 # m
WHEEL_RADIUS = 0.03 # m
# DESIRED_LINEAR_VEL = 0.5 # m/s
MAX_WHEEL_SPEED = 500.0 # in scale units
USE_CURRICULUM = True

BASIS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/ojfer/Documents/Projects/Robotics/Theatrum/assets/BasisMK2.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # max_depenetration_velocity=100.0,
            # enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BASIS_HEIGHT), joint_pos={"l_wheel_joint": 0.0, "r_wheel_joint": 0.0}
    ),
    actuators={
        "l_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["l_wheel_joint"],
            effort_limit=400.0,
            velocity_limit=1000.0,
            stiffness=0.0,
            damping=100_000.0,
        ),
        "r_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["r_wheel_joint"],
            effort_limit=400.0,
            velocity_limit=1000.0,
            stiffness=0.0,
            damping=100_000.0,
        ),
    },
)

@configclass
class BasisSceneCfg(InteractiveSceneCfg):
    """Configuration for the Basis scene."""

    
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation. Must be called 'robot' for leveraging Managed RL Scenes.
    robot: ArticulationCfg = BASIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Note: cannot rename without changing other "contact_forces" references.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


# @configclass
# class ContactSensorSceneCfg(InteractiveSceneCfg):


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # joint_torques = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["l_wheel_joint", "r_wheel_joint"], scale=10.0)

    # Max_vel (i.e. scale * 1.0) divided by 60 is the RPM of the wheel.
    # Wheel radius = 0.03m ==> Wheel circumference = 2*pi*0.03 = 0.188496m
    # Want: max linear velocity = 3m/s ==> 3m / 0.188496m = 15.915 circumferences per 3 meters
    # Need 15.915 revolutions per second to get 3m/s 
    # scale = 15.915 * 60 / (1/6) = 15.915 * 360 = 5729

    # scale = 360.0 * DESIRED_LINEAR_VEL / (2*math.pi*WHEEL_RADIUS)
    # print(f"Lin vel scale: {scale}")
    joint_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["l_wheel_joint", "r_wheel_joint"], scale=MAX_WHEEL_SPEED)
    # joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["l_wheel_joint", "wheel_joint_right"], scale=1.0)

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    # null = mdp.NullCommandCfg()

    # Cannot rename without changing other instances of "base_velocity" (e.g. in custom_metrics)
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0,4.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(0,0),
            ang_vel_z=(0,0), #(-1.0, 1.0),
            heading=(-math.pi/2, math.pi/2)
        ),
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # on startup
    add_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["robot_base_footprint"]),
            "mass_distribution_params": (-0.2,0.2),
            "operation": "add",
        },
    )
    
    # reset
    set_l_wheel_vel = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_joint"]),
            "position_range": (0,0),
            "velocity_range": (-10,10),
        },
    )

    set_r_wheel_vel = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["r_wheel_joint"]),
            "position_range": (0,0),
            "velocity_range": (-10,10),
        },
    )

    set_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"pitch": (-0.15,0.15)},
            # "pose_range": {"roll": (-0.10,0.10)},
            "velocity_range": {"pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Random perturbations
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.0,2.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "pitch": (-0.5,0.5),"yaw": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("robot")
            },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": BASIS_HEIGHT},
        )
    # (4) Low velocity regularization
    wheel_joint_left_vel = RewTerm(
        func=custom_metrics.joint_vel_truncated_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_joint"]),
            "max_penalty": 10,
            "max_penalty_domain_upper_bound": MAX_WHEEL_SPEED/3},
    )
    wheel_joint_right_vel = RewTerm(
        func=custom_metrics.joint_vel_truncated_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["r_wheel_joint"]),
            "max_penalty": 10,
            "max_penalty_domain_upper_bound": MAX_WHEEL_SPEED/3},
    )
    # (5) Tracking velocity commands
    # These have very low weights until ~ 3000 steps in, see curriculum below
    base_angular_velocity = RewTerm(
        func=custom_metrics.base_angular_velocity_reward,
        weight=0.01,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewTerm(
        func=custom_metrics.base_linear_velocity_reward,
        weight=0.01,
        params={"std": 1.0, "ramp_rate": 2.0, "ramp_at_vel": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )

    # (6) Penalize for wheels not touching the ground
    wheels_on_ground = RewTerm(
        func = custom_metrics.lost_contact,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*wheel_link"),
        }
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Decrease importance of base height after 3000 steps
    base_height_cur = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "base_height", "weight": -1e-3, "num_steps": 3_000}
    )

    # Start rewarding for following commands once it's able to balance
    base_angular_velocity_cur = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "base_angular_velocity", "weight": 10, "num_steps": 3_000}
    )

    base_linear_velocity_cur = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "base_linear_velocity", "weight": 10, "num_steps": 3_000}
    )

    # Regularize wheel vel less heavily once it's able to balance, so that it can follow commands
    wheel_joint_left_vel_cur = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "wheel_joint_left_vel", "weight": -0.1, "num_steps": 2_000}
    )
    wheel_joint_right_vel_cur = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "wheel_joint_right_vel", "weight": -0.1, "num_steps": 2_000}
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    if not USE_CURRICULUM:
        pass

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    robot_within_inclination_bounds = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi/3},
    )  
    # (3) Wheels lost contact. Important at the beginning, less important later on.
    robot_on_ground = DoneTerm(
        func=custom_metrics.prolonged_lost_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*wheel_link"),
        }
    )

@configclass
class BasisEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the wheeled quadruped environment."""

    # Scene settings
    scene: BasisSceneCfg = BasisSceneCfg(num_envs=4096, env_spacing=0.8)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (0.0, 8.0, 5.0)
        # simulation settings
        self.sim.render_interval = self.decimation
