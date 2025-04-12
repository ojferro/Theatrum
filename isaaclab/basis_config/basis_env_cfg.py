
import argparse

import math

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
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
from isaaclab.envs import mdp

# add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
# parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

"""Rest everything follows."""
BASIS_HEIGHT = 4.5

BASIS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/ojfer/Documents/Projects/Robotics/Theatrum/assets/MockBasis.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
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
        pos=(0.0, 0.0, BASIS_HEIGHT), joint_pos={"wheel_joint_left": 0.0, "wheel_joint_right": 0.0}
    ),
    actuators={
        "rl_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel_joint_left"],
            effort_limit=400.0,
            velocity_limit=500.0,
            stiffness=0.0,
            damping=100_000.0,
        ),
        "rr_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel_joint_right"],
            effort_limit=400.0,
            velocity_limit=500.0,
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


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["wheel_joint_left", "wheel_joint_right"], scale=1.0)
    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["wheel_joint_left", "wheel_joint_right"], scale=1.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.base_height_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": BASIS_HEIGHT},
        )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    robot_on_the_ground = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi/3},
    )  

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()

@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass

@configclass
class BasisEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the wheeled quadruped environment."""

    # Scene settings
    scene: BasisSceneCfg = BasisSceneCfg(num_envs=4096, env_spacing=4.0)
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
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.render_interval = self.decimation
