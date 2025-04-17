from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

### WHEEL VELOCITY PENALTY ###
def joint_vel_truncated_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_penalty = 1, max_penalty_domain_upper_bound = 1) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel clamped at max_penalty.
    max_penalty_domain_upper_bound means that any abs(vel) < max_penalty_domain_upper_bound will incur L2 cost (up to max_penalty); i.e. it stretches out the quadratic basin,
    otherwise it would saturate at vel = 1 for max_penalty = 1.

    penalty = clamp( max_penalty * (vel**2 / max_penalty_domain_upper_bound**2)  , max_penalty)

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.clamp(max_penalty * torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]) / (max_penalty_domain_upper_bound**2), max=max_penalty), dim=1)


### WHEEL CONTACT ###
def lost_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force drops to 0 (no contact)."""
    eps = 1e-5
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    should_terminate = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] < eps, dim=1
    )

    # print(f"SHOULD TERMINATE: {should_terminate}")

    return should_terminate

def prolonged_lost_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force drops to 0 (no contact)."""
    eps = 1e-5
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    should_terminate = torch.all(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] < eps, dim=1
    )

    # print(f"SHOULD TERMINATE: {should_terminate}")

    return should_terminate


### COMMANDS ###
def base_angular_velocity_penalty_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_penalty: float) -> torch.Tensor:
    """Penalize tracking of angular velocity commands (yaw) using l1 kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.clamp(ang_vel_error, max=max_penalty)


def base_linear_velocity_penalty_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_penalty: float) -> torch.Tensor:
    """Penalize tracking of linear velocity commands (xy axes) using l1 kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)

    return torch.clamp(lin_vel_error, max=max_penalty)

# Forward progress
def forward_progress(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    fwd_progress = asset.data.root_lin_vel_b[:, 0]
    return fwd_progress

# Observation of commands, mainly for debugging purposes
def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""

    cmd = env.command_manager.get_command(command_name)
    # print(cmd)

    # import pdb; pdb.set_trace()

    return cmd

def angle_penalty(
    env: ManagerBasedRLEnv, max_penalty_angle_rad: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize from deviation from upright.

    Returns 0 penalty if upright, 1 if angle is max_penalty_angle_rad, >1 if angle exceeds max_penalty_angle_rad

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    angle_rad = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs()

    return angle_rad/max_penalty_angle_rad

def projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    gravity = asset.data.projected_gravity_b

    angle_deg = torch.rad2deg(torch.acos(-gravity[:, 2]).abs())

    # import pdb; pdb.set_trace()

    return gravity


# dbg
# def joint_efforts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
#     asset: RigidObject = env.scene[asset_cfg.name]
#     asset.
#     return 0