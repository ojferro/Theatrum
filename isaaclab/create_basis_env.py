"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=-1, help="Random seed.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from basis_config import *
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.app import AppLauncher

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import mdp

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import random

verbose = False


def main():
    """Main function."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
        print(f"Using seed {args_cli.seed}")

    # create environment configuration
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Unofficial tutorial
    # env_cfg = BasisEnvCfg()
    # env_cfg.scene.num_envs = args_cli.num_envs
    # # setup RL environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            with torch.inference_mode():
                # sample actions from -1 to 1
                actions = 250 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions
                obs, rew, terminated, truncated, info = env.step(actions)
                if verbose:
                    print(f"Action space shape: {env.action_space.shape}")
                    print(f"Observation space shape: {env.observation_space.shape}")
                    print(f"Obs: {obs}, Reward: {rew}, Term: {terminated}, Trunc: {truncated}")

            # sample random actions
            # joint_vel = torch.randn_like(env.action_manager.action)
            # step the environment
            # obs, rew, terminated, truncated, info = env.step(joint_vel)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()