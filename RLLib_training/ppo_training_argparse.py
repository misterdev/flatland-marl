import os
import gym
import numpy as np
import tempfile
import ray
import argparse

from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv

from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.seed import seed as set_seed

from ray import tune
from ray.tune.logger import UnifiedLogger
from ray.tune.logger import pretty_print

from RLLib_training.RailEnvRLLibWrapper import RailEnvRLLibWrapper
from RLLib_training.custom_preprocessors import TreeObsPreprocessor

ModelCatalog.register_custom_preprocessor("tree_obs_prep", TreeObsPreprocessor)

ray.init()

__file_dirname__ = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':
    folder_name = 'config_example'  # TODO To Modify
    dir = os.path.join(__file_dirname__, 'experiment_configs', folder_name)
    
    # env_width and height change with  training curriculum to better generalize, also other params relative to env
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--experiment_name', type=str, default='experiment_example', help='Give a name to the experiment to store results')
    # Flatland parameters
    parser.add_argument('--obs_builder', type=str, default='TreeObsForRailEnv', help='Type of observation to use')
    parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help="Predictor class to use with observations")
    # Network parameters
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden size for the linear layer in the NN")

    # General training parameters
    parser.add_argument('--n_episodes', type=int, default=6000, help="Number of episodes on which to train the agents")
    # PPO parameters
    parser.add_argument('--entropy_coeff', type=float, default=0.001, help='')
    parser.add_argument('--kl_coeff', type=float, default=0.2, help='')
    parser.add_argument('--lambda_gae', type=float, default=0.9, help='')
    # parser.add_argument('--step_memory', type=float, default=1, help='Number of steps to use as state when training') 
    parser.add_argument('--min_dist', type=int, default=10, help='')

    # Setup
    args = parser.parse_args()
    
    # Compute max_steps in function of env.width and height TODO
    # TODO Vedi come interfacciare tune con la train
    tune.run(
        train,
        name=args.experiment_name,
        stop={"num_iterations_trained": num_iterations},
        config={"n_agents": ,
                "hidden_sizes": hidden_sizes,  # Array containing the sizes of the network layers
                "save_every": save_every,
                "map_width": map_width,
                "map_height": map_height,
                "local_dir": local_dir,
                'policy_folder_name': policy_folder_name,
                "obs_builder": obs_builder,
                "entropy_coeff": entropy_coeff,
                "seed": seed,
                "conv_model": conv_model,
                "rail_generator": rail_generator,
                "nr_extra": nr_extra,
                "kl_coeff": kl_coeff,
                "lambda_gae": lambda_gae,
                "min_dist": min_dist,
                "step_memory": step_memory  # If equal to two, the current observation plus
                # the observation of last time step will be given as input the the model.
                },
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },
        verbose=2,
        local_dir=local_dir
    )
