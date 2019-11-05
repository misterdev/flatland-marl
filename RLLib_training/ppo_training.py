import os
import gin
import gym
import numpy as np
import tempfile
import ray

from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv

# Import PPO trainer: we can replace these imports by any other trainer from RLLib
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, PPOTrainer
#from ray.rllib.agents.ppo.ppo import PPOTrainer  # ray==0.7.6
#from ray.rllib.agents.ppo.ppo import PPOAgent  # ray==0.7.0
# from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph  # Works for version ray==0.7.0
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy  # For ray==0.7.6
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.seed import seed as set_seed

from ray import tune
from ray.tune.logger import UnifiedLogger
from ray.tune.logger import pretty_print

from RLLib_training.RailEnvRLLibWrapper import RailEnvRLLibWrapper
from RLLib_training.custom_preprocessors import TreeObsPreprocessor

gin.external_configurable(TreeObsForRailEnv)
gin.external_configurable(ShortestPathPredictorForRailEnv)

ModelCatalog.register_custom_preprocessor("tree_obs_prep", TreeObsPreprocessor)
ray.init()  # object_store_memory=150000000000, redis_max_memory=30000000000)

__file_dirname__ = os.path.dirname(os.path.realpath(__file__))


def on_episode_start(info):
    episode = info['episode']
    map_width = info['env'].envs[0].width
    map_height = info['env'].envs[0].height
    episode.horizon = 3*(map_width + map_height)


def on_episode_end(info):
    episode = info['episode']

    # Calculation of a custom score metric: sum of all accumulated rewards, divided by the number of agents
    # and the number of the maximum time steps of the episode.
    score = 0
    for k, v in episode._agent_reward_history.items():
        score += np.sum(v)
    score /= (len(episode._agent_reward_history) * episode.horizon)

    # Calculation of the proportion of solved episodes before the maximum time step
    done = 0
    if len(episode._agent_reward_history[0]) <= episode.horizon-5:
        done = 1

    episode.custom_metrics["score"] = score
    episode.custom_metrics["proportion_episode_solved"] = done


def train(config, reporter):
    print('Init Env')
    # Set seed for numpy, random and tf
    #set_seed(config['seed'], config['seed'], config['seed'])

    # Compute state_size given tree_depth
    #num_features_per_node = config['obs_builder'].observation_dim # non sono sicura che il builder sia già stato inizializzato qua 
    num_features_per_node = 11
    tree_depth = 2
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    obs_size = num_features_per_node * nr_nodes

    # Environment parameters
    env_config = {"width": config['map_width'], # Usare argpars
                  "height": config['map_height'],
                  "rail_generator": config["rail_generator"],
                  "nr_extra": config["nr_extra"],
                  "number_of_agents": config['n_agents'],
                  "seed": config['seed'],
                  "obs_builder": config['obs_builder'],
                  #"obs_builder": TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()),
                  "min_dist": config['min_dist'],
                  "step_memory": config["step_memory"]}

    # Observation space and action space definitions

    #if isinstance(config["obs_builder"], TreeObsForRailEnv):
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(obs_size,)),) * 2)
    preprocessor = "tree_obs_prep"
    #else:
    #    raise ValueError("Undefined observation space") # Only TreeObservation implemented for now.

    act_space = gym.spaces.Discrete(5)

    # Dict with the different policies to train. In this case, all trains follow the same policy
    policy_graphs = {
        "ppo_policy": (PPOTFPolicy, obs_space, act_space, {})  # v > 0.7
        #"ppo_policy": (PPOPolicyGraph, obs_space, act_space, {}) # v = 0.6.5
    }

    # Function that maps an agent id to the name of its respective policy.
    def policy_mapping_fn(agent_id):
        return "ppo_policy"

    # Trainer configuration
    trainer_config = DEFAULT_CONFIG.copy()
    trainer_config['model'] = {"fcnet_hiddens": config['hidden_sizes'],
                               "custom_preprocessor": preprocessor,
                                "custom_options": {
                                    "step_memory": config["step_memory"],
                                    "obs_size": obs_size}}

    trainer_config['multiagent'] = {"policy_graphs": policy_graphs,
                                    "policy_mapping_fn": policy_mapping_fn,
                                    "policies_to_train": list(policy_graphs.keys())}

    # Maximum time steps for an episode is set to 3*map_width*map_height
    trainer_config["horizon"] = 3 * (config['map_width'] + config['map_height'])

    # Parameters for calculation parallelization
    trainer_config["num_workers"] = 0
    trainer_config["num_cpus_per_worker"] = 4
    trainer_config["num_gpus"] = 0
    trainer_config["num_gpus_per_worker"] = 0
    trainer_config["num_cpus_for_driver"] = 1
    trainer_config["num_envs_per_worker"] = 1

    # Parameters for PPO training
    trainer_config['entropy_coeff'] = config['entropy_coeff']
    trainer_config["env_config"] = env_config
    trainer_config["batch_mode"] = "complete_episodes"
    trainer_config['simple_optimizer'] = False
    trainer_config['log_level'] = 'WARN'
    trainer_config['num_sgd_iter'] = 10
    trainer_config['clip_param'] = 0.2
    trainer_config['kl_coeff'] = config['kl_coeff']
    trainer_config['lambda'] = config['lambda_gae']
    trainer_config['callbacks'] = {
            "on_episode_start": tune.function(on_episode_start),
            "on_episode_end": tune.function(on_episode_end)
        }

    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix."""
        logdir = config['policy_folder_name'].format(**locals())
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir=config['local_dir'])
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator

    # RailEnvRLLibWrapper is based on flatland RailEnv and adapted to be used with RLLib
    trainer = PPOTrainer(env=RailEnvRLLibWrapper, config=trainer_config, logger_creator=logger)
    # trainer = PPOAgent(env=RailEnvRLLibWrapper, config=trainer_config, logger_creator=logger)

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print(pretty_print(trainer.train()))
        '''
        if i % config['save_every'] == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        '''
        reporter(num_iterations_trained=trainer._iteration)


@gin.configurable
def run_experiment(name,
                   num_iterations,
                   n_agents,
                   hidden_sizes,
                   save_every,
                   map_width,
                   map_height,
                   policy_folder_name,
                   local_dir,
                   obs_builder,
                   entropy_coeff,
                   seed,
                   conv_model,
                   rail_generator,
                   nr_extra,
                   kl_coeff,
                   lambda_gae,
                   step_memory,
                   min_dist):
    tune.run(
        train,
        name=name,
        stop={"num_iterations_trained": num_iterations},
        config={"n_agents": n_agents,
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


if __name__ == '__main__':
    folder_name = 'config_example'  # TODO To Modify
    # gin.parse_config_file(os.path.join(__file_dirname__, 'experiment_configs', folder_name, 'config.gin'))
    dir = os.path.join(__file_dirname__, 'experiment_configs', folder_name)
    run_experiment(local_dir=dir)

