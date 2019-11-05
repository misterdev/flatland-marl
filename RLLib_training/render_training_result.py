from RailEnvRLLibWrapper import RailEnvRLLibWrapper
from custom_preprocessors import TreeObsPreprocessor
import gym
import os

from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import PPOTrainer as Trainer
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph as PolicyGraph

from ray.rllib.models import ModelCatalog

import ray
import numpy as np

import gin

from flatland.envs.predictions import DummyPredictorForRailEnv, ShortestPathPredictorForRailEnv
gin.external_configurable(DummyPredictorForRailEnv)
gin.external_configurable(ShortestPathPredictorForRailEnv)

from ray.rllib.utils.seed import seed as set_seed
from flatland.envs.observations import TreeObsForRailEnv

from flatland.utils.rendertools import RenderTool
import time

gin.external_configurable(TreeObsForRailEnv)

ModelCatalog.register_custom_preprocessor("tree_obs_prep", TreeObsPreprocessor)
ray.init()  # object_store_memory=150000000000, redis_max_memory=30000000000)

__file_dirname__ = os.path.dirname(os.path.realpath(__file__))

CHECKPOINT_PATH = os.path.join(__file_dirname__, 'experiment_configs', 'config_example', 'ppo_policy_two_obs_with_predictions_n_agents_4_map_size_20q58l5_f7',
                               'checkpoint_101', 'checkpoint-101')  # To Modify
N_EPISODES = 10
N_STEPS_PER_EPISODE = 50


def render_training_result(config):
    print('Init Env')

    set_seed(config['seed'], config['seed'], config['seed'])

    # Example configuration to generate a random rail
    env_config = {"width": config['map_width'],
                  "height": config['map_height'],
                  "rail_generator": config["rail_generator"],
                  "nr_extra": config["nr_extra"],
                  "number_of_agents": config['n_agents'],
                  "seed": config['seed'],
                  "obs_builder": config['obs_builder'],
                  "min_dist": config['min_dist'],
                  "step_memory": config["step_memory"]}

    # Observation space and action space definitions
    if isinstance(config["obs_builder"], TreeObsForRailEnv):
        obs_space = gym.spaces.Tuple((gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(168,)),) * 2)
        preprocessor = TreeObsPreprocessor

    else:
        raise ValueError("Undefined observation space")

    act_space = gym.spaces.Discrete(5)

    # Dict with the different policies to train
    policy_graphs = {
        "ppo_policy": (PolicyGraph, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id):
        return "ppo_policy"

    # Trainer configuration
    trainer_config = DEFAULT_CONFIG.copy()

    trainer_config['model'] = {"fcnet_hiddens": config['hidden_sizes']}

    trainer_config['multiagent'] = {"policy_graphs": policy_graphs,
                                    "policy_mapping_fn": policy_mapping_fn,
                                    "policies_to_train": list(policy_graphs.keys())}

    trainer_config["num_workers"] = 0
    trainer_config["num_cpus_per_worker"] = 4
    trainer_config["num_gpus"] = 0.2
    trainer_config["num_gpus_per_worker"] = 0.2
    trainer_config["num_cpus_for_driver"] = 1
    trainer_config["num_envs_per_worker"] = 1
    trainer_config['entropy_coeff'] = config['entropy_coeff']
    trainer_config["env_config"] = env_config
    trainer_config["batch_mode"] = "complete_episodes"
    trainer_config['simple_optimizer'] = False
    trainer_config['postprocess_inputs'] = True
    trainer_config['log_level'] = 'WARN'
    trainer_config['num_sgd_iter'] = 10
    trainer_config['clip_param'] = 0.2
    trainer_config['kl_coeff'] = config['kl_coeff']
    trainer_config['lambda'] = config['lambda_gae']

    env = RailEnvRLLibWrapper(env_config)

    trainer = Trainer(env=RailEnvRLLibWrapper, config=trainer_config)

    trainer.restore(CHECKPOINT_PATH)

    policy = trainer.get_policy("ppo_policy")

    preprocessor = preprocessor(obs_space, {"step_memory": config["step_memory"]})
    env_renderer = RenderTool(env, gl="PILSVG")
    for episode in range(N_EPISODES):

        observation = env.reset()
        for i in range(N_STEPS_PER_EPISODE):
            preprocessed_obs = []
            for obs in observation.values():
                preprocessed_obs.append(preprocessor.transform(obs))
            action, _, infos = policy.compute_actions(preprocessed_obs, [])
            logits = infos['behaviour_logits']
            actions = dict()

            # We select the greedy action.
            for j, logit in enumerate(logits):
                actions[j] = np.argmax(logit)

            # In case we prefer to sample an action stochastically according to the policy graph.
            # for j, act in enumerate(action):
                # actions[j] = act

            # Time to see the rendering at one step
            time.sleep(1)

            env_renderer.renderEnv(show=True, frames=True, iEpisode=episode, iStep=i,
                                   action_dict=list(actions.values()))

            observation, _, _, _ = env.step(actions)

    env_renderer.close_window()


@gin.configurable
def run_experiment(name, num_iterations, n_agents, hidden_sizes, save_every,
                   map_width, map_height, policy_folder_name, obs_builder,
                   entropy_coeff, seed, conv_model, rail_generator, nr_extra, kl_coeff, lambda_gae,
                   step_memory, min_dist):

    render_training_result(
        config={"n_agents": n_agents,
                "hidden_sizes": hidden_sizes,  # Array containing the sizes of the network layers
                "save_every": save_every,
                "map_width": map_width,
                "map_height": map_height,
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
                "step_memory": step_memory
                }
    )


if __name__ == '__main__':
    gin.parse_config_file(os.path.join(__file_dirname__, 'experiment_configs', 'config_example', 'config.gin'))  # To Modify
    run_experiment()
