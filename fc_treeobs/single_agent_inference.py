# Import packages for plotting and system
import getopt
import random
import sys
from collections import deque

# make sure the root path is in system path
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from importlib_resources import path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

import fc_treeobs.nets
from fc_treeobs.dueling_double_dqn import Agent
from fc_treeobs.utils import norm_obs_clip, split_tree_into_feature_groups


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_episodes="])
    except getopt.GetoptError:
        print('single_agent_inference.py -n <n_episodes>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_episodes'):
            n_episodes = int(arg)

    random.seed(1)
    np.random.seed(1)

    # Preload an agent
    training = False

    # Initialize a random map with a random number of agents
    x_dim = np.random.randint(20, 40)
    y_dim = np.random.randint(20, 40)
    n_agents = 1  # np.random.randint(3, 8)
    n_goals = n_agents + np.random.randint(0, 3)
    min_dist = int(0.75 * min(x_dim, y_dim))
    tree_depth = 4

    # Use a the malfunction generator to break agents from time to time
    stochastic_data = {'prop_malfunction': 0.0,  # Percentage of defective agents
                       'malfunction_rate': 0,  # Rate of malfunction occurence
                       'min_duration': 0,  # Minimal duration of malfunction
                       'max_duration': 0  # Max duration of malfunction
                       }

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # Get an observation builder and predictor
    observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=3,
                                                       # Number of cities in map (where train stations are)
                                                       seed=1,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=observation_helper)
    env.reset(True, True)
    env_renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=False,
                              screen_height=1000,  # Adjust these parameters to fit your resolution
                              screen_width=1000)  # Adjust these parameters to fit your resolution

    handle = env.get_agent_handles()
    features_per_node = env.obs_builder.observation_dim
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = features_per_node * nr_nodes
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_episodes' not in locals():
        n_episodes = 1000

    # Set max number of steps per episode as well as other training relevant parameter
    max_steps = int(3 * (env.height + env.width))
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998
    action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()
    # Initialize the agent
    agent = Agent(state_size, action_size)

    # Here you can pre-load an agent
    with path(fc_treeobs.nets, "single_agent_navigation_checkpoint1000.pth") as file_in:
        agent.qnetwork_local.load_state_dict(torch.load(file_in))

    # Do training over n_episodes
    for episodes in range(1, n_episodes + 1):

        # Reset environment
        obs, info = env.reset(True, True)
        env_renderer.reset()

        # Build agent specific observations
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree_into_feature_groups(obs[a], tree_depth)
            data = norm_obs_clip(data)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            agent_obs[a] = obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        # Run episode
        for step in range(max_steps):

            # Action
            for a in range(env.get_num_agents()):
                # action = agent.act(np.array(obs[a]), eps=eps)
                action = agent.act(agent_obs[a], eps=eps)
                action_prob[action] += 1
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)
            env_renderer.render_env(show=True, show_predictions=True, show_observations=False)

            if done['__all__']:
                break


if __name__ == '__main__':
    main(sys.argv[1:])
