import random
from collections import deque

import numpy as np
import torch
from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from importlib_resources import path

import torch_training.Nets
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import norm_obs_clip, split_tree

random.seed(1)
np.random.seed(1)

# Parameters for the Environment
x_dim = 10
y_dim = 10
n_agents = 1
n_goals = 5
min_dist = 5

# Load the Environment
env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                    max_dist=99999,
                                                    seed=0),
              obs_builder_object=TreeObsForRailEnv(max_depth=2),
              number_of_agents=n_agents)
env.reset(True, True)

env_renderer = RenderTool(env, gl="PILSVG", )
# Given the depth of the tree observation and the number of features per node we get the following state_size
num_features_per_node = env.obs_builder.observation_dim
tree_depth = 2
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes

# The action space of flatland is 5 discrete actions
action_size = 5

n_trials = 100
max_steps = int(3 * (env.height + env.width))

# And some variables to keep track of the progress
action_dict = dict()
final_action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
time_obs = deque(maxlen=2)
scores = []
dones_list = []
action_prob = [0] * action_size
agent_obs = [None] * env.get_num_agents()
agent_next_obs = [None] * env.get_num_agents()

# Now we load a Double dueling DQN agent
agent = Agent(state_size, action_size, "FC", 0)
with path(torch_training.Nets, "navigator_checkpoint6000.pth") as file_in:
    agent.qnetwork_local.load_state_dict(torch.load(file_in))

    for trials in range(1, n_trials + 1):

        # Reset environment
        obs = env.reset(True, True)
        env_renderer.set_new_rail()

        # Split the observation tree into its parts and normalize the observation using the utility functions.
        # Build agent specific local observation
        for a in range(env.get_num_agents()):
            rail_data, distance_data, agent_data = split_tree(tree=np.array(obs[a]),
                                                              num_features_per_node=num_features_per_node,
                                                              current_depth=0)
            rail_data = norm_obs_clip(rail_data)
            distance_data = norm_obs_clip(distance_data)
            agent_data = np.clip(agent_data, -1, 1)
            agent_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))

        # Reset score and done
        score = 0
        env_done = 0

        # Run episode
        for step in range(max_steps):

            env_renderer.renderEnv(show=True, show_observations=False)

            # Chose the actions
            for a in range(env.get_num_agents()):
                eps = 0
                action = agent.act(agent_obs[a], eps=eps)
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            for a in range(env.get_num_agents()):
                rail_data, distance_data, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                                  num_features_per_node=num_features_per_node,
                                                                  current_depth=0)
                rail_data = norm_obs_clip(rail_data)
                distance_data = norm_obs_clip(distance_data)
                agent_data = np.clip(agent_data, -1, 1)
                agent_next_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))

            agent_obs = agent_next_obs.copy()
            if done['__all__']:
                env_done = 1
                break
