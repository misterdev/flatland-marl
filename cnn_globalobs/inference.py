import random
from collections import deque

import numpy as np
import torch
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from importlib_resources import path

import cnn_globalobs.nets
from cnn_globalobs.dueling_double_dqn import DQNAgent
from cnn_globalobs.utils import preprocess_obs
from cnn_globalobs.global_observations import CustomGlobalObsForRailEnv


random.seed(1)
np.random.seed(1)
# Parameters for the environment
x_dim = 40
y_dim = 40
n_agents = 4


# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.05,  # Percentage of defective agents
                   'malfunction_rate': 50,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }


# Different agent types (trains) with different speeds.
speed_ratio_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

observation_helper = CustomGlobalObsForRailEnv()

env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=sparse_rail_generator(max_num_cities=3,
                                                   # Number of cities in map (where train stations are)
                                                   seed=1,  # Random seed
                                                   grid_mode=False,
                                                   max_rails_between_cities=2,
                                                   max_rails_in_city=3),
              schedule_generator=sparse_schedule_generator(speed_ratio_map),
              number_of_agents=n_agents,
              stochastic_data=stochastic_data,  # Malfunction data generator
              obs_builder_object=observation_helper)
env.reset(True, True)

env_renderer = RenderTool(env, gl="PILSVG", )

action_size = 5

if 'n_trials' not in locals():
    n_trials = 60000
    
max_steps = int(3 * (env.height + env.width))
eps = 1.
eps_end = 0.005
eps_decay = 0.9995
action_dict = dict()
final_action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
scores = []
dones_list = []
action_prob = [0] * action_size
agent_obs = [None] * env.get_num_agents()
agent_next_obs = [None] * env.get_num_agents()

agent = DQNAgent(action_size, double_dqn=True)

with path(cnn_globalobs.nets, "avoider_checkpoint1000.pth") as file_in:
    agent.qnetwork_local.load_state_dict(torch.load(file_in))

record_images = False
frame_step = 0

for trials in range(1, n_trials + 1):

    # Reset environment
    obs, info = env.reset(True, True)
    env_renderer.reset()
    # Build agent specific observations
    for a in range(env.get_num_agents()):
        agent_obs[a] = preprocess_obs(obs[a], input_channels=9, max_env_width=40, max_env_height=40)
    # Reset score and done
    score = 0
    env_done = 0

    # Run episode
    for step in range(max_steps):
        # Action
        for a in range(env.get_num_agents()):
            action = agent.act(agent_obs[a], eps=0.)
            action_prob[action] += 1
            action_dict.update({a: action})
        # Environment step
        obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=True, show_predictions=False, show_observations=False)
        # Build agent specific observations and normalize
        for a in range(env.get_num_agents()):
            agent_obs[a] = preprocess_obs(obs[a], input_channels=9, max_env_width=40, max_env_height=40)

        if done['__all__']:
            break

