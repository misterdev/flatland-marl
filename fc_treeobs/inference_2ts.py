import sys
import random
from collections import deque
import time

import numpy as np
import torch
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

# make sure the root path is in system path
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from importlib_resources import path

import fc_treeobs.nets
from fc_treeobs.dueling_double_dqn import Agent
from fc_treeobs.utils import normalize_observation


def main(argv):

    random.seed(1)
    np.random.seed(1)

    # Initialize a random map with a random number of agents
    x_dim = np.random.randint(20, 40)
    y_dim = np.random.randint(20, 40)
    n_agents = np.random.randint(3, 4)
    n_goals = n_agents + np.random.randint(0, 3)
    min_dist = int(0.75 * min(x_dim, y_dim))
    tree_depth = 4

    # Get an observation builder and predictor
    predictor = ShortestPathPredictorForRailEnv()
    observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=predictor)

    # Use a the malfunction generator to break agents from time to time
    stochastic_data = {'prop_malfunction': 0.0,  # Percentage of defective agents
                       'malfunction_rate': 0,  # Rate of malfunction occurrence
                       'min_duration': 3,  # Minimal duration of malfunction
                       'max_duration': 20  # Max duration of malfunction
                       }

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

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

    # Initiate the renderer
    env_renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=False,
                              screen_height=1000,  # Adjust these parameters to fit your resolution
                              screen_width=1000)  # Adjust these parameters to fit your resolution
    handle = env.get_agent_handles()
    num_features_per_node = env.obs_builder.observation_dim

    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = 2 * num_features_per_node * nr_nodes
    action_size = 5

    n_trials = 10
    observation_radius = 10
    max_steps = int(3 * (env.height + env.width))
    action_dict = dict()
    time_obs = deque(maxlen=2)
    agent_obs = [None] * env.get_num_agents()

    # Init and load agent
    agent = Agent(state_size, action_size)
    with path(fc_treeobs.nets, "multi_agent_2ts_checkpoint200.pth") as file_in:
        agent.qnetwork_local.load_state_dict(torch.load(file_in))

    # Vars used to record agent performance
    record_images = False
    frame_step = 0

    for trials in range(1, n_trials + 1):
        # Reset environment
        obs, info = env.reset(True, True)
        env_renderer.reset()

        # Build first two-time step observation
        for a in range(env.get_num_agents()):
            obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
        # Accumulate two time steps of observation (Here just twice the first state)
        for i in range(2):
            time_obs.append(obs)
        # Build the agent specific double ti
        for a in range(env.get_num_agents()):
            agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))

        # Run episode
        for step in range(max_steps):
            time.sleep(0.01)

            env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

            if record_images:
                env_renderer.gl.save_image("./Images/Avoiding/flatland_frame_{:04d}.bmp".format(frame_step))
                frame_step += 1

            # Perform action for each agent
            for a in range(env.get_num_agents()):
                action = agent.act(agent_obs[a], eps=0)
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            # Collect observation after environment step
            for a in range(env.get_num_agents()):
                next_obs[a] = normalize_observation(next_obs[a], tree_depth, observation_radius=10)
            # Add new obs to the obs vector
            # Since time_obs is a deque of max_len = 2, an append on the right side when the deque is full
            # provokes a pop of the element from the left side
            time_obs.append(next_obs)
            # Create obs using obs at time step t-1 and ob at time step t
            for a in range(env.get_num_agents()):
                agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))

            if done['__all__']:
                break


if __name__ == '__main__':
    main(sys.argv[1:])
