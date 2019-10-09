# Import packages for plotting and system
import getopt
import random
import sys
from collections import deque

# make sure the root path is in system path
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch
from importlib_resources import path

import torch_training.Nets
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from flatland.envs.schedule_generators import sparse_schedule_generator
from torch_training.dueling_double_dqn import DQNAgent
from torch_training.utils import preprocess_obs
# from utils.observation_utils import normalize_observation


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_episodes="])
    except getopt.GetoptError:
        print('training.py -n <n_episodes>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_episodes'):
            n_episodes = int(arg)

    ## Initialize the random
    random.seed(1)
    np.random.seed(1)

    # Initialize a random map with a random number of agents

    # Parameters for the environment
    x_dim = 20
    y_dim = 20
    n_agents = 3

    # Use the malfunction generator to break agents from time to time
    stochastic_data = {'prop_malfunction': 0.1,  # Percentage of defective agents
                       'malfunction_rate': 30,  # Rate of malfunction occurrence
                       'min_duration': 3,  # Minimal duration of malfunction
                       'max_duration': 20  # Max duration of malfunction
                       }

    # Custom observation builder
    observation_helper = GlobalObsForRailEnv()

    # Different agent types (trains) with different speeds.
    speed_ratio_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(num_cities=5,
                                                       # Number of cities in map (where train stations are)
                                                       num_intersections=4,
                                                       # Number of intersections (no start / target)
                                                       num_trainstations=10,  # Number of possible start/targets on map
                                                       min_node_dist=3,  # Minimal distance of nodes
                                                       node_radius=2,  # Proximity of stations to city center
                                                       num_neighb=3,
                                                       # Number of connections to other cities/intersections
                                                       seed=15,  # Random seed
                                                       grid_mode=True,
                                                       enhance_intersection=False
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ratio_map),
                  number_of_agents=n_agents,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=observation_helper)
    env.reset(True, True)

    handle = env.get_agent_handles()
    state_size = env.width * env.height
    action_size = 5

    # We set the number of episodes we would like to train on if no args were specified
    if 'n_episodes' not in locals():
        n_episodes = 60000

    # Set max number of steps per episode as well as other training relevant parameter
    max_steps = int((env.height + env.width))
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

    # Initialize the agent
    agent = DQNAgent(action_size, double_dqn=True)

    # Here you can pre-load an agent
    #if False:
    #    with path(torch_training.Nets, "avoid_checkpoint500.pth") as file_in:
    #        agent.qnetwork_local.load_state_dict(torch.load(file_in))

    # Do training over n_episodes
    for episodes in range(1, n_episodes + 1):
        """
        Training Curriculum: In order to get good generalization we change the number of agents
        and the size of the levels every 50 episodes.
        """
        if episodes % 50 == 1:
            env = RailEnv(width=x_dim,
                          height=y_dim,
                          rail_generator=sparse_rail_generator(num_cities=5,
                                                               # Number of cities in map (where train stations are)
                                                               num_intersections=4,
                                                               # Number of intersections (no start / target)
                                                               num_trainstations=10,
                                                               # Number of possible start/targets on map
                                                               min_node_dist=3,  # Minimal distance of nodes
                                                               node_radius=2,  # Proximity of stations to city center
                                                               num_neighb=3,
                                                               # Number of connections to other cities/intersections
                                                               seed=15,  # Random seed
                                                               grid_mode=True,
                                                               enhance_intersection=False
                                                               ),
                          schedule_generator=sparse_schedule_generator(speed_ratio_map),
                          number_of_agents=n_agents,
                          stochastic_data=stochastic_data,  # Malfunction data generator
                          obs_builder_object=observation_helper)

            # Adjust the parameters according to the new env.
            max_steps = int((env.height + env.width))
            agent_obs = [None] * env.get_num_agents()
            agent_next_obs = [None] * env.get_num_agents()

        # Reset environment
        obs = env.reset(True, True)

        # Setup placeholder for finals observation of a single agent. This is necessary because agents terminate at
        # different times during an episode
        final_obs = agent_obs.copy()
        final_obs_next = agent_next_obs.copy()
        register_action_state = np.zeros(env.get_num_agents(), dtype=bool)

        # Build agent specific observations
        for a in range(env.get_num_agents()):
            agent_obs[a] = preprocess_obs(obs[a])

        score = 0
        env_done = 0

        # Run episode
        for step in range(max_steps):

            # Action
            for a in range(env.get_num_agents()):
                if env.agents[a].speed_data['position_fraction'] == 0.:
                    register_action_state[a] = True
                else:
                    register_action_state[a] = False
                action = agent.act(agent_obs[a], eps=eps)
                action_prob[action] += 1
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            # Build agent specific observations and normalize
            for a in range(env.get_num_agents()):
                agent_next_obs[a] = preprocess_obs(obs[a])

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                if done[a]:
                    final_obs[a] = agent_obs[a].copy()
                    final_obs_next[a] = agent_next_obs[a].copy()
                    final_action_dict.update({a: action_dict[a]})
                if not done[a] and register_action_state[a]:
                    agent.step(agent_obs[a], action_dict[a], all_rewards[a], agent_next_obs[a], done[a])
                score += all_rewards[a] / env.get_num_agents()

            # Copy observation
            agent_obs = agent_next_obs.copy()

            if done['__all__']:
                env_done = 1
                for a in range(env.get_num_agents()):
                    agent.step(final_obs[a], final_action_dict[a], all_rewards[a], final_obs_next[a], done[a])
                break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Collection information about training
        tasks_finished = 0
        for _idx in range(env.get_num_agents()):
            if done[_idx] == 1:
                tasks_finished += 1
        done_window.append(tasks_finished / env.get_num_agents())
        scores_window.append(score / max_steps)  # Save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                episodes,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, action_prob / np.sum(action_prob)), end=" ")

        if episodes % 100 == 0:
            print(
                '\rTraining {} Agents.\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(),
                    episodes,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    eps,
                    action_prob / np.sum(action_prob)))
            torch.save(agent.qnetwork_local.state_dict(),
                       './Nets/avoid_checkpoint' + str(episodes) + '.pth')
            action_prob = [1] * action_size
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
