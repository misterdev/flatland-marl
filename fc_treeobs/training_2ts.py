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
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator

# Import Torch and utility functions to normalize observation
import fc_treeobs.nets
from fc_treeobs.dueling_double_dqn import Agent
from fc_treeobs.utils import norm_obs_clip, split_tree_into_feature_groups


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_episodes="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_episodes>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_episodes'):
            n_episodes = int(arg)

    ## Initialize the random
    random.seed(1)
    np.random.seed(1)

    # Initialize a random map with a random number of agents
    x_dim = np.random.randint(8, 20)
    y_dim = np.random.randint(8, 20)
    n_agents = np.random.randint(3, 8)
    n_goals = n_agents + np.random.randint(0, 3)
    min_dist = int(0.75 * min(x_dim, y_dim))
    tree_depth = 2
    print("main2")
    demo = False

    # Get an observation builder and predictor
    observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                        max_dist=99999,
                                                        seed=0),
                  schedule_generator=complex_schedule_generator(),
                  obs_builder_object=observation_helper,
                  number_of_agents=n_agents)
    env.reset(True, True)

    handle = env.get_agent_handles()
    features_per_node = env.obs_builder.observation_dim
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = 2 * features_per_node * nr_nodes  # We will use two time steps per observation --> 2x state_size
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_episodes' not in locals():
        n_episodes = 60000

    # Set max number of steps per episode as well as other training relevant parameter
    max_steps = int(3 * (env.height + env.width))
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.9995
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
    # Initialize the agent
    agent = Agent(state_size, action_size)

    # Here you can pre-load an agent
    if False:
        with path(torch_training.Nets, "avoid_checkpoint500.pth") as file_in:
            agent.qnetwork_local.load_state_dict(torch.load(file_in))

    # Do training over n_episodes
    for episodes in range(1, n_episodes + 1):
        """
        Training Curriculum: In order to get good generalization we change the number of agents
        and the size of the levels every 50 episodes.
        """
        if episodes % 50 == 0:
            x_dim = np.random.randint(8, 20)
            y_dim = np.random.randint(8, 20)
            n_agents = np.random.randint(3, 8)
            n_goals = n_agents + np.random.randint(0, 3)
            min_dist = int(0.75 * min(x_dim, y_dim))
            env = RailEnv(width=x_dim,
                          height=y_dim,
                          rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                                max_dist=99999,
                                                                seed=0),
                          schedule_generator=complex_schedule_generator(),
                          obs_builder_object=TreeObsForRailEnv(max_depth=3,
                                                               predictor=ShortestPathPredictorForRailEnv()),
                          number_of_agents=n_agents)

            # Adjust the parameters according to the new env.
            max_steps = int(3 * (env.height + env.width))
            agent_obs = [None] * env.get_num_agents()
            agent_next_obs = [None] * env.get_num_agents()

        # Reset environment
        obs, info = env.reset(True, True)

        # Setup placeholder for finals observation of a single agent. This is necessary because agents terminate at
        # different times during an episode
        final_obs = agent_obs.copy()
        final_obs_next = agent_next_obs.copy()

        # Build agent specific observations
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree_into_feature_groups(obs[a], tree_depth)
            data = norm_obs_clip(data)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        # Accumulate two time steps of observation (Here just twice the first state)
        for i in range(2):
            time_obs.append(obs)

        # Build the agent specific double ti
        for a in range(env.get_num_agents()):
            agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))

        score = 0
        env_done = 0
        # Run episode
        for step in range(max_steps):

            # Action
            for a in range(env.get_num_agents()):
                if demo:
                    eps = 0
                # action = agent.act(np.array(obs[a]), eps=eps)
                action = agent.act(agent_obs[a], eps=eps)
                action_prob[action] += 1
                action_dict.update({a: action})
            # Environment step

            next_obs, all_rewards, done, _ = env.step(action_dict)
            for a in range(env.get_num_agents()):
                data, distance, agent_data = split_tree_into_feature_groups(next_obs[a], tree_depth)
                data = norm_obs_clip(data)
                distance = norm_obs_clip(distance)
                agent_data = np.clip(agent_data, -1, 1)
                next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))
            time_obs.append(next_obs)

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                agent_next_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))
                if done[a]:
                    final_obs[a] = agent_obs[a].copy()
                    final_obs_next[a] = agent_next_obs[a].copy()
                    final_action_dict.update({a: action_dict[a]})
                if not demo and not done[a]:
                    agent.step(agent_obs[a], action_dict[a], all_rewards[a], agent_next_obs[a], done[a])
                score += all_rewards[a] / env.get_num_agents()

            agent_obs = agent_next_obs.copy()
            if done['__all__']:
                env_done = 1
                for a in range(env.get_num_agents()):
                    agent.step(final_obs[a], final_action_dict[a], all_rewards[a], final_obs_next[a], done[a])
                break
        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        done_window.append(env_done)
        scores_window.append(score / max_steps)  # save most recent score
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
                       './nets/avoid_checkpoint' + str(episodes) + '.pth')
            action_prob = [1] * action_size
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
