from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
# make sure the root path is in system path
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params

# We also include a renderer because we want to visualize what is going on in the environment

from fc_graphobs.graph_observations import GraphObsForRailEnv
from fc_graphobs.predictions import ShortestPathPredictorForRailEnv

# I can import files from the other folder
from fc_graphobs.dueling_double_dqn_mod import Agent
from fc_graphobs.print_info import print_info


def main():

    width = 50  # Width of map
    height = 50  # Height of map
    nr_trains = 10  # Number of trains that have an assigned task in the env
    cities_in_map = 6  # Number of cities where agents can start or end
    seed = 5  # Random seed
    grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
    max_rails_between_cities = 4  # Max number of tracks allowed between cities. This is number of entry point to a city
    max_rail_in_city = 6  # Max number of parallel tracks within a city, representing a realistic train station

    rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                           seed=seed,
                                           grid_mode=grid_distribution_of_cities,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rails_in_city=max_rail_in_city,
                                           )
    # Maps speeds to % of appearance in the env
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    schedule_generator = sparse_schedule_generator(speed_ration_map)

    stochastic_data = {
        'malfunction_rate': 500,  # Rate of malfunction occurrence of single agent
        'min_duration': 20,  # Minimal duration of malfunction
        'max_duration': 21  # Max duration of malfunction
        }

    prediction_depth = 40
    bfs_depth = 4
    observation_builder = GraphObsForRailEnv(bfs_depth=bfs_depth, predictor=ShortestPathPredictorForRailEnv(max_depth=prediction_depth))

    # Construct the environment with the given observation, generators, predictors, and stochastic data
    env = RailEnv(width=width,
                  height=height,
                  rail_generator=rail_generator,
                  schedule_generator=schedule_generator,
                  number_of_agents=nr_trains,
                  obs_builder_object=observation_builder,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  remove_agents_at_target=True)
    env.reset()
    # Hardcoded params
    state_size = prediction_depth + 3
    network_action_size = 2  # {follow path, stop}
    railenv_action_size = 5  # The RailEnv possible actions
    n_episodes = 1000
    max_steps = int(4 * 2 * (width + height + nr_trains/cities_in_map))
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.99995
    # Need to have two since env works with RailEnv actions but agent works with network actions
    network_action_dict = dict()
    railenv_action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * railenv_action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()

    agent = Agent(state_size=state_size, action_size=network_action_size, double_dqn=True)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(True, True)
        final_obs = agent_obs.copy()
        final_obs_next = agent_next_obs.copy()

        # Normalize obs TODO now it does nothing
        for a in range(env.get_num_agents()):
            agent_obs[a] = obs[a]

        score = 0
        env_done = 0

        # Pick first action
        for a in range(env.get_num_agents()):
            shortest_path_action = observation_builder.get_shortest_path_action(a)
            # 'railenv_action' is in [0, 4], network_action' is in [0, 1]
            # 'network_action' is None if act() returned a random sampled action
            railenv_action, network_action = agent.act(agent_obs[a], shortest_path_action, eps=eps)
            action_prob[railenv_action] += 1
            railenv_action_dict.update({a: railenv_action})
            network_action_dict.update({a: network_action})

        # Environment step
        next_obs, all_rewards, done, infos = env.step(railenv_action_dict)
        for step in range(max_steps - 1):

            # Logging
            #print_info(env)

            for a in infos['action_required']:
                # Agent performs action only if required
                    shortest_path_action = observation_builder.get_shortest_path_action(a)
                    # 'railenv_action' is in [0, 4], network_action' is in [0, 1]
                    # 'network_action' is None if act() returned a random sampled action
                    railenv_action, network_action = agent.act(agent_obs[a], shortest_path_action, eps=eps)
                    action_prob[railenv_action] += 1
                    railenv_action_dict.update({a: railenv_action})
                    network_action_dict.update({a: network_action})

            # Environment step
            next_obs, all_rewards, done, infos = env.step(railenv_action_dict)

            # Which agents needs to pick and action
            '''
            print("\n The following agents can register an action:")
            print("========================================")
            for info in infos['action_required']:
                print("Agent {} needs to submit an action.".format(info))
            '''
            # Normalize obs TODO
            for a in range(env.get_num_agents()):
                agent_next_obs[a] = next_obs[a]
                if done[a]:             
                    final_obs[a] = agent_obs[a].copy()
                    final_obs_next[a] = agent_next_obs[a].copy()
                    final_action_dict.update({a: network_action_dict[a]})
                    
                else:
                    agent.step(agent_obs[a], network_action_dict[a], all_rewards[a], agent_next_obs[a], done[a])

                score += all_rewards[a] / env.get_num_agents()  # Update score

            agent_obs = agent_next_obs.copy()
            if done['__all__']:
                env_done = 1
                # Perform last actions separately
                for a in range(env.get_num_agents()):
                    agent.step(final_obs[a], final_action_dict[a], all_rewards[a], final_obs_next[a], done[a])
                break
        
        # At the end of the episode
        eps = max(eps_end, eps_decay * eps)  # Decrease epsilon
        # Metrics
        done_window.append(env_done)
        num_agents_done = 0  # Num of agents that reached their target
        for a in range(env.get_num_agents()):
            if done[a]:
                num_agents_done += 1

        scores_window.append(score / max_steps)  # Save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        action_prob_float = action_prob / np.sum(action_prob)
        formatted_action_prob = ['{:.5f}'.format(ap) for ap in action_prob_float]

        # Print training results info
        print(
            '\r{} Agents on ({},{}).\t Ep: {}\t Avg Score: {:.3f}\t Env Dones so far: {:.2f}%\t Done Agents in ep: {:.2f}%\t Eps: {:.2f}\t Action Probs: {} '.format(
                env.get_num_agents(), width, height,
                ep,
                np.mean(scores_window),
                100 * np.mean(done_window),
                100 * (num_agents_done/nr_trains),
                eps,
                formatted_action_prob), end=" ")

        if ep % 100 == 0:
            torch.save(agent.qnetwork_local.state_dict(),'./nets/avoid_checkpoint' + str(ep) + '.pth')
            action_prob = [1] * railenv_action_size

    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
