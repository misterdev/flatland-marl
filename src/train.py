from collections import deque
import numpy as np
import torch
import sys
import argparse
import pprint
# make sure the root path is in system path
from pathlib import Path
# These 2 lines must go before the import from src/
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params#, MalfunctionParameters

from src.graph_observations import GraphObsForRailEnv
from src.local_observations import LocalObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv
from src.utils import preprocess_obs

from src.dueling_double_dqn import Agent
from src.print_info import print_info



def main(args):

    rail_generator = sparse_rail_generator(max_num_cities=args.max_num_cities,
                                           seed=args.seed,
                                           grid_mode=args.grid_mode,
                                           max_rails_between_cities=args.max_rails_between_cities,
                                           max_rails_in_city=args.max_rails_in_city,
                                           )
    # Maps speeds to % of appearance in the env
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    schedule_generator = sparse_schedule_generator(speed_ration_map)
    
    ''' THIS WORKS WITH NEXT VERSION
    stochastic_data = MalfunctionParameters(
        malfunction_rate=args.malfunction_rate,  # Rate of malfunction occurrence of single agent
        min_duration=args.min_duration,  # Minimal duration of malfunction
        max_duration=args.max_duration  # Max duration of malfunction
    )
    '''
    
    stochastic_data = {
        'malfunction_rate' : args.malfunction_rate,
        'min_duration' : args.min_duration,
        'max_duration' : args.max_duration
    }
    
    if args.observation_builder == 'GraphObsForRailEnv':
        
        prediction_depth = args.prediction_depth
        bfs_depth = args.bfs_depth
        observation_builder = GraphObsForRailEnv(bfs_depth=bfs_depth, predictor=ShortestPathPredictorForRailEnv(max_depth=prediction_depth))
        state_size = args.prediction_depth * 3 + 4 # TODO
        network_action_size = 2  # {follow path, stop}
        railenv_action_size = 5  # The RailEnv possible actions
        agent = Agent(network_type='fc', state_size=state_size, action_size=network_action_size)

    elif args.observation_builder == 'LocalObsForRailEnv':
        
        observation_builder = LocalObsForRailEnv(args.view_semiwidth, args.view_height, args.offset)
        #state_size = (2 * args.view_semiwidth + 1) * args.height
        state_size = 16 + 5 + 2 # state_size == in_channels
        railenv_action_size = 5
        agent = Agent(network_type='conv', state_size=state_size, action_size=railenv_action_size)

    # Construct the environment with the given observation, generators, predictors, and stochastic data
    env = RailEnv(width=args.width,
                  height=args.height,
                  rail_generator=rail_generator,
                  schedule_generator=schedule_generator,
                  number_of_agents=args.num_agents,
                  obs_builder_object=observation_builder,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  remove_agents_at_target=True)
    env.reset()

    # max_steps = env.compute_max_episode_steps(args.width, args.height, args.num_agents/args.max_num_cities)
    max_steps = 200 # TODO DEBUG
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998
    # Need to have two since env works with RailEnv actions but agent works with network actions
    network_action_dict = dict()
    railenv_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * railenv_action_size
    agent_obs = [None] * env.get_num_agents()
    agent_obs_buffer = [None] * env.get_num_agents()
    agent_action_buffer = [2] * env.get_num_agents()
    update_values = [False] * env.get_num_agents()  # Used to update agent if action was performed in this step
    qvalues = {}
    
    for ep in range(1, args.num_episodes + 1):
        
        obs, info = env.reset()

        if args.observation_builder == 'GraphObsForRailEnv':
            for a in range(env.get_num_agents()):
                agent_obs[a] = obs[a].copy()
                agent_obs_buffer[a] = agent_obs[a].copy()
            for a in range(env.get_num_agents()):
                action = np.random.choice((0, 2))
                railenv_action_dict.update({a: action})  # All'inizio faccio partire a random TODO Prova
            next_obs, all_rewards, done, info = env.step(railenv_action_dict)
        # Normalize obs, only for LocalObs now
        elif args.observation_builder == 'LocalObsForRailEnv':       
            for a in range(env.get_num_agents()):
                if obs[a]:
                    agent_obs[a] = preprocess_obs(obs[a])
                    agent_obs_buffer[a] = agent_obs[a].copy()
        
        score = 0
        env_done = 0

        
        ############# Main loop
        for step in range(max_steps - 1):
            '''
            print(
                '\r{} Agents on ({},{}).\t Ep: {}\t Step/MaxSteps: {} / {}'.format(
                    env.get_num_agents(), args.width, args.height,
                    ep,
                    step,
                    max_steps), end=" ")
            '''
            # Logging
            #print_info(env)
    
            for a in range(env.get_num_agents()):
                
                if args.observation_builder == 'GraphObsForRailEnv':
                    if info['action_required'][a]:
                        # 'railenv_action' is in [0, 4], network_action' is in [0, 1]
                        network_action = agent.act(agent_obs[a], eps=eps)
                        # Pick railenv action according to network decision if it's safe to go or to stop
                        railenv_action = observation_builder.choose_railenv_action(a, network_action)
                        update_values[a] = True
                        qvalues.update({a: agent.get_q_values(agent_obs[a])})
                    else:
                        network_action = 0
                        railenv_action = 0
                        update_values[a] = False
                        qvalues.update({a: [0, 0]})
                    # Update action dicts
                    action_prob[railenv_action] += 1
                    railenv_action_dict.update({a: railenv_action})
                    network_action_dict.update({a: network_action})
    
                elif args.observation_builder == 'LocalObsForRailEnv':
                    if info['action_required'][a]:
                        railenv_action = agent.act(agent_obs[a], eps=eps)
                        update_values[a] = True
                    else:
                        railenv_action = 0 # If action is not required DO_NOTHING
                        update_values[a] = False
                    action_prob[railenv_action] += 1
                    railenv_action_dict.update({a: railenv_action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(railenv_action_dict)
            if step == 100:
                print('QValues: {}'.format(qvalues))
            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                if update_values[a] or done[a]:
                    if args.observation_builder == 'GraphObsForRailEnv':
                        agent.step(agent_obs_buffer[a], network_action_dict[a], all_rewards[a], agent_obs[a], done[a])
                    else:
                        agent.step(agent_obs_buffer[a], network_action_dict[a], all_rewards[a], agent_obs[a], done[a])
                    agent_obs_buffer[a] = agent_obs[a].copy()
                    '''
                    if args.observation_builder == 'GraphObsForRailEnv':
                        agent_action_buffer[a] = network_action_dict[a]
                    elif args.observation_builder == 'LocalObsForRailEnv':
                        agent_action_buffer[a] = railenv_action_dict[a]
                    ''' 
                # Preprocessing and normalization
                if args.observation_builder == 'GraphObsForRailEnv':
                    agent_obs[a] = next_obs[a].copy()
                if args.observation_builder == 'LocalObsForRailEnv' and next_obs[a]:
                    agent_obs[a] = preprocess_obs(next_obs[a])
                
                score += all_rewards[a] / env.get_num_agents()  # Update score

            if done['__all__']:
                env_done = 1
                break
    
        ################### At the end of the episode TODO This part could be done in another script
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
                env.get_num_agents(), args.width, args.height,
                ep,
                np.mean(scores_window),
                100 * np.mean(done_window),
                100 * (num_agents_done/args.num_agents),
                eps,
                formatted_action_prob), end=" ")

        if ep % 50 == 0:
            print(
                '\rTraining {} Agents.\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(),
                    ep,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    100 * (num_agents_done/args.num_agents),
                    eps,
                    formatted_action_prob))
            torch.save(agent.qnetwork_local.state_dict(),'./nets/' + str(args.model_name) + str(ep) + '.pth')
            action_prob = [1] * railenv_action_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Flatland')
    # Flatland parameters
    parser.add_argument('--width', type=int, default=100, help='Environment width')
    parser.add_argument('--height', type=int, default=100, help='Environment height')
    parser.add_argument('--num-agents', type=int, default=50, help='Number of agents in the environment')
    parser.add_argument('--max-num-cities', type=int, default=6, help='Maximum number of cities where agents can start or end')
    parser.add_argument('--seed', type=int, default=1, help='Seed used to generate grid environment randomly')
    parser.add_argument('--grid-mode', type=bool, default=False, help='Type of city distribution, if False cities are randomly placed')
    parser.add_argument('--max-rails-between-cities', type=int, default=4, help='Max number of tracks allowed between cities, these count as entry points to a city')
    parser.add_argument('--max-rails-in-city', type=int, default=6, help='Max number of parallel tracks within a city allowed')
    parser.add_argument('--malfunction-rate', type=int, default=1000, help='Rate of malfunction occurrence of single agent')
    parser.add_argument('--min-duration', type=int, default=20, help='Min duration of malfunction')
    parser.add_argument('--max-duration', type=int, default=50, help='Max duration of malfunction')
    parser.add_argument('--observation-builder', type=str, default='GraphObsForRailEnv', help='Class to use to build observation for agent')
    parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help='Class used to predict agent paths and help observation building')
    parser.add_argument('--bfs-depth', type=int, default=4, help='BFS depth of the graph observation')
    parser.add_argument('--prediction-depth', type=int, default=108, help='Prediction depth for shortest path strategy, i.e. length of a path')
    parser.add_argument('--view-semiwidth', type=int, default=7, help='Semiwidth of field view for agent in local obs')
    parser.add_argument('--view-height', type=int, default=30, help='Height of the field view for agent in local obs')
    parser.add_argument('--offset', type=int, default=25, help='Offset of agent in local obs')
    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes on which to train the agents')
    parser.add_argument('--model-name', type=str, default='avoid_checkpoint', help='Name to use to save the model .pth')
    # DDQN hyperparameters
    
    args = parser.parse_args()
    # Check arguments
    if args.offset > args.height:
        raise ValueError("Agent offset can't be greater than view height in local obs")
    if args.offset < 0:
        raise ValueError("Agent offset must be a positive integer")
    
    main(args)