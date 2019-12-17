# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import sys
import numpy as np
import torch
from tqdm import trange
from knockknock import telegram_sender
from pathlib import Path

# These 2 lines must go before the import from src/
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_file, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

from src.state_machine.state_machine import act

def main(args):
	
	
	rail_generator = sparse_rail_generator(max_num_cities=args.max_num_cities,
	                                       #seed=args.seed,
	                                       seed=0, # 0, 3, 7, 10, 14, 16, 20, 22, 23, 25, 26, 32
	                                       grid_mode=args.grid_mode,
	                                       max_rails_between_cities=args.max_rails_between_cities,
	                                       max_rails_in_city=args.max_rails_in_city,
	                                       )
	
	# Maps speeds to % of appearance in the env
	speed_ration_map = {1.: 0.25,  # Fast passenger train
	                    1. / 2.: 0.25,  # Fast freight train
	                    1. / 3.: 0.25,  # Slow commuter train
	                    1. / 4.: 0.25}  # Slow freight train

	observation_builder = GraphObsForRailEnv(bfs_depth=args.bfs_depth,
	                                         predictor=ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth))

	# Construct the environment with the given observation, generators, predictors, and stochastic data
	env = RailEnv(width=args.width,
	              height=args.height,
	              rail_generator= rail_generator, # rail_from_file boh...
	              schedule_generator=sparse_schedule_generator(speed_ration_map),
	              number_of_agents=args.num_agents,
	              obs_builder_object=observation_builder,
	              malfunction_generator_and_process_data=malfunction_from_params(
		              parameters={
		              'malfunction_rate': args.malfunction_rate,  # Rate of malfunction occurrence
		              'min_duration': args.min_duration,  # Minimal duration of malfunction
		              'max_duration': args.max_duration  # Max duration of malfunction
	              }))

	env_renderer = RenderTool(
		env,
		agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
		show_debug=True,
		screen_height=1080,
		screen_width=1920)

	network_action_dict = {}
	railenv_action_dict = {}
	max_time_steps = 150
	T_rewards = []  # List of episodes rewards
	T_Qs = []  # List of q values
	T_num_done_agents = []  # List of number of done agents for each episode
	T_all_done = []  # If all agents completed in each episode
	
	for ep in range(args.num_episodes):
		# Reset info at the beginning of an episode
		# env.load(filename="map" + str(ep))
		state, info = env.reset()
		# env.save(filename="map" + str(ep))
		env_renderer.reset()
		reward_sum, all_done = 0, False  # reward_sum contains the cumulative reward obtained as sum during the steps
		num_done_agents = 0
		
		for step in range(max_time_steps):
			
			for a in range(env.get_num_agents()):
				network_action = act(args, state[a]) # State machine picks action
				railenv_action = observation_builder.choose_railenv_action(a, network_action)
				railenv_action_dict.update({a: railenv_action})
				network_action_dict.update({a: network_action})
				
			state, reward, done, info = env.step(railenv_action_dict)  # Env step
			env_renderer.render_env(show=True, show_observations=False, show_predictions=True)
			
			for a in range(env.get_num_agents()):
				print('#########################################')
				print('Info for agent {}'.format(a))
				print('Occupancy, first layer: {}'.format(state[a][:args.prediction_depth]))
				print('Occupancy, second layer: {}'.format(state[a][args.prediction_depth:args.prediction_depth * 2]))
				print('Forks: {}'.format(state[a][args.prediction_depth * 2:args.prediction_depth * 3]))
				print('Target: {}'.format(state[a][args.prediction_depth * 3:args.prediction_depth * 4]))
				print('Priority: {}'.format(state[a][args.prediction_depth * 4]))
				print('Max priority encountered: {}'.format(state[a][args.prediction_depth * 4 + 1]))
				print('Num malfunctoning agents (globally): {}'.format(state[a][args.prediction_depth * 4 + 2]))
				print('Num agents ready to depart (globally): {}'.format(state[a][args.prediction_depth * 4 + 3]))
				print('Status: {}'.format(info['status'][a]))
				print('Position: {}'.format(env.agents[a].position))
				print('Moving? {} at speed: {}'.format(env.agents[a].moving, info['speed'][a]))
				print('Action required? {}'.format(info['action_required'][a]))
				print('Network action: {}'.format(network_action_dict[a]))
				print('Railenv action: {}'.format(railenv_action_dict[a]))
				# print('Q values: {}'.format(qvalues[a]))
				print('Rewards: {}'.format(reward[a]))
				
			reward_sum += sum(reward[a] for a in range(env.get_num_agents()))

			if done['__all__']:
				all_done = True
				break
		# No need to close the renderer since env parameter sizes stay the same
		T_rewards.append(reward_sum)
		# Compute num of agents that reached their target
		for a in range(env.get_num_agents()):
			if done[a]:
				num_done_agents += 1
		T_num_done_agents.append(num_done_agents / env.get_num_agents())  # In proportion to total
		T_all_done.append(all_done)

	avg_done_agents = sum(T_num_done_agents) / len(
		T_num_done_agents)  # Average number of agents that reached their target
	avg_reward = sum(T_rewards) / len(T_rewards)
	avg_norm_reward = avg_reward / (max_time_steps / env.get_num_agents())
	
	print("Avg. done agents: {}".format(avg_done_agents))
	print("Avg. reward: {}".format(avg_reward))
	print("Avg. norm reward: {}".format(avg_norm_reward))
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='State machine')
	# Env parameters
	parser.add_argument('--network-action-space', type=int, default=2, help='Number of actions allowed in the environment')
	parser.add_argument('--width', type=int, default=20, help='Environment width')
	parser.add_argument('--height', type=int, default=20, help='Environment height')
	parser.add_argument('--num-agents', type=int, default=4, help='Number of agents in the environment')
	parser.add_argument('--max-num-cities', type=int, default=3, help='Maximum number of cities where agents can start or end')
	#parser.add_argument('--seed', type=int, default=1, help='Seed used to generate grid environment randomly')
	parser.add_argument('--grid-mode', type=bool, default=True, help='Type of city distribution, if False cities are randomly placed')
	parser.add_argument('--max-rails-between-cities', type=int, default=2, help='Max number of tracks allowed between cities, these count as entry points to a city')
	parser.add_argument('--max-rails-in-city', type=int, default=3, help='Max number of parallel tracks within a city allowed')
	parser.add_argument('--malfunction-rate', type=int, default=2000, help='Rate of malfunction occurrence of single agent')
	parser.add_argument('--min-duration', type=int, default=0, help='Min duration of malfunction')
	parser.add_argument('--max-duration', type=int, default=0, help='Max duration of malfunction')
	parser.add_argument('--observation-builder', type=str, default='GraphObsForRailEnv', help='Class to use to build observation for agent')
	parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help='Class used to predict agent paths and help observation building')
	parser.add_argument('--bfs-depth', type=int, default=4, help='BFS depth of the graph observation')
	parser.add_argument('--prediction-depth', type=int, default=60, help='Prediction depth for shortest path strategy, i.e. length of a path')
	parser.add_argument('--num-episodes', type=int, default=20, help='Number of episodes to run')

	args = parser.parse_args()
	main(args)