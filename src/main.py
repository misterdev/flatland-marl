# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
import os

from collections import deque

# These 2 lines must go before the import from src/
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from src.rail_observations import RailObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnvActions

from src.preprocessing import preprocess_obs
from src.agent import DQNAgent

import src.utils.debug as debug

def main(args):
	rail_generator = sparse_rail_generator(max_num_cities=args.max_num_cities,
	                                       seed=args.seed,
	                                       grid_mode=args.grid_mode,
	                                       max_rails_between_cities=args.max_rails_between_cities,
	                                       max_rails_in_city=args.max_rails_in_city,
	                                       )

	# Maps speeds to % of appearance in the env
	# TODO! temporary set all speed to 1
	speed_ration_map = {1.: 1}  # Slow freight train
	# speed_ration_map = {1.: 0.25,  # Fast passenger train
	#                     1. / 2.: 0.25,  # Fast freight train
	#                     1. / 3.: 0.25,  # Slow commuter train
	#                     1. / 4.: 0.25}  # Slow freight train

	schedule_generator = sparse_schedule_generator(speed_ration_map)
	
	prediction_builder = ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth)
	obs_builder = RailObsForRailEnv(predictor=prediction_builder)

	env = RailEnv(width=args.width,
	              height=args.height,
	              rail_generator=rail_generator,
	              random_seed=0,
	              schedule_generator=schedule_generator,
	              number_of_agents=args.num_agents,
	              obs_builder_object=obs_builder,
	              malfunction_generator_and_process_data=malfunction_from_params(
		              parameters={
			              'malfunction_rate': args.malfunction_rate,
			              'min_duration': args.min_duration,
			              'max_duration': args.max_duration
		              })
	              )

	if args.render:
		env_renderer = RenderTool(
			env,
			agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
			show_debug=True,
			screen_height=1080,
			screen_width=1920)

	max_rails = 100 # TODO Must be a parameter of the env (estimated)
	# max_steps = env.compute_max_episode_steps(env.width, env.height)
	max_steps = 100
	
	dqn = DQNAgent(args, bitmap_height=max_rails * 3, action_space=2)
	
	if args.render:
		file = os.path.isfile("checkpoints/"+args.model_name)
		if file:
			dqn.qnetwork_local.load_state_dict(torch.load(file))
			
	eps = 1.
	eps_end = 0.005
	railenv_action_dict = {}
	network_action_dict = {}
	scores_window = deque(maxlen=100)
	done_window = deque(maxlen=100)
	scores = []
	dones_list = []
	update_values = [False] * args.num_agents
	buffer_obs = [None] * args.num_agents
	next_obs = [None] * args.num_agents

	############ Main loop
	for ep in range(args.num_episodes):
		score = 0
		env_done = 0
		_, info = env.reset()
		if args.render:
			env_renderer.reset()
		maps = obs_builder.get_initial_bitmaps(args.print)

		if args.print:
			debug.print_bitmaps(maps)

		for step in range(max_steps - 1):
			# rem first bit is 0 for agent not departed
			for a in range(env.get_num_agents()):
				network_action = None
				# TODO evaluate only once
				agent_speed = env.agents[a].speed_data["speed"]
				times_per_cell = int(np.reciprocal(agent_speed))
				# If two first consecutive bits in the bitmap are the same
				if np.all(maps[a, :, 0] == maps[a, :, times_per_cell]) or not info['action_required'][a]:
					obs = preprocess_obs(a, maps[a], maps, max_rails)
					buffer_obs[a] = obs.copy()
					update_values[a] = False # Network doesn't need to choose a move and I don't store the experience
					
					network_action = 1
					maps = obs_builder.unroll_bitmap(a)
					action = obs_builder.get_agent_action(a)

				else: # Changing rails - need to perform a move
					# TODO check how this works with new action pick mehanic
					altmaps, predictions = obs_builder.get_altmaps(a)

					if len(altmaps) > 1:
						q_values = np.array([])
						for i in range(len(altmaps)):
							obs = preprocess_obs(a, altmaps[i], maps, max_rails)
							q_values = np.concatenate([q_values, dqn.act(obs)])

						# Epsilon-greedy action selection
						if np.random.random() > eps:
							argmax = np.argmax(q_values)
							network_action = argmax % 2
							best_i = argmax // 2
						else:
							network_action = np.random.choice([0, 1])
							best_i = np.random.choice(np.arange(len(altmaps)))

						# Update bitmaps and predictions
						maps[a, :, :] = altmaps[best_i]
						obs_builder.prediction_dict[a] = predictions[best_i]
					
					else: # Continue on the same path
						q_values = dqn.act(obs) # Network chooses action
						if np.random.random() > eps:
							network_action = np.argmax(q_values)
						else:
							network_action = np.random.choice([0, 1])	

					obs = preprocess_obs(a, maps[a], maps, max_rails)
					update_values[a] = True
					# Save current state in buffer
					buffer_obs[a] = obs.copy()
					# Update bitmaps and get new action
					# TODO? detect crash function
					# TODO? get_action function
					action, maps, crash = obs_builder.update_bitmaps(a, network_action, maps)

					if args.train and crash:
						network_action = 0 # TODO! are you sure?
						print('ADDING CRASH TUPLE')
						dqn.step(buffer_obs[a], 1, -2000, buffer_obs[a], True)

					next_obs[a] = preprocess_obs(a, maps[a], maps, max_rails)

				network_action_dict.update({a: network_action})
				railenv_action_dict.update({a: action})

			# Obs is computed from bitmaps while state is computed from env step (temporarily)
			# TODO? return bitmaps as state?
			_, reward, done, info = env.step(railenv_action_dict)  # Env step

			if args.render:
				env_renderer.render_env(show=True, show_observations=False, show_predictions=True)
			
			if args.debug:
				for a in range(env.get_num_agents()):
					print('#########################################')
					print('Info for agent {}'.format(a))
					print('Status: {}'.format(info['status'][a]))
					print('Position: {}'.format(env.agents[a].position))
					print('Target: {}'.format(env.agents[a].target))
					print('Moving? {} at speed: {}'.format(env.agents[a].moving, info['speed'][a]))
					print('Action required? {}'.format(info['action_required'][a]))
					print('Network action: {}'.format(network_action_dict[a]))
					print('Railenv action: {}'.format(railenv_action_dict[a]))
					#print('Q values: {}'.format(q_values[a]))
			
			# Update replay buffer and train agent
			if args.train:
				for a in range(env.get_num_agents()):
					if update_values[a] or done[a]:
						dqn.step(buffer_obs[a], network_action_dict[a], reward[a], next_obs[a], done[a])
						buffer_obs[a] = next_obs[a].copy()
			
			for a in range(env.get_num_agents()):	
				score += reward[a] / env.get_num_agents() # Update score
				
			if done['__all__']:
				env_done = 1
				break

		################### End of the episode
		eps = max(eps_end, args.eps_decay * eps)  # Decrease epsilon
		# Metrics
		done_window.append(env_done)
		num_agents_done = 0  # Num of agents that reached their target
		for a in range(env.get_num_agents()): # TODO: env sets done[all] = True for everyone when time limit is reached
			if done[a]:
				num_agents_done += 1

		scores_window.append(score / max_steps)  # Save most recent score
		scores.append(np.mean(scores_window))
		dones_list.append((np.mean(done_window)))

		# Print training results info
		print(
			'\r{} Agents on ({},{}).\t Ep: {}\t Avg Score: {:.3f}\t Env Dones so far: {:.2f}%\t Done Agents in ep: {:.2f}%\t Eps: {:.2f}'.format(
				env.get_num_agents(), args.width, args.height,
				ep,
				np.mean(scores_window),
				100 * np.mean(done_window),
				100 * (num_agents_done / args.num_agents),
				eps), end=" ")

		if ep % 20 == 0:
			print(
				'\rTraining {} Agents.\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t'.format(
					env.get_num_agents(),
					ep,
					np.mean(scores_window),
					100 * np.mean(done_window),
					100 * (num_agents_done / args.num_agents),
					eps))
			if args.train:
				torch.save(dqn.qnetwork_local.state_dict(), './checkpoints/' + str(args.model_name) + str(ep) + '.pth')
		
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Railobs')
	# Env parameters
	parser.add_argument('--network-action-space', type=int, default=2, help='Number of actions allowed in the environment')
	parser.add_argument('--width', type=int, default=20, help='Environment width')
	parser.add_argument('--height', type=int, default=20, help='Environment height')
	parser.add_argument('--num-agents', type=int, default=4, help='Number of agents in the environment')
	parser.add_argument('--max-num-cities', type=int, default=3, help='Maximum number of cities where agents can start or end')
	parser.add_argument('--seed', type=int, default=1, help='Seed used to generate grid environment randomly')
	parser.add_argument('--grid-mode', type=bool, default=True, help='Type of city distribution, if False cities are randomly placed')
	parser.add_argument('--max-rails-between-cities', type=int, default=2, help='Max number of tracks allowed between cities, these count as entry points to a city')
	parser.add_argument('--max-rails-in-city', type=int, default=3, help='Max number of parallel tracks within a city allowed')
	parser.add_argument('--malfunction-rate', type=int, default=2000, help='Rate of malfunction occurrence of single agent')
	parser.add_argument('--min-duration', type=int, default=0, help='Min duration of malfunction')
	parser.add_argument('--max-duration', type=int, default=0, help='Max duration of malfunction')
	parser.add_argument('--observation-builder', type=str, default='GraphObsForRailEnv', help='Class to use to build observation for agent')
	parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help='Class used to predict agent paths and help observation building')
	parser.add_argument('--bfs-depth', type=int, default=4, help='BFS depth of the graph observation')
	parser.add_argument('--prediction-depth', type=int, default=500, help='Prediction depth for shortest path strategy, i.e. length of a path')
	# Training
	parser.add_argument('--model-name', type=str, default="ddqn-replay-buffer", help="Model name")
	parser.add_argument('--num-episodes', type=int, default=15000, help="Number of episodes to run")
	parser.add_argument('--eps-decay', type=float, default=0.998, help="Factor to decrease eps in eps-greedy")
	# Misc
	parser.add_argument('--debug', action='store_true', help='Print debug info')
	parser.add_argument('--render', action='store_true', help='Render map')
	parser.add_argument('--train', action='store_true', help='Perform training')
	parser.add_argument('--print', action='store_true', help='Save internal representations as files')
	args = parser.parse_args()
	main(args)