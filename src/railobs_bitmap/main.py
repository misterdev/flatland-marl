# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
import os

from collections import deque

# These 2 lines must go before the import from src/
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from src.railobs_bitmap.rail_observations import RailObsForRailEnv
from src.railobs_bitmap.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnvActions

from src.railobs_bitmap.preprocessing import preprocess_obs
from src.railobs_bitmap.agent import DQNAgent


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
	
	prediction_builder = ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth)
	observation_builder = RailObsForRailEnv(predictor=prediction_builder)

	env = RailEnv(width=args.width,
	              height=args.height,
	              rail_generator=rail_generator,
	              random_seed=0,
	              schedule_generator=schedule_generator,
	              number_of_agents=args.num_agents,
	              obs_builder_object=observation_builder,
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
	max_conflicting_agents = 4 # TODO Decide subset of agents to feed
	# max_steps = env.compute_max_episode_steps(env.width, env.height)
	max_steps = 100
	
	dqn = DQNAgent(args, bitmap_height=max_conflicting_agents * max_rails, action_space=2)
	
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
		state, info = env.reset()
		if args.render:
			env_renderer.reset()
		maps = observation_builder.get_initial_bitmaps()

		# TODO : For the moment I'm considering only the shortest path, and no alternative paths
		for step in range(max_steps - 1):
			# rem first bit is 0 for agent not departed
			for a in range(env.get_num_agents()):
				# If two first consecutive bits in the bitmap are the same
				# print(maps[a, :, :])
				if np.all(maps[a, :, 0] == maps[a, :, 1]):
					obs = preprocess_obs(a, maps, max_conflicting_agents, max_rails)
					buffer_obs[a] = obs.copy()		
					update_values[a] = False # Network doesn't need to choose a move and I don't store the experience
					action = RailEnvActions.MOVE_FORWARD
					network_action = 1
					maps[a, :, 0] = 0
					maps[a] = np.roll(maps[a], -1)
				else: # Changing rails - need to perform a move
					update_values[a] = True
					# Print info TODO These are wrong if step = 0 agents not departed
					current_rail = np.argmax(np.absolute(maps[a, :, 0]))
					current_dir = maps[a, current_rail, 0]
					'''
					if maps[a, current_rail, 0] == 0:  # The first el is 0 for an agent READY_TO_DEPART
						if args.debug:
							print("Train {} ready to start".format(a))
					else:
						#print("Train {} on rail {} in direction {}".format(a, current_rail, current_dir))
						assert (maps[a, current_rail, 1] == 0)
					'''
					# Let the network choose the action : current random_move()
					obs = preprocess_obs(a, maps, max_conflicting_agents, max_rails)
					# Save current state in buffer
					buffer_obs[a] = obs.copy()		
					network_action = dqn.act(obs) # Network chooses action
					# Add code to handle bitmap ...
					action, maps = observation_builder.update_bitmaps(a, network_action, maps)

				next_obs[a] = preprocess_obs(a, maps, max_conflicting_agents, max_rails)
				network_action_dict.update({a: network_action})
				railenv_action_dict.update({a: action})
				
			# Obs is computed from bitmaps while state is computed from env step (temporarily)
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
		################### At the end of the episode
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
	args = parser.parse_args()
	main(args)