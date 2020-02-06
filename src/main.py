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
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus

from src.rail_observations import RailObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv
from src.preprocessing import preprocess_obs
from src.agent import DQNAgent

from src.utils.plot import plot_metric
import src.utils.debug as debug

def main(args):
	
	# Show options and values
	print(' ' * 26 + 'Options')
	for k, v in vars(args).items():
		print(' ' * 26 + k + ': ' + str(v))
	# Where to save models
	results_dir = os.path.join('results', args.model_id)
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	
	rail_generator = sparse_rail_generator(max_num_cities=args.max_num_cities,
	                                       seed=args.seed,
	                                       grid_mode=args.grid_mode,
	                                       max_rails_between_cities=args.max_rails_between_cities,
	                                       max_rails_in_city=args.max_rails_in_city,
	                                       )

	# Maps speeds to % of appearance in the env
	# TODO! temporary set all speed to 1
	speed_ration_map = {1.: 1}  # Slow freight train
	'''
	speed_ration_map = {1.: 0.25,  # Fast passenger train
	                    1. / 2.: 0.25,  # Fast freight train
	                    1. / 3.: 0.25,  # Slow commuter train
	                    1. / 4.: 0.25}  # Slow freight train
	'''
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
			screen_height=800,
			screen_width=800)

	max_rails = 100 # TODO Must be a parameter of the env (estimated)
	# max_steps = env.compute_max_episode_steps(env.width, env.height)
	max_steps = 200
	
	dqn = DQNAgent(args, bitmap_height=max_rails * 3, action_space=2)
	
	if args.render:
		file = os.path.isfile("checkpoints/"+args.model_id) # TODO
		if file:
			dqn.qnetwork_local.load_state_dict(torch.load(file))
	
	eps = args.start_eps
	railenv_action_dict = {}
	network_action_dict = {}
	# Metrics
	done_window = deque(maxlen=args.window_size) # Env dones over last window_size episodes
	done_agents_window = deque(maxlen=args.window_size) # Fraction of done agents over last ...
	reward_window = deque(maxlen=args.window_size) # Cumulative rewards over last window_size episodes
	norm_reward_window = deque(maxlen=args.window_size) # Normalized cum. rewards over last window_size episodes
	# Track means over windows of window_size episodes
	mean_dones = [] 
	mean_agent_dones = []
	mean_rewards = []
	mean_norm_rewards = []
	# Episode rewards/dones/norm rewards since beginning of training TODO
	#env_dones = []
	
	update_values = [False] * args.num_agents
	buffer_obs = [None] * args.num_agents
	next_obs = [None] * args.num_agents

	############ Main loop
	for ep in range(args.num_episodes):
		cumulative_reward = 0
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
				agent = env.agents[a]

				# TODO evaluate those only once
				agent_speed = agent.speed_data["speed"]
				times_per_cell = int(np.reciprocal(agent_speed))

				# If two first consecutive bits in the bitmap are the same
				# Handle if train is arrived
				if agent.status == RailAgentStatus.DONE:
					# TODO? can you improve this? do i need this?
					obs = preprocess_obs(a, maps[a], maps, max_rails)
					buffer_obs[a] = obs.copy()
					update_values[a] = False
					
					# TODO? TEMP
					rail = np.argmax(np.absolute(maps[a, :, 0]))
					assert abs(maps[a, rail, 0]) == 1

				elif agent.status == RailAgentStatus.DONE_REMOVED:
					# TODO? can you improve this? do i need this?
					obs = preprocess_obs(a, maps[a], maps, max_rails)
					buffer_obs[a] = obs.copy()
					update_values[a] = False

					network_action = 0
					action = RailEnvActions.DO_NOTHING
					assert(not info['action_required'][a])

					# TODO? TEMP
					maps[a, :, 0] = 0 # TODO? this should be done only once
					rail = np.argmax(np.absolute(maps[a, :, 0]))
					assert abs(maps[a, rail, 0]) == 0

				elif not info['action_required'][a]:
					obs = preprocess_obs(a, maps[a], maps, max_rails)
					buffer_obs[a] = obs.copy()
					update_values[a] = False # Network doesn't need to choose a move and I don't store the experience

					network_action = 1
					action = RailEnvActions.DO_NOTHING
					maps = obs_builder.unroll_bitmap(a, maps)

				else:
					if not obs_builder.should_generate_altmaps(a):
						obs = preprocess_obs(a, maps[a], maps, max_rails)
						buffer_obs[a] = obs.copy()
						update_values[a] = False # Network doesn't need to choose a move and I don't store the experience

						# TODO? is this useful?
						# if obs_builder.next_cell_occupied(a):
						# 	network_action = 0
						# 	action = RailEnvActions.STOP_MOVING
						# else:
						network_action = 1
						action = obs_builder.get_agent_action(a)
						maps = obs_builder.unroll_bitmap(a, maps)

					else: # Changing rails - need to perform a move
						altmaps, altpaths = obs_builder.get_altmaps(a)

						# if len(altmaps) > 1: # TODO? is this useful? (1/2)
						q_values = np.array([])
						altobs = []
						for i in range(len(altmaps)):
							obs = preprocess_obs(a, altmaps[i], maps, max_rails)
							altobs.append(obs)
							q_values = np.concatenate([q_values, dqn.act(obs).cpu().data.numpy()])

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
						obs_builder.paths[a] = altpaths[best_i]
						obs = altobs[best_i]

						# else: # Continue on the same path # TODO? is this useful? (2/2)
						# 	obs = preprocess_obs(a, maps[a], maps, max_rails)
						# 	q_values = dqn.act(obs).cpu().data.numpy() # Network chooses action
						# 	if np.random.random() > eps:
						# 		network_action = np.argmax(q_values)
						# 	else:
						# 		network_action = np.random.choice([0, 1])	

						update_values[a] = True
						# Save current state in buffer
						buffer_obs[a] = obs.copy()
						# Update bitmaps and get new action
						# TODO? detect crash function
						# TODO? get_action function
						action, maps, crash = obs_builder.update_bitmaps(a, network_action, maps)
################################################
						if args.train and crash: # TODO Non sicura del livello di indentazione
							network_action = 0 # TODO! are you sure?
							# print('ADDING CRASH TUPLE')
							dqn.step(buffer_obs[a], 1, -2000, buffer_obs[a], True)
					if args.train:
						next_obs[a] = preprocess_obs(a, maps[a], maps, max_rails)
################################################
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
				cumulative_reward += reward[a] # / env.get_num_agents() # Update cumulative reward (not norm)
				
				
			if done['__all__']: # TODO!: env sets done[all] = True for everyone when time limit is reached
				env_done = 1
				break

		################### End of the episode
		eps = max(args.end_eps, args.eps_decay * eps)  # Decrease epsilon
		# Metrics
		done_window.append(env_done) # Save done in this episode
		
		num_agents_done = 0  # Num of agents that reached their target in the last episode
		for a in range(env.get_num_agents()): 
			if done[a]:
				num_agents_done += 1
		done_agents_window.append(num_agents_done / env.get_num_agents())
		reward_window.append(cumulative_reward)  # Save cumulative reward in this episode
		normalized_reward = cumulative_reward / (env.compute_max_episode_steps(env.width, env.height) + env.get_num_agents())
		norm_reward_window.append(normalized_reward)
		
		mean_dones.append((np.mean(done_window)))
		mean_agent_dones.append((np.mean(done_agents_window)))
		mean_rewards.append(np.mean(reward_window))
		mean_norm_rewards.append(np.mean(norm_reward_window))

		# Print training results info
		print(
			'\r{} Agents on ({},{}). Episode: {}\t Mean done agents: {:.2f}\t Mean reward: {:.2f}\t Mean normalized reward: {:.2f}\t Done agents in last episode: {:.2f}%\t Epsilon: {:.2f}'.format(
				env.get_num_agents(), args.width, args.height,
				ep,
				mean_agent_dones[-1],  # Fraction of done agents
				mean_rewards[-1],
				mean_norm_rewards[-1],
				(num_agents_done / args.num_agents),
				eps), end=" ")

		if ep != 0 and (ep + 1) % args.checkpoint_interval == 0:
			print(
				'\r{} Agents on ({},{}). Episode: {}\t Mean done agents: {:.2f}\t Mean reward: {:.2f}\t Mean normalized reward: {:.2f}\t Epsilon: {:.2f}'.format(
					env.get_num_agents(), args.width, args.height,
					ep,
					mean_agent_dones[-1],
					mean_rewards[-1],
					mean_norm_rewards[-1],
					eps))
			
			# Save model and metrics
			if args.train: # TODO! Now means are on moving-window (could also be done differently)
				torch.save(dqn.qnetwork_local.state_dict(), results_dir + '/model.pth')
				plot_metric(list(range(ep+1)), mean_dones, 'agents_done', path=results_dir)
				plot_metric(list(range(ep+1)), mean_rewards, 'reward', path=results_dir)
				plot_metric(list(range(ep+1)), mean_norm_rewards, 'norm_reward', path=results_dir)


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
	parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help='Class used to predict agent paths and help observation building')
	parser.add_argument('--prediction-depth', type=int, default=500, help='Prediction depth for shortest path strategy, i.e. length of a path')
	
	# Training
	parser.add_argument('--model-id', type=str, default="ddqn-example", help="Model name/id")
	parser.add_argument('--num-episodes', type=int, default=15000, help="Number of episodes to run")
	parser.add_argument('--start-eps', type=float, default=1.0, help="Initial value of epsilon")
	parser.add_argument('--end-eps', type=float, default=0.005, help="Lower limit of epsilon (i.e. can't decrease more)")
	parser.add_argument('--eps-decay', type=float, default=0.998, help="Factor to decrease eps in eps-greedy")
	parser.add_argument('--buffer-size', type=int, default=100000, help='Size of experience replay buffer (i.e. number of tuples')
	parser.add_argument('--batch-size', type=int, default=512, help='Size of mini-batch for replay buffer')
	parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
	parser.add_argument('--tau', type=float, default=1e-3, help='Interpolation parameter for soft update of target network weights')
	parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate for SGD')
	parser.add_argument('--update-every', type=int, default=10, help='How often to update the target network')
	
	# Misc
	parser.add_argument('--debug', action='store_true', help='Print debug info')
	parser.add_argument('--render', action='store_true', help='Render map')
	parser.add_argument('--train', action='store_true', help='Perform training')
	parser.add_argument('--window-size', type=int, default=100, help='Number of episodes to consider for moving average when evaluating model learning curve')
	parser.add_argument('--checkpoint-interval', type=int, default=50, help='Interval of episodes for each save of plots and model')
	parser.add_argument('--print', action='store_true', help='Save internal representations as files')
	
	args = parser.parse_args()
	main(args)