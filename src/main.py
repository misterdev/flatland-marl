# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
import os

from collections import deque

from torch.utils.tensorboard import SummaryWriter

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
from src.preprocessing import ObsPreprocessor
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
	speed_ration_map = {1.: 1}  # Fast passenger train
	
	if args.multi_speed:
		speed_ration_map = {1.: 0.25,  # Fast passenger train
							1. / 2.: 0.25,  # Fast freight train
							1. / 3.: 0.25,  # Slow commuter train
							1. / 4.: 0.25}  # Slow freight train

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

	if args.plot:
		writer = SummaryWriter(log_dir='runs/' + args.model_id)

	max_rails = 100 # TODO Must be a parameter of the env (estimated)
	# max_steps = env.compute_max_episode_steps(env.width, env.height)
	max_steps = 200

	preprocessor = ObsPreprocessor(max_rails, args.reorder_rails)

	dqn = DQNAgent(args, bitmap_height=max_rails * 3, action_space=2)

	if args.load_path:
		file = os.path.isfile(args.load_path)
		if file:
			dqn.qnetwork_local.load_state_dict(torch.load(args.load_path))
			print('WEIGHTS LOADED from: ', args.load_path)

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
	
	crash = [False] * args.num_agents
	update_values = [False] * args.num_agents
	buffer_obs = [[]] * args.num_agents

	############ Main loop
	for ep in range(args.num_episodes):
		cumulative_reward = 0
		env_done = 0
		altmaps = [None] * args.num_agents
		altpaths = [[]] * args.num_agents
		buffer_rew = [0] * args.num_agents
		buffer_done = [False] * args.num_agents
		curr_obs = [None] * args.num_agents

		maps, info = env.reset()
		if args.print:
			debug.print_bitmaps(maps)

		if args.render:
			env_renderer.reset()

		for step in range(max_steps - 1):
			# Save a copy of maps at the beginning
			buffer_maps = maps.copy()
			# rem first bit is 0 for agent not departed
			for a in range(env.get_num_agents()):
				agent = env.agents[a]
				crash[a] = False
				update_values[a] = False
				network_action = None
				action = None

				# If agent is arrived
				if agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
					# TODO if agent !removed you should leave a bit in the bitmap
					# TODO? set bitmap only the first time
					maps[a, :, :] = 0
					network_action = 0
					action = RailEnvActions.DO_NOTHING

				# If agent is not departed
				elif agent.status == RailAgentStatus.READY_TO_DEPART:
					update_values[a] = True
					obs = preprocessor.get_obs(a, maps[a], buffer_maps)
					curr_obs[a] = obs.copy()

					# Network chooses action
					q_values = dqn.act(obs).cpu().data.numpy()
					if np.random.random() > eps:
						network_action = np.argmax(q_values)
					else:
						network_action = np.random.choice([0, 1])

					if network_action == 0:
						action = RailEnvActions.DO_NOTHING
					else: # Go
						crash[a] = obs_builder.check_crash(a, maps)
						
						if crash[a]:
							network_action = 0
							action = RailEnvActions.STOP_MOVING
						else:
							maps = obs_builder.update_bitmaps(a, maps)
							action = obs_builder.get_agent_action(a)
		
				# If the agent is entering a switch
				elif obs_builder.is_before_switch(a) and info['action_required'][a]:
					# If the altpaths cache is empty or already contains
					# the altpaths from the current agent's position
					if len(altpaths[a]) == 0 or agent.position != altpaths[a][0][0].position:
						altmaps[a], altpaths[a] = obs_builder.get_altmaps(a)

					if len(altmaps[a]) > 0:
						update_values[a] = True
						altobs = [None] * len(altmaps[a])
						q_values = np.array([])
						for i in range(len(altmaps[a])):
							altobs[i] = preprocessor.get_obs(a, altmaps[a][i], buffer_maps)
							q_values = np.concatenate([q_values, dqn.act(altobs[i]).cpu().data.numpy()])

						# Epsilon-greedy action selection
						if np.random.random() > eps:
							argmax = np.argmax(q_values)
							network_action = argmax % 2
							best_i = argmax // 2
						else:
							network_action = np.random.choice([0, 1])
							best_i = np.random.choice(np.arange(len(altmaps[a])))
						
						# Use new bitmaps and paths
						maps[a, :, :] = altmaps[a][best_i]
						obs_builder.set_agent_path(a, altpaths[a][best_i])
						curr_obs[a] = altobs[best_i].copy()

					else:
						print('[ERROR] NO ALTHPATHS EP: {} STEP: {} AGENT: {}'.format(ep, step, a))
						network_action = 0

					if network_action == 0:
						action = RailEnvActions.STOP_MOVING
					else:
						crash[a] = obs_builder.check_crash(a, maps, is_before_switch=True)
						
						if crash[a]:
							network_action = 0
							action = RailEnvActions.STOP_MOVING
						else:
							action = obs_builder.get_agent_action(a)
							maps = obs_builder.update_bitmaps(a, maps, is_before_switch=True)
		
				# If the agent is following a rail
				elif info['action_required'][a]:
					crash[a] = obs_builder.check_crash(a, maps)

					if crash[a]:
						network_action = 0
						action = RailEnvActions.STOP_MOVING
					else:
						network_action = 1
						action = obs_builder.get_agent_action(a)
						maps = obs_builder.update_bitmaps(a, maps)

				else: # not action_required
					network_action = 1
					action = RailEnvActions.DO_NOTHING
					maps = obs_builder.update_bitmaps(a, maps)
				
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
			# Update replay buffer and train agent
			if args.train:
				for a in range(env.get_num_agents()):
					if args.crash_penalty and crash[a]:
						# Store bad experience
						dqn.step(curr_obs[a], 1, -100, curr_obs[a], True)

					if not args.switch2switch:
						if update_values[a] and not buffer_done[a]:
							next_obs = preprocessor.get_obs(a, maps[a], maps)
							dqn.step(curr_obs[a], network_action_dict[a], reward[a], next_obs, done[a])

					else:
						if update_values[a] and not buffer_done[a]:
							# If I had an obs from a previous switch
							if len(buffer_obs[a]) != 0:
								dqn.step(buffer_obs[a], 1, buffer_rew[a], curr_obs[a], done[a])
								buffer_obs[a] = []
								buffer_rew[a] = 0

							if network_action_dict[a] == 0:
								dqn.step(curr_obs[a], 1, reward[a], curr_obs[a], False)
							elif network_action_dict[a] == 1:
								# I store the obs and update at the next switch
								buffer_obs[a] = curr_obs[a].copy()

						# Cache reward only if we have an obs from a prev switch
						if len(buffer_obs[a]) != 0:
							buffer_rew[a] += reward[a]

					# Now update the done cache to avoid adding experience many times
					buffer_done[a] = done[a]

			for a in range(env.get_num_agents()):	
				cumulative_reward += reward[a] # / env.get_num_agents() # Update cumulative reward (not norm)
			 
			# TODO? env sets done[all] = True for everyone when time limit is reached
			# devid: I also remember this, but debuggind doesn't seem to happen
			if done['__all__']:
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

		if args.train and ep != 0 and (ep + 1) % args.save_interval == 0:
			torch.save(dqn.qnetwork_local.state_dict(), results_dir + '/weights.pt' )

		if args.plot:
			writer.add_scalar('mean_agent_dones', mean_agent_dones[-1], ep)
			writer.add_scalar('mean_rewards', mean_rewards[-1], ep)
			writer.add_scalar('mean_dones', mean_dones[-1], ep)
			writer.add_scalar('mean_norm_rewards', mean_norm_rewards[-1], ep)
			writer.add_scalar('epsilon', eps, ep)

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
	parser.add_argument('--prediction-depth', type=int, default=150, help='Prediction depth for shortest path strategy, i.e. length of a path')
	parser.add_argument('--multi-speed', action='store_true', help='Enable agents with fractionary speeds')
	
	# Training
	parser.add_argument('--model-id', type=str, default="ddqn-example", help="Model name/id")
	parser.add_argument('--num-episodes', type=int, default=10000, help="Number of episodes to run")
	parser.add_argument('--start-eps', type=float, default=1.0, help="Initial value of epsilon")
	parser.add_argument('--end-eps', type=float, default=0.005, help="Lower limit of epsilon (i.e. can't decrease more)")
	parser.add_argument('--eps-decay', type=float, default=0.998, help="Factor to decrease eps in eps-greedy")
	parser.add_argument('--buffer-size', type=int, default=10000, help='Size of experience replay buffer (i.e. number of tuples')
	parser.add_argument('--batch-size', type=int, default=512, help='Size of mini-batch for replay buffer')
	parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
	parser.add_argument('--tau', type=float, default=1e-3, help='Interpolation parameter for soft update of target network weights')
	parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate for SGD')
	parser.add_argument('--update-every', type=int, default=10, help='How often to update the target network')

	# Algo
	parser.add_argument('--crash-penalty', action='store_true', help='Add crash penalty for wrong actions')
	parser.add_argument('--reorder-rails', action='store_true', help='Change rails order in bitmaps')
	parser.add_argument('--switch2switch', action='store_true', help='Train using only bitmaps where the agent is before a switch')
	parser.add_argument('--train', action='store_true', help='Perform training')
	parser.add_argument('--load-path', type=str, default=None, help="Weights path")
	parser.add_argument('--save-interval', type=int, default=2500, help='Interval of episodes for each save of model')
	parser.add_argument('--checkpoint-interval', type=int, default=200, help='Interval of episodes for each print')

	# Misc
	parser.add_argument('--plot', action='store_true', help='Plot execution info')
	parser.add_argument('--profile', action='store_true', help='Print a profiling of where the program spent most of its time')
	parser.add_argument('--print', action='store_true', help='Save internal representations as files')
	parser.add_argument('--debug', action='store_true', help='Print debug info')
	parser.add_argument('--render', action='store_true', help='Render map')
	parser.add_argument('--window-size', type=int, default=100, help='Number of episodes to consider for moving average when evaluating model learning curve')
	
	args = parser.parse_args()

	if args.profile:
		import cProfile
		cProfile.run('main(args)')
	else:
		main(args)