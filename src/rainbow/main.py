# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2  # Compression library
from datetime import datetime
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
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

from src.rainbow.agent import RainbowAgent
from src.rainbow.memory import ReplayMemory
from src.rainbow.test import test

def main(args):
	# Show options and values
	print(' ' * 26 + 'Options')
	for k, v in vars(args).items():
		print(' ' * 26 + k + ': ' + str(v))
	# Where to save models
	results_dir = os.path.join('results', args.id)
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	metrics = {'steps': [],
			   'rewards': [],
			   'Qs': [],
			   'best_avg_done_agents': -float('inf'),
			   'best_avg_reward': -float('inf')}
	np.random.seed(args.seed)
	torch.manual_seed(np.random.randint(1, 10000))
	# Set cpu or gpu
	if torch.cuda.is_available() and not args.disable_cuda:
		args.device = torch.device('cuda')
		torch.cuda.manual_seed(np.random.randint(1, 10000))
		torch.backends.cudnn.enabled = args.enable_cudnn
	else:
		args.device = torch.device('cpu')
	
	
	# Simple ISO 8601 timestamped logger
	def log(s):
		print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
	
	
	def load_memory(memory_path, disable_bzip):
		if disable_bzip:
			with open(memory_path, 'rb') as pickle_file:
				return pickle.load(pickle_file)
		else:
			with bz2.open(memory_path, 'rb') as zipped_pickle_file:
				return pickle.load(zipped_pickle_file)
	
	
	def save_memory(memory, memory_path, disable_bzip):
		if disable_bzip:
			with open(memory_path, 'wb') as pickle_file:
				pickle.dump(memory, pickle_file)
		else:
			with bz2.open(memory_path, 'wb') as zipped_pickle_file:
				pickle.dump(memory, zipped_pickle_file)
	
	
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
	
	stochastic_data = {'prop_malfunction': 0.3,  # Percentage of defective agents
					   'malfunction_rate': 30,  # Rate of malfunction occurrence
					   'min_duration': 3,  # Minimal duration of malfunction
					   'max_duration': 20  # Max duration of malfunction
					   }
	
	observation_builder = GraphObsForRailEnv(bfs_depth=args.bfs_depth, predictor=ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth))
	
	# Construct the environment with the given observation, generators, predictors, and stochastic data
	env = RailEnv(width=args.width,
				  height=args.height,
				  rail_generator=rail_generator,
				  schedule_generator=schedule_generator,
				  number_of_agents=args.num_agents,
				  obs_builder_object=observation_builder,
				  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
				  remove_agents_at_target=True  # Removes agents at the end of their journey to make space for others
				  )
	env.reset()
	
	state_size = args.prediction_depth * 2 + 4 # TODO
	action_space = args.network_action_space
	network_action_dict = {}
	railenv_action_dict = {}
	# Init agent
	dqn = RainbowAgent(args, state_size, env)
	
	# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
	if args.model is not None and not args.evaluate:
		if not args.memory:
			raise ValueError('Cannot resume training without memory save path. Aborting...')
		elif not os.path.exists(args.memory):
			raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))
	
		mem = load_memory(args.memory, args.disable_bzip_memory)
	else:
		# Init one replay buffer for each agent (TODO)
		mems = [ReplayMemory(args, int(args.memory_capacity/args.num_agents)) for a in range(args.num_agents)]
		# mem = ReplayMemory(args, args.memory_capacity)  # Init empty replay buffer
	
	priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
	
	# Construct validation memory
	val_mem = ReplayMemory(args, args.evaluation_size)
	T = 0
	all_done = True
	update_values = [False] * env.get_num_agents() # Used to update agent if action was performed in this step
	
	# Number of transitions to do for validating Q
	while T < args.evaluation_size:
		
		for a in range(env.get_num_agents()):
			if all_done:
				state, info = env.reset()
				all_done = False
		
		for a in range(env.get_num_agents()):
			action = np.random.choice(np.arange(5))
			railenv_action_dict.update({a: action})
			
		next_state, rewards, done, info = env.step(railenv_action_dict)
		val_mem.append(state[0], None, None, all_done) # TODO Using only state from agent 0 for now
		all_done = done['__all__']
		state = next_state
		T += 1
	
	
	if args.evaluate:
		dqn.eval() # Set DQN (online network) to evaluation mode
		avg_done_agents, avg_reward, avg_norm_reward = test(args, 0, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
		#print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
		print('Avg. done agents: ' + str(avg_done_agents) + ' | Avg. cumulative reward: ' + str(avg_reward) + 
			  ' | Avg. normalized reward: ' + str(avg_norm_reward))
	else:
		# Training loop
		dqn.train()
		################## Episodes loop #######################
		for ep in range(1, args.num_episodes + 1):
			# Reset env at the beginning of one episode
			state, info = env.reset()
	
			# Pick first action # TODO Decide entering of agents, now random
			for a in range(env.get_num_agents()):
				action = np.random.choice((0,2))
				railenv_action_dict.update({a: action})
			next_state, reward, done, info = env.step(railenv_action_dict)  # Env first step
	
			############## Steps loop ##########################
			for T in trange(1, args.T_max + 1):
				if T % args.replay_frequency == 0:
					dqn.reset_noise()  # Draw a new set of noisy weights
	
				for a in range(env.get_num_agents()):
					if info['action_required'][a]:
						network_action = dqn.act(state[a])  # Choose an action greedily (with noisy weights)
						railenv_action = observation_builder.choose_railenv_action(a, network_action)
						update_values[a] = True
					else:
						network_action = 0
						railenv_action = 0
						update_values[a] = False
					# Update action dicts
					railenv_action_dict.update({a: railenv_action})
					network_action_dict.update({a: network_action})
	
				next_state, reward, done, info = env.step(railenv_action_dict)  # Env step
				
				if args.debug:
					for a in range(env.get_num_agents()):
						print('#########################################')
						print('Info for agent {}'.format(a))
						print('Obs: {}'.format(state[a]))
						print('Status: {}'.format(info['status'][a]))
						print('Moving? {} at speed: {}'.format(env.agents[a].moving, info['speed'][a]))
						print('Action required? {}'.format(info['action_required'][a]))
						print('Network action: {}'.format(network_action_dict[a]))
						print('Railenv action: {}'.format(railenv_action_dict[a]))
					
				# Clip reward and update replay buffer
				for a in range(env.get_num_agents()):
					if args.reward_clip > 0:
						reward[a] = max(min(reward[a], args.reward_clip), -args.reward_clip)
					if update_values[a]:  # Store transition only if this agent performed action in this time step
						mems[a].append(state[a], network_action_dict[a], reward[a], done[a]) # Append to own buffer
						#mem.append(state[a], network_action_dict[a], reward[a], done[a])  # Append transition to memory
				
	
				state = next_state.copy()
				# Train and test
				if ep >= args.learn_start:
					# Anneal importance sampling weight β to 1
					#mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
					for a in range(args.num_agents):
						mems[a].priority_weight = min(mems[a].priority_weight + priority_weight_increase, 1)
	
					if T % args.replay_frequency == 0: # TODO Should learn from info derived from all the replay buffers
						a = np.random.choice(np.arange(args.num_agents))
						dqn.learn(mems[a]) # Learn randomly from one of the available replay buffer
						# dqn.learn(mem)  # Train with n-step distributional double-Q learning
	
					if (ep % args.evaluation_interval) == 0 and (T == args.T_max):  # Eval only at the last step of an episode
	
						dqn.eval()  # Set DQN (online network) to evaluation mode
						avg_done_agents, avg_reward, avg_norm_reward = test(args, T, ep, dqn, val_mem, metrics, results_dir)  # Test
						log(
							'T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. done agents: ' + str(avg_done_agents) +
							' | Avg. reward: ' + str(avg_reward) + ' | Avg. normalized reward: ' + str(avg_norm_reward))
						dqn.train()  # Set DQN (online network) back to training mode
	
						# If memory path provided, save it
	
						if args.memory is not None:
							save_memory(mems[0], args.memory, args.disable_bzip_memory) # Save only first replay buffer (?)
							#save_memory(mem, args.memory, args.disable_bzip_memory)
	
					# Update target network
					if T % args.target_update == 0:
						dqn.update_target_net()
	
				if done['__all__']:
					break
			# Checkpoint the network every 'checkpoint_interval' episodes
			if (args.checkpoint_interval != 0) and (ep % args.checkpoint_interval == 0):
				dqn.save(results_dir, 'checkpoint.pth')
					
if __name__ == '__main__':

	# Hyperparameters
	parser = argparse.ArgumentParser(description='Rainbow')
	parser.add_argument('--id', type=str, default='default', help='Experiment ID')
	parser.add_argument('--seed', type=int, default=123, help='Random seed')
	parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
	parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
	parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
	parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
	parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
	parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
	parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
	parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
	parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
	parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
	parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
	parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
	parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
	parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
	parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
	parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
	parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
	parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
	parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
	parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
	parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
	parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
	parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
	parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
	parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
	# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
	parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
	parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
	parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
	parser.add_argument('--checkpoint-interval', type=int, default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
	parser.add_argument('--memory', help='Path to save/load the memory from')
	parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
	parser.add_argument('--debug', action='store_true', help='Print more info during execution')
	
	# Env parameters
	# parser.add_argument('--state_size', type=int, help='Size of state to feed to the neural network') # Depends on prediction_depth
	parser.add_argument('--network-action-space', type=int, default=2, help='Number of actions allowed in the environment')
	parser.add_argument('--width', type=int, default=100, help='Environment width')
	parser.add_argument('--height', type=int, default=100, help='Environment height')
	parser.add_argument('--num-agents', type=int, default=50, help='Number of agents in the environment')
	parser.add_argument('--max-num-cities', type=int, default=6, help='Maximum number of cities where agents can start or end')
	# parser.add_argument('--seed', type=int, default=1, help='Seed used to generate grid environment randomly')
	parser.add_argument('--grid-mode', type=bool, default=False, help='Type of city distribution, if False cities are randomly placed')
	parser.add_argument('--max-rails-between-cities', type=int, default=4, help='Max number of tracks allowed between cities, these count as entry points to a city')
	parser.add_argument('--max-rails-in-city', type=int, default=6, help='Max number of parallel tracks within a city allowed')
	parser.add_argument('--malfunction-rate', type=int, default=1000, help='Rate of malfunction occurrence of single agent')
	parser.add_argument('--min-duration', type=int, default=20, help='Min duration of malfunction')
	parser.add_argument('--max-duration', type=int, default=50, help='Max duration of malfunction')
	parser.add_argument('--observation-builder', type=str, default='GraphObsForRailEnv', help='Class to use to build observation for agent')
	parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help='Class used to predict agent paths and help observation building')
	parser.add_argument('--bfs-depth', type=int, default=4, help='BFS depth of the graph observation')
	parser.add_argument('--prediction-depth', type=int, default=40, help='Prediction depth for shortest path strategy, i.e. length of a path')
	parser.add_argument('--view-semiwidth', type=int, default=7, help='Semiwidth of field view for agent in local obs')
	parser.add_argument('--view-height', type=int, default=30, help='Height of the field view for agent in local obs')
	parser.add_argument('--offset', type=int, default=25, help='Offset of agent in local obs')
	# Training parameters
	parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes on which to train the agents')

	# Setup
	args = parser.parse_args()
	# Check arguments
	if args.offset > args.height:
		raise ValueError("Agent offset can't be greater than view height in local obs")
	if args.offset < 0:
		raise ValueError("Agent offset must be a positive integer")
	
	main(args)