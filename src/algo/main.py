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
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

from src.algo.graph.utils import map_to_graph


def main(args):
	
	'''
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

	stochastic_data = {'malfunction_rate': 30,  # Rate of malfunction occurrence
					   'min_duration': 3,  # Minimal duration of malfunction
					   'max_duration': 20  # Max duration of malfunction
					   }

	observation_builder = GraphObsForRailEnv(bfs_depth=args.bfs_depth,
											 predictor=ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth))

	# Construct the environment with the given observation, generators, predictors, and stochastic data
	env = RailEnv(width=args.width,
				  height=args.height,
				  rail_generator=rail_generator,
				  schedule_generator=schedule_generator,
				  number_of_agents=args.num_agents,
				  obs_builder_object=observation_builder,
				  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data)
				  )
	'''

	n_goals = args.num_agents + np.random.randint(0, 3)
	min_dist = int(0.75 * min(10, 10))
	rail_generator = complex_rail_generator(nr_start_goal=n_goals,
											nr_extra=2,
											min_dist=min_dist,
											max_dist=9999,
											seed=0)
	
	observation_builder = GraphObsForRailEnv(bfs_depth=args.bfs_depth,
											 predictor=ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth))

	env = RailEnv(width=args.width,
				  height=args.height,
				  rail_generator=rail_generator,
				  number_of_agents=args.num_agents,
				  obs_builder_object=observation_builder
				  )
	
	env.reset()
	
	env_renderer = RenderTool(
		env,
		agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
		show_debug=True,
		screen_height=1080,
		screen_width=1920)
	
	env_renderer.reset()
	railenv_action_dict = {}
	
	for step in range(10):
		for a in range(env.get_num_agents()):
			action = np.random.choice(np.arange(5))
			railenv_action_dict.update({a: action})
		state, reward, done, info = env.step(railenv_action_dict)  # Env step
		env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

		if step == 9: map_to_graph(env)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Algo')
	# Env parameters
	parser.add_argument('--network-action-space', type=int, default=2, help='Number of actions allowed in the environment')
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
	parser.add_argument('--prediction-depth', type=int, default=40, help='Prediction depth for shortest path strategy, i.e. length of a path')
	
	args = parser.parse_args()
	main(args)