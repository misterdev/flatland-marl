import torch
import sys
import time

# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator, rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.malfunction_generators import malfunction_from_params

# make sure the root path is in system path
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from importlib_resources import path
from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv
from src.dueling_double_dqn_mod import Agent
from src.print_info import print_info
import src.nets

from configobj import ConfigObj

config = ConfigObj("./tests-config.ini")
tests = config.sections
n_tests = len(tests)

seed = 5  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
init_render_env = True # True if RenderEnv must be initialized

# rail_generator = rail_from_file("../test-envs/Test_0/Level_0.pkl")

# Maps speeds to % of appearance in the env
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

schedule_generator = sparse_schedule_generator(speed_ration_map)

prediction_depth = 40
observation_builder = GraphObsForRailEnv(bfs_depth=4, predictor=ShortestPathPredictorForRailEnv(max_depth=prediction_depth))

state_size = prediction_depth + 3
network_action_size = 2
controller = Agent(state_size, network_action_size)
network_action_dict = dict()
railenv_action_dict = dict()


# Here you can pre-load an agent
with path(src.nets, "avoid_checkpoint300.pth") as file_in:
    controller.qnetwork_local.load_state_dict(torch.load(file_in))
# TODO fix, doesn't work
for test in tests:
    score = 0
    # Build the env according to config parameters
    env = RailEnv(width=config[test].as_int('width'),
                  height=config[test].as_int('height'),
                  rail_generator=sparse_rail_generator(
                            max_num_cities=config[test].as_int('max_num_cities'), 
                            seed=seed,
                            grid_mode=grid_distribution_of_cities,
                            max_rails_between_cities=config[test].as_int('max_rails_between_cities'),
                            max_rails_in_city=config[test].as_int('max_rail_in_city')
                  ),
                  schedule_generator=schedule_generator,
                  number_of_agents=config[test].as_int('num_agents'),
                  obs_builder_object=observation_builder,
                  malfunction_generator_and_process_data=malfunction_from_params(
                      parameters={
                        'malfunction_rate': config[test].as_int('malfunction_rate'),  # Rate of malfunction occurrence of single agent
                        'min_duration': config[test].as_int('min_duration'),  # Minimal duration of malfunction
                        'max_duration': config[test].as_int('max_duration')  # Max duration of malfunction
                        }),
                  remove_agents_at_target=True)
    
    observations, infos = env.reset(True, True)

    # Initiate the renderer
    env_renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              screen_height=1080,
                              screen_width=1920)

    env_renderer.reset()
    # TODO Change 'max_num_cities' to 'num_cities' (effective) when this info will be available
    # max_time_steps = int(4 * 2 * (config[test].as_int('width') + config[test].as_int('height') + config[test].as_int('num_agents') / config[test].as_int('max_num_cities')))
    max_time_steps = int(3 * (config[test].as_int('width') + config[test].as_int('height')))
    
    # Pick first action
    for a in range(env.get_num_agents()):
        # Agent performs action only if required
        # 'railenv_action' is in [0, 4], network_action' is in [0, 1]
        network_action = controller.act(observations[a])
        railenv_action = observation_builder.choose_railenv_action(a, network_action)
        railenv_action_dict.update({a: railenv_action})
        network_action_dict.update({a: network_action})

    # Environment step
    next_obs, all_rewards, done, infos = env.step(railenv_action_dict)

    for step in range(max_time_steps - 1):
    
        # Logging
        # print_info(env)
        print('\rTest: {}\t Step / MaxSteps: {} / {}'.format(
            test,
            step+1,
            max_time_steps
        ), end=" ")
        
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            if infos['action_required'][a]:
                # Agent performs action only if required
                # 'railenv_action' is in [0, 4], network_action' is in [0, 1]
                network_action = controller.act(observations[a])
                railenv_action = observation_builder.choose_railenv_action(a, network_action)
            else:
                network_action = 0
                railenv_action = 0  # DO NOTHING
                
            railenv_action_dict.update({a: railenv_action})
            network_action_dict.update({a: network_action})
    
        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(railenv_action_dict)
    
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        for a in range(env.get_num_agents()):
            score += all_rewards[a]
    
        observations = next_obs.copy()
        if done['__all__']:
            break
            
    env_renderer.close_window()
    print('\nTest: {}\t Score = {}'.format(test, score))
