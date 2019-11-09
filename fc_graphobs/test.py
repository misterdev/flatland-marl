import torch
import sys
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator, rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.malfunction_generators import malfunction_from_params

# make sure the root path is in system path
from pathlib import Path
from importlib_resources import path
from fc_graphobs.graph_observations import GraphObsForRailEnv
from fc_graphobs.predictions import ShortestPathPredictorForRailEnv
from fc_graphobs.dueling_double_dqn_mod import Agent
from fc_graphobs.print_info import print_info
import fc_graphobs.nets
import time

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        return 2

    def step(self, memories):
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return


width = 60
height = 60
nr_trains = 5  # Number of trains that have an assigned task in the env
cities_in_map = 4  # Number of cities where agents can start or end
seed = 2  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 3  # Max number of parallel tracks within a city, representing a realistic train station


rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )


# rail_generator = rail_from_file("../test-envs/Test_0/Level_0.pkl")

# Maps speeds to % of appearance in the env TODO Find reasonable values
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

schedule_generator = sparse_schedule_generator(speed_ration_map)

stochastic_data = {
    'malfunction_rate': 10000,  # Rate of malfunction occurrence of single agent
    'min_duration': 15,  # Minimal duration of malfunction
    'max_duration': 50  # Max duration of malfunction
    }

prediction_depth = 30
observation_builder = GraphObsForRailEnv(bfs_depth=4, predictor=ShortestPathPredictorForRailEnv(max_depth=prediction_depth))

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

# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=True,
                          screen_height=1080,  # Adjust these parameters to fit your resolution
                          screen_width=1920)  # Adjust these parameters to fit your resolution

state_size = prediction_depth + 3
network_action_size = 2
# controller = RandomAgent(218, env.action_space[0])
controller = Agent(state_size, network_action_size)
network_action_dict = dict()
railenv_action_dict = dict()

observations, infos = env.reset(True, True)
env_renderer.reset()

score = 0
# Run episode
frame_step = 0

# Here you can pre-load an agent
with path(fc_graphobs.nets, "avoid_checkpoint20.pth") as file_in:
    controller.qnetwork_local.load_state_dict(torch.load(file_in))

for step in range(200):

    print_info(env)
    # Chose an action for each agent in the environment
    for a in range(env.get_num_agents()):
        shortest_path_action = int(observation_builder.get_shortest_path_action(a))
        # 'railenv_action' is in [0, 4], network_action' is in [0, 1]
        # 'network_action' is None if act() returned a random sampled action
        railenv_action, network_action = controller.act(observations[a], shortest_path_action)
        railenv_action_dict.update({a: railenv_action})
        network_action_dict.update({a: network_action})

    # Environment step which returns the observations for all agents, their corresponding
    # reward and whether their are done

    next_obs, all_rewards, done, _ = env.step(railenv_action_dict)

    # Which agents needs to pick and action
    print("\n The following agents can register an action:")
    print("========================================")
    for info in infos['action_required']:
        print("Agent {} needs to submit an action.".format(info))

    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    frame_step += 1
    # Update replay buffer and train agent
    for a in range(env.get_num_agents()):
        controller.step(observations[a], network_action_dict[a], all_rewards[a], next_obs[a], done[a])
        score += all_rewards[a]

    observations = next_obs.copy()
    if done['__all__']:
        break

    print('Episode: Steps {}\t Score = {}'.format(step, score))
