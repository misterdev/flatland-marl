from flatland.envs.observations import GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from fc_graphobs.graph_observations import GraphObsForRailEnv
from fc_graphobs.predictions import ShortestPathPredictorForRailEnv


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


width = 40
height = 40
nr_trains = 3  # Number of trains that have an assigned task in the env
cities_in_map = 2  # Number of cities where agents can start or end
seed = 1  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 3  # Max number of parallel tracks within a city, representing a realistic train station


rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )
# Maps speeds to % of appearance in the env TODO Find reasonable values
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

observation_builder = GraphObsForRailEnv(bfs_depth=4, predictor=ShortestPathPredictorForRailEnv(max_depth=10))

# Construct the environment with the given observation, generators, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              stochastic_data=stochastic_data,  # Malfunction data generator
              obs_builder_object=observation_builder,
              remove_agents_at_target=True  # Removes agents at the end of their journey to make space for others
              )
env.reset()

# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=True,
                          screen_height=1080,  # Adjust these parameters to fit your resolution
                          screen_width=1920)  # Adjust these parameters to fit your resolution

controller = RandomAgent(218, env.action_space[0])
action_dict = dict()

# Chose an action for each agent
for a in range(env.get_num_agents()):
    action = controller.act(0)
    action_dict.update({a: action})
# Do the environment step
observations, rewards, dones, information = env.step(action_dict)
print("\n The following agents can register an action:")
print("========================================")
for info in information['action_required']:
    print("Agent {} needs to submit an action.".format(info))
env_renderer.reset()

score = 0
# Run episode
frame_step = 0

for step in range(50):
    # Chose an action for each agent in the environment
    for a in range(env.get_num_agents()):
        action = controller.act(observations[a])  # always returns 2
        action_dict.update({a: action})

    # Environment step which returns the observations for all agents, their corresponding
    # reward and whether their are done

    next_obs, all_rewards, done, _ = env.step(action_dict)

    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    frame_step += 1
    # Update replay buffer and train agent
    for a in range(env.get_num_agents()):
        controller.step((observations[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
        score += all_rewards[a]

    observations = next_obs.copy()
    if done['__all__']:
        break

    print('Episode: Steps {}\t Score = {}'.format(step, score))
