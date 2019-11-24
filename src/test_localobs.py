import torch
import sys
import time
import pprint
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.malfunction_generators import malfunction_from_params

# make sure the root path is in system path
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from importlib_resources import path
from src.local_observations import LocalObsForRailEnv
from src.dueling_double_dqn import Agent
from src.utils import preprocess_obs
import src.nets

grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

schedule_generator = sparse_schedule_generator(speed_ration_map)

observation_builder = LocalObsForRailEnv(view_semiwidth=7, view_height=30, offset=25) # 7, 30, 25
#observation_builder = LocalObsForRailEnv(view_semiwidth=2, view_height=6, offset=4)

in_channels = state_size = 16 + 5 + 2
action_size = 5
controller = Agent(network_type='conv', state_size=state_size, action_size=action_size)
railenv_action_dict = dict()


with path(src.nets, "exp_local_obs400.pth") as file_in:
    controller.qnetwork_local.load_state_dict(torch.load(file_in))


score = 0
num_agents_done = 0
# Build the env according to config parameters
env = RailEnv(width=40,
              height=40,
              rail_generator=sparse_rail_generator(
                  max_num_cities=4,
                  seed=0,
                  grid_mode=grid_distribution_of_cities,
                  max_rails_between_cities=4,
                  max_rails_in_city=6
              ),
              schedule_generator=schedule_generator,
              number_of_agents=1,
              obs_builder_object=observation_builder,
              malfunction_generator_and_process_data=malfunction_from_params(
                  parameters={
                      'malfunction_rate': 1000,
                      'min_duration': 20,
                      'max_duration': 30
                  })
              )
# Initiate the renderer
env_renderer = RenderTool(env,
                          gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                          show_debug=True,
                          screen_height=1080,
                          screen_width=1920)
obs, info = env.reset()
env_renderer.reset()

# Preprocess and normalize obs
for a in range(env.get_num_agents()):
    if obs[a]:
        obs[a] = preprocess_obs(obs[a])

max_time_steps = env.compute_max_episode_steps(env.width, env.height)

for step in range(max_time_steps):
    
    print('\rStep / MaxSteps: {} / {}\n'.format(
        step+1,
        max_time_steps
    ), end=" ")
    
    '''
    for agent_idx, agent in enumerate(env.agents):
        print(
            "Agent {} ha state {} in (current) position {} with malfunction {}".format(
                agent_idx, str(agent.status), str(agent.position), str(agent.malfunction_data['malfunction'])))
    '''

    # Choose an action for each agent in the environment
    for a in range(env.get_num_agents()):
        if info['action_required'][a]:
            railenv_action = controller.act(obs[a])
        else:
            railenv_action = 0  # DO NOTHING

        railenv_action_dict.update({a: railenv_action})

    for a in range(1):
        print('#########################################')
        print('Info for agent {}'.format(a))
        print('State: {}'.format(obs[a]))
        print('Railenv action: {}'.format(railenv_action_dict[a]))

    next_obs, all_rewards, done, info = env.step(railenv_action_dict)
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False, selected_agent=0)
    for a in range(env.get_num_agents()):
        if next_obs[a]:
            next_obs[a] = preprocess_obs(next_obs[a])
            obs[a] = next_obs[a].copy()
        score += all_rewards[a]

    if done['__all__']:
        break

env_renderer.close_window()
# Compute num of agents that reached their target
for a in range(env.get_num_agents()):
    if done[a]:
        num_agents_done += 1

print(
    '\nAgent Dones: {}%\t Score: {}'.format(
        100 * (num_agents_done / env.get_num_agents()),
        score))

# TODO Render of obs and env step are not synced!
