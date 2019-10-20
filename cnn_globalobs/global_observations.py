"""
Collection of my own ObservationBuilder.
"""

import collections
from typing import Optional, List, Dict, Tuple

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent

from cnn_globalobs.utils import convert_transitions_map


class CustomGlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - obs_rail: array with dimensions (env.height, env.width, 2) with
            - first channel containing the cell types in [0, 10]
            - second channel containing the cell rotation [0, 90, 180, 270]

        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
            - fifth channel containing number of other agents ready to depart

        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """

    def __init__(self):
        super(CustomGlobalObsForRailEnv, self).__init__()

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        rail_obs_16_channels = np.zeros((self.env.height, self.env.width, 16))
        for i in range(rail_obs_16_channels.shape[0]):
            for j in range(rail_obs_16_channels.shape[1]):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                rail_obs_16_channels[i, j] = np.array(bitlist)

        self.rail_obs = convert_transitions_map(rail_obs_16_channels)

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray):

        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        obs_targets = np.zeros((self.env.height, self.env.width, 2))
        obs_agents_state = np.zeros((self.env.height, self.env.width, 5)) - 1

        # TODO can we do this more elegantly?
        # for r in range(self.env.height):
        #     for c in range(self.env.width):
        #         obs_agents_state[(r, c)][4] = 0
        obs_agents_state[:,:,4] = 0

        obs_agents_state[agent_virtual_position][0] = agent.direction
        obs_targets[agent.target][0] = 1

        for i in range(len(self.env.agents)):
            other_agent: EnvAgent = self.env.agents[i]

            # ignore other agents not in the grid any more
            if other_agent.status == RailAgentStatus.DONE_REMOVED:
                continue

            obs_targets[other_agent.target][1] = 1

            # second to fourth channel only if in the grid
            if other_agent.position is not None:
                # second channel only for other agents
                if i != handle:
                    obs_agents_state[other_agent.position][1] = other_agent.direction
                obs_agents_state[other_agent.position][2] = other_agent.malfunction_data['malfunction']
                obs_agents_state[other_agent.position][3] = other_agent.speed_data['speed']
            # fifth channel: all ready to depart on this position
            if other_agent.status == RailAgentStatus.READY_TO_DEPART:
                obs_agents_state[other_agent.initial_position][4] += 1
        return self.rail_obs, obs_agents_state, obs_targets


def RGBObsForRailEnv(ObservationBuilder):
    """
    This obs contains 1 img RGB in a numpy array [x, y, 3]
    :param ObservationBuilder: 
    :return: 
    """
    def __init__(self):
        super(RGBObsForRailEnv, self).__init__()

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        pass

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray):
        pass

    pass
