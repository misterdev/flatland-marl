'''
Implement something very similar to the LocalObs available in Round 1

'''

import collections
from typing import Optional, List, Dict, Tuple

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet


class LocalObsForRailEnv(ObservationBuilder):
    """
    Gives a local observation of the rail environment around the agent.
    The observation is composed of the following elements:

    - transition map array with dimensions (view_height, view_width, 16),\
    assuming 16 bits encoding of transitions (one-hot encoding)

    - obs_agents_state: np.array of shape (view_height, view_width, 5) with
        - first channel containing the agent position and direction (int on grid)
        - second channel containing the other agents positions and direction (int on grid)
        - third channel containing agent/other agent malfunctions
        - fourth channel containing agent/other agent fractional speeds (float)
        - fifth channel containing number of other agents ready to depart (int on grid in correspondence to starting position)

    - obs_targets: np.array of shape (view_height, view_width, 2) containing respectively the position of the given agent\
     target/subtarget and the positions of the other agents targets/subtargets (flag only, no counter!). one-hot encoding.
    Subtargets are computed according to shortest path algorithm.

    Use the parameters view_width and view_height to define the rectangular view of the agent.
    The center parameters moves the agent along the height axis of this rectangle. If it is 0 the agent only has
    observation in front of it.
    """
    def __init__(self, view_width, view_height, center):

        super(LocalObsForRailEnv, self).__init__()
        self.view_width = view_width
        self.view_height = view_height
        self.center = center
        self.max_padding = max(self.view_width, self.view_height - self.center)
        self.rail_obs = np.zeros((self.view_height, self.view_width, 16))

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        # We build the transition map with a view_radius empty cells expansion on each side.
        # This helps to collect the local transition map view when the agent is close to a border.
        for i in range(self.view_height):
            for j in range(self.view_width):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # Get rail_obs/transition_map
        
        # Build agents obs
        # Collect agent position and direction 
        
        # Collect other agents position and direction
        
        # Collect data about malfunctions
        
        # Collect speed/priority
        
        # Collect data about agent ready to depart
        # Build targets obs
        # Collect position of agent target
        
        # Collect positions of other agents targets
        pass

    def _field_of_view(self, position, direction, state=None):
        # Field of view
        pass
