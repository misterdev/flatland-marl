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

    - rail_obs, transition map array with dimensions (view_height, 2 * view_semiwidth + 1, 16),\
    assuming 16 bits encoding of transitions (one-hot encoding)

    - agents_state_obs: np.array of shape (view_height, 2 * view_semiwidth + 1, 5) with
        - first channel containing the agent position and direction (int on grid)
        - second channel containing the other agents positions and direction (int on grid)
        - third channel containing agent/other agent malfunctions
        - fourth channel containing agent/other agent fractional speeds (float)
        - fifth channel containing number of other agents ready to depart (int on grid in correspondence to starting position)

    - targets_obs: np.array of shape (view_height, 2 * view_semiwidth + 1, 2) containing respectively the position of the given agent\
     target/subtarget and the positions of the other agents targets/subtargets (flag only, no counter!). one-hot encoding.
    Subtargets are computed according to shortest path algorithm.

    Use the parameters view_semiwidth and view_height to define the rectangular view of the agent, where
    view_semiwidth defines the observable space at each 'side' of the agent.
    The base field view as a rectangle is defined with the agent facing north (direction = 0) where the origin lies
    at the upper-left corner (as in the Flatland grid env).
    The offset parameter moves the agent along the height axis of this rectangle, 0 <= offset <= view_height.
    If equal to view_height the agent only has observation in front of it, if equal to 0 the agent has only observation 
     behind.
    """
    def __init__(self, view_semiwidth, view_height, offset):

        super(LocalObsForRailEnv, self).__init__()
        self.view_semiwidth = view_semiwidth
        self.view_width = 2 * self.view_semiwidth + 1
        self.view_height = view_height
        self.offset = offset  # Agent offset along axis of the agent's direction
        self.rail_obs = np.zeros((self.view_width, self.view_height, 16))
        self.agents_state_obs = np.zeros((self.view_width, self.view_height, 5))
        self.targets_obs = np.zeros((self.view_width, self.view_height, 2))

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        
        pass
        # Compute from agent actual position
        '''
        for i in range(self.view_height):
            for j in range(self.view_width):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)
        '''

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        
        agent = self.env.agents[handle]
        
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:  # agent is DONE_REMOVED
            return None
            
        # Compute field of view
        visible_cells = self._field_of_view(agent_virtual_position, agent.direction)
        
        # Add the visited cells to the observed cells (for visualization)
        self.env.dev_obs_dict[handle] = set(visible_cells) # tipo (position[0], position[1], direction) TODO not sure if works like this
        
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
    '''
    Given agent current position and direction, returns the field of view of the agent as np.array of cells on the grid in
    absolute coordinates. Value is -np.inf to indicate padding (when agent lies on border).
    '''
    def _field_of_view(self, position, direction):
        
        # Compute visible cells
        visible_cells = np.full((self.view_width, self.view_height, 2), -np.inf)        

        if direction == 0:  # North
            origin = (position[0] - self.offset, position[1] - self.view_semiwidth)
        elif direction == 1:  # East
            origin = (position[0] - self.view_semiwidth, position[1] + self.offset)
        elif direction == 2:  # South
            origin = (position[0] + self.offset, position[1] + self.view_semiwidth)
        else:  # West
            origin = (position[0] + self.view_semiwidth, position[1] - self.offset)

        for i in range(self.view_width):
            for j in range(self.view_height):
                if direction == 0:
                    cell_to_add = (origin[0] + j, origin[1] + i)
                elif direction == 1:  # Rectangle is flipped 90°
                    cell_to_add = (origin[0] + i, origin[1] - j)
                elif direction == 2:
                    cell_to_add = (origin[0] - j, origin[1] - i)
                elif direction == 3:
                    cell_to_add = (origin[0] - i, origin[1] + j)

                if cell_to_add[0] >= self.env.height or cell_to_add[1] >= self.env.width or cell_to_add[0] < 0 or \
                        cell_to_add[1] < 0:
                    break
                visible_cells[i, j] = cell_to_add  # In absolute coordinates
                
        return visible_cells

