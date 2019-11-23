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
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_shortest_paths import get_shortest_paths


class LocalObsForRailEnv(ObservationBuilder):
    """
    Gives a local observation of the rail environment around the agent.
    The observation is composed of the following elements:

    - local_rail_obs, transition map array with dimensions (view_height, 2 * view_semiwidth + 1, 16),\
    assuming 16 bits encoding of transitions (one-hot encoding)

    - agents_state_obs: np.array of shape (view_height, 2 * view_semiwidth + 1, 5) with
        - first channel containing the agent position and direction (int on grid)
        - second channel containing active agents positions and direction (int on grid)
        - third channel containing agent/other agent malfunctions (int, duration)
        - fourth channel containing agent/other agent fractional speeds (float)
        - fifth channel containing directions of agents ready to depart (flag in correspondence to initial position)

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
        self.rail_obs = None
        self.targets_obs = None
        self.shortest_paths = None

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        # Useful for precomputing stuff - at the beginning of an episode
        # Precompute rail_obs of ALL env - then compute local rail obs from this
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))  # Transition map of the whole env
        for i in range(self.env.height):
            for j in range(self.env.width):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)
        # Global targets - not subtargets
        self.targets_obs = np.zeros((self.view_height, self.view_width, 2))
        distance_map: DistanceMap = self.env.distance_map
        self.shortest_paths = get_shortest_paths(distance_map)  # TODO Must be computed in the get_many since at reset it doesn't fill values. but i don't want to compute it everytime


    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        
        :param handle: 
        :return: local obs for handle agent, None if agent has status DONE_REMOVED
        """
        agents = self.env.agents
        agent = agents[handle]
        
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:  # agent is DONE_REMOVED
            return None
            
        # Compute field of view
        visible_cells, rel_coords = self._field_of_view(agent_virtual_position, agent.direction)
        subtarget = self._find_subtarget(handle, visible_cells)
        # Add the visited cells to the observed cells (for visualization)
        self.env.dev_obs_dict[handle] = set(visible_cells)
        
        # Get local rail_obs
        local_rail_obs = np.zeros((self.view_height, self.view_width, 16))
        '''
        for i in range(self.view_height):
            for j in range(self.view_width):
                pos = visible_cells[i,j]
                if pos is not -np.inf:
                    local_rail_obs[i, j] = self.rail_obs[pos[0], pos[1]]
        '''
        # Build agents obs
        agents_state_obs = np.zeros((self.view_height, self.view_width, 5))
        # Build targets obs
        targets_obs = np.zeros((self.view_height, self.view_width, 2))
        i = 0
        for pos in visible_cells:  # Absolute coords
            curr_rel_coord = rel_coords[i]  # Convert into relative coords
            local_rail_obs[curr_rel_coord[0], curr_rel_coord[1], :] = self.rail_obs[pos[0], pos[1], :]
            
            if pos == agent_virtual_position:
                # Collect this agent position and direction
                agents_state_obs[curr_rel_coord[0], curr_rel_coord[1], 0] = agent.direction
            if pos == subtarget:
                # Collect position of agent target
                targets_obs[curr_rel_coord[0], curr_rel_coord[1], 0] = 1     
            for a in agents:
                # Collect info about active agents: positions and directions, malfunctions length, speed/priorities
                if a.status == RailAgentStatus.ACTIVE:
                    if pos == a.position:
                        agents_state_obs[curr_rel_coord[0], curr_rel_coord[1], 1] = a.direction
                        agents_state_obs[curr_rel_coord[0], curr_rel_coord[1], 2] = a.malfunction_data['malfunction']
                        agents_state_obs[curr_rel_coord[0], curr_rel_coord[1], 3] = a.speed_data['speed']
                # Collect info about ready to depart agents
                elif a.status == RailAgentStatus.READY_TO_DEPART:
                    if pos == a.initial_position:
                        agents_state_obs[curr_rel_coord[0], curr_rel_coord[1], 4] = a.initial_direction
                # Collect positions of other agents targets
                if pos == a.target:
                    targets_obs[curr_rel_coord[0], curr_rel_coord[1], 1] = 1
            i += 1
            
        return local_rail_obs, agents_state_obs, targets_obs
    
    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """
        self.shortest_paths = get_shortest_paths(self.env.distance_map)
        return super().get_many(handles)
    
    '''
    Given agent current position and direction, returns the field of view of the agent as np.array of cells on the grid in
    absolute coordinates. Value is -np.inf to indicate padding (when agent lies on border).
    '''
    def _field_of_view(self, position, direction):
        
        # Compute visible cells
        # visible_cells = np.full((self.view_height, self.view_width, 2), -np.inf, dtype=int)
        visible_cells = list()
        rel_coords = list()

        if direction == 0:  # North
            origin = (position[0] - self.offset, position[1] - self.view_semiwidth)
        elif direction == 1:  # East
            origin = (position[0] - self.view_semiwidth, position[1] + self.offset)
        elif direction == 2:  # South
            origin = (position[0] + self.offset, position[1] + self.view_semiwidth)
        else:  # West
            origin = (position[0] + self.view_semiwidth, position[1] - self.offset)

        for i in range(self.view_height):
            for j in range(self.view_width):
                if direction == 0:
                    cell_to_add = (origin[0] + i, origin[1] + j)
                elif direction == 1:  # Rectangle is flipped 90°
                    cell_to_add = (origin[0] + j, origin[1] - i)
                elif direction == 2:
                    cell_to_add = (origin[0] - i, origin[1] - j)
                elif direction == 3:
                    cell_to_add = (origin[0] - j, origin[1] + i)

                if cell_to_add[0] >= self.env.height or cell_to_add[1] >= self.env.width or cell_to_add[0] < 0 or \
                        cell_to_add[1] < 0:
                    break
                # visible_cells[i, j] = cell_to_add  # In absolute coordinates
                visible_cells.append(cell_to_add)
                rel_coords.append((i, j))
                
        return visible_cells, rel_coords

    
    def _find_subtarget(self, handle, visible_cells):
        """
        
        :param handle: agent id, 
                visible_cells: pos (y, x) of visible cells for agent in absolute coordinates
        :return: cell in the shortest path of this agent (handle) that is closest to target and visible to the agent 
        (i.e. in its field of view). Equal to real target when already visible.
        """
        if self.shortest_paths is not None:
            # Get shortest path
            shortest_path = self.shortest_paths[handle]
            # Walk path from target to source to find first pos that is in view
            for i in range(len(shortest_path) - 1, -1, -1):
                cell = shortest_path[i][0]
                if cell in visible_cells: # Not really efficient
                    return cell
        else:
            return None