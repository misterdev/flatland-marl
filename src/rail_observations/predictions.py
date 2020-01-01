"""
Collection of environment-specific PredictionBuilder.
"""

import numpy as np
from collections import defaultdict
from typing import NamedTuple, Tuple, List


from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvActions, RailEnvNextAction
from flatland.envs.rail_env_shortest_paths import get_shortest_paths, get_new_position
from flatland.utils.ordered_set import OrderedSet



WalkingElement = \
    NamedTuple('WalkingElement',
               [('position', Tuple[int, int]), ('direction', int), ('next_action_element', RailEnvNextAction)])

# Flatland v3
# A way point is the entry into a cell defined by
# - the row and column coordinates of the cell entered
# - direction, in which the agent is facing to enter the cell.
# This induces a graph on top of the FLATland cells:
# - four possible way points per cell
# - edges are the possible transitions in the cell.
Waypoint = NamedTuple('Waypoint', [('position', Tuple[int, int]), ('direction', int)])


# TODO 'Add action taken to come here' info

class ShortestPathPredictorForRailEnv(PredictionBuilder):
    """
    ShortestPathPredictorForRailEnv object.

    This object returns shortest-path predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def __init__(self, max_depth: int = 20):
        self.shortest_paths = None
        super().__init__(max_depth)

    def get(self, handle: int = None):
        """
        Requires distance_map to extract the shortest path.
        Does not take into account future positions of other agents!

        If there is no shortest path, the agent just stands still and stops moving.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        np.array
            Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here - must be implemented TODO
            The prediction at 0 is the current position, direction etc.
        """
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]
        distance_map: DistanceMap = self.env.distance_map
        # Use map_depth + 1 to consider current time step 
        self.shortest_paths = shortest_paths = get_shortest_paths(distance_map, max_depth=self.max_depth + 1)
        
        prediction_dict = {}
        for agent in agents:

            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_virtual_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                agent_virtual_position = agent.target
            else:  # agent.status == DONE_REMOVED, prediction must be None
                prediction = np.zeros(shape=(self.max_depth + 1, 5))
                for i in range(self.max_depth + 1):
                    prediction[i] = [i, None, None, None, None]
                prediction_dict[agent.handle] = prediction
                continue

            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            # First cell is info relative to actual time step
            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # If there is a shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position
            visited = OrderedSet()
            for index in range(1, self.max_depth + 1):
                # If we're at the target or not moving, stop moving until max_depth is reached
                #if new_position == agent.target or not agent.moving or not shortest_path:
                if new_position == agent.target or not shortest_path: # Writing like this you don't consider the fact that the agent is stopped
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                if index % times_per_cell == 0:

                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction

                    shortest_path = shortest_path[1:]

                # Prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                visited.add((*new_position, new_direction))

            # TODO: very bady side effects for visualization only: hand the dev_pred_dict back instead of setting on env!
            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

        return prediction_dict

    def compute_cells_sequence(self, prediction_dict):
        """
        Given prediction dict for all agents, return sequence of cells walked in the prediction as a dict
        where key is the agent handle and value is the list of tuples (xi, yi) that are crossed.
        Mostly used to debug.
        :param prediction_dict: 
        :return: 
        """

        cells_sequence = defaultdict(list)
        agents = self.env.agents
        for a in agents:
            for step in prediction_dict[a.handle]:
                cell_pos = (step[1], step[2])  # Takes (yi, xi)
                cells_sequence[a.handle].append(cell_pos)

        return cells_sequence

    def get_prediction_depth(self):
        """
        
        :return: 
        """
        return self.max_depth

    def get_shortest_path_action(self, handle):
        """
        Takes an agent handle and returns next action for that agent following shortest path:
        - if agent status == READY_TO_DEPART => agent moves forward;
        - if agent status == ACTIVE => pick action according to shortest path;
        - if agent status == DONE => agent does nothing.
        :param handle: 
        :return: 
        """

        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            action = RailEnvActions.MOVE_FORWARD

        elif agent.status == RailAgentStatus.ACTIVE:
            # This can return None when rails are disconnected or there was an error in the DistanceMap
            if self.shortest_paths[handle] is None:  # Railway disrupted
                action = RailEnvActions.STOP_MOVING
            else:
                step = self.shortest_paths[handle][0]
                next_action_element = step[2][0]  # Get next_action_element

                # Just to use the correct form/name
                if next_action_element == 1:
                    action = RailEnvActions.MOVE_LEFT
                elif next_action_element == 2:
                    action = RailEnvActions.MOVE_FORWARD
                elif next_action_element == 3:
                    action = RailEnvActions.MOVE_RIGHT

        else:  # If status == DONE
            action = RailEnvActions.DO_NOTHING

        return action