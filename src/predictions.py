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
            - action taken to come here - my implementation
            The prediction at 0 is the current position, direction etc.
        """
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]
        distance_map: DistanceMap = self.env.distance_map

        shortest_paths = get_shortest_paths(distance_map, max_depth=self.max_depth)
        self.shortest_paths = shortest_paths
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
                for i in range(self.max_depth):
                    prediction[i] = [i, None, None, None, None]
                prediction_dict[agent.handle] = prediction
                continue

            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            # First cell is info relative to actual timestep, but I don't need this one TODO
            #prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # If there is a shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position
            visited = OrderedSet()
            for index in range(self.max_depth):
                # If we're at the target or not moving, stop moving until max_depth is reached
                # TODO Changing this to avoid stuck agent
                #if new_position == agent.target or not agent.moving or not shortest_path:
                if new_position == agent.target or not shortest_path:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                if index % times_per_cell == 0:

                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction

                    shortest_path = shortest_path[1:]

                # Prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                # prediction[index] = [index, *new_position, new_direction, action]
                visited.add((*new_position, new_direction))

            # TODO: very bady side effects for visualization only: hand the dev_pred_dict back instead of setting on env!
            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

        return prediction_dict

    '''
    Given prediction dict for all agents, return sequence of cells walked in the prediction as a dict
    where key is the agent handle and value is the list of tuples (xi, yi) that are crossed.
    Mostly used to debug.
    '''

    def compute_cells_sequence(self, prediction_dict):

        cells_sequence = defaultdict(list)
        agents = self.env.agents
        for a in agents:
            for step in prediction_dict[a.handle]:
                cell_pos = (step[1], step[2])  # Takes (yi, xi)
                cells_sequence[a.handle].append(cell_pos)

        return cells_sequence

    def get_prediction_depth(self):
        return self.max_depth

    def get_shortest_paths(self):
        return self.shortest_paths

    def get_k_shortest_paths(self,
                             source_position: Tuple[int, int],
                             source_direction: int,
                             target_position=Tuple[int, int],
                             k: int = 1, debug=False) -> List[Tuple[Waypoint]]:
        """
        Computes the k shortest paths using modified Dijkstra
        following pseudo-code https://en.wikipedia.org/wiki/K_shortest_path_routing
        In contrast to the pseudo-code in wikipedia, we do not a allow for loopy paths.

        Parameters
        ----------
        env :             RailEnv
        source_position:  Tuple[int,int]
        source_direction: int
        target_position:  Tuple[int,int]
        k :               int
            max number of shortest paths
        debug:            bool
            print debug statements

        Returns
        -------
        List[Tuple[WalkingElement]]
            We use tuples since we need the path elements to be hashable.
            We use a list of paths in order to keep the order of length.
        """

        # P: set of shortest paths from s to t
        # P =empty,
        shortest_paths: List[Tuple[Waypoint]] = []

        # countu: number of shortest paths found to node u
        # countu = 0, for all u in V
        count = {(r, c, d): 0 for r in range(self.env.height) for c in range(self.env.width) for d in range(4)}

        # B is a heap data structure containing paths
        # N.B. use OrderedSet to make result deterministic!
        heap: OrderedSet[Tuple[Waypoint]] = OrderedSet()

        # insert path Ps = {s} into B with cost 0
        heap.add((Waypoint(source_position, source_direction),))

        # while B is not empty and countt < K:
        while len(heap) > 0 and len(shortest_paths) < k:
            if debug:
                print("iteration heap={}, shortest_paths={}".format(heap, shortest_paths))
            # – let Pu be the shortest cost path in B with cost C
            cost = np.inf
            pu = None
            for path in heap:
                if len(path) < cost:
                    pu = path
                    cost = len(path)
            u: Waypoint = pu[-1]
            if debug:
                print("  looking at pu={}".format(pu))

            #     – B = B − {Pu }
            heap.remove(pu)
            #     – countu = countu + 1

            urcd = (*u.position, u.direction)
            count[urcd] += 1

            # – if u = t then P = P U {Pu}
            if u.position == target_position:
                if debug:
                    print(" found of length {} {}".format(len(pu), pu))
                shortest_paths.append(pu)

            # – if countu ≤ K then
            # CAVEAT: do not allow for loopy paths
            elif count[urcd] <= k:
                possible_transitions = self.env.rail.get_transitions(*urcd)
                if debug:
                    print("  looking at neighbors of u={}, transitions are {}".format(u, possible_transitions))
                #     for each vertex v adjacent to u:
                for new_direction in range(4):
                    if debug:
                        print("        looking at new_direction={}".format(new_direction))
                    if possible_transitions[new_direction]:
                        new_position = get_new_position(u.position, new_direction)
                        if debug:
                            print("        looking at neighbor v={}".format((*new_position, new_direction)))

                        v = Waypoint(position=new_position, direction=new_direction)
                        # CAVEAT: do not allow for loopy paths
                        if v in pu:
                            continue

                        # – let Pv be a new path with cost C + w(u, v) formed by concatenating edge (u, v) to path Pu
                        pv = pu + (v,)
                        #     – insert Pv into B
                        heap.add(pv)

        # return P
        return shortest_paths


    
