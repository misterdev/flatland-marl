"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.core.grid.grid4_utils import get_new_position

import collections
from typing import Optional, List
import queue


import numpy as np


class GraphObsForRailEnv(ObservationBuilder):

    Node = collections.namedtuple('Node', 'node_position')

    def __init__(self, bfs_depth = 2):
        super(GraphObsForRailEnv, self).__init__()
        self.bfs_depth = bfs_depth

    def set_env(selfself, env: Environment):
        super().set_env(env)

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> Node:
        return True
    
    # Return a graph (dict) of nodes, where nodes are identified by ids, graph is directed, depends on agent direction
    # (that are tuples that represent the cell position)
    # Decide graph depth to use bfs
    def get(self, handle: int = 0) -> {}:

        obs_graph = {}
        visited_nodes = {}
        agent = self.env_agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None
        
        distance_map = self.env.distance_map
        agent_curr_dir = agent.direction
        
        # Push root node into the queue
        bfs_queue = queue.Queue(maxsize=3) # 3 like forward, right, left, max of 3 nodes at one level 
        root_node_obs = GraphObsForRailEnv.Node(node_position=agent_virtual_position)
        
        bfs_queue.put(root_node_obs) # TODO serve anche la dir probably

        # Perform BFS of depth bfs_depth
        for i in range(1, self.bfs_depth):
            while not bfs_queue.empty():
                current_node = bfs_queue.get()
                agent_position = current_node # tmp: now it stores only pos as tuple
                possible_transitions = self.env.rail.get_transitions(*agent_position, agent_curr_dir)
                num_transitions = np.count_nonzero(possible_transitions)

                orientation = agent.direction
                # If there's only one possible transition just pick that
                if num_transitions == 1:
                    orientation = np.argmax(possible_transitions)

                for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):
                    if possible_transitions[branch_direction]:
                        new_cell = get_new_position(agent_virtual_position, branch_direction)
                        adj_node = self._explore_path(handle, new_cell, branch_direction, 1, 1)
                        bfs_queue.put(adj_node)
                        obs_graph[current_node] = adj_node
            
            # left, forward, right
                        
        return obs_graph

    def _explore_path(self, handle, position, direction, depth):
        if depth > self.bfs_depth:
            return [], []
        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        # 4 different cases to have a branching point:
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell or cycle
        last_is_target = False # target was reached
        agent = self.env.agents[handle]
        visited = OrderedSet()
        
        while exploring:

            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            cell_transitions = self.env.rail.get_transitions(*position, direction)
            num_transitions = np.count_nonzero(cell_transitions)
            cell_transitions_bitmap = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = cell_transitions_bitmap.count("1")

            if num_transitions == 1:
                # Check if dead-end (1111111111111111), or if we can go forward along direction
                if total_transitions == 1:
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)

            elif num_transitions > 0:
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break
        # Out of while loop - a branching point was found
        # TODO tmp build node features and save them here
        node = position

        return node

# TODO follow only valid paths, not all of them