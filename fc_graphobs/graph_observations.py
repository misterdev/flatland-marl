"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""

import collections
from typing import Optional, List, Dict, Tuple
import queue
import numpy as np
from collections import defaultdict


from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.core.grid.grid4_utils import get_new_position
from fc_graphobs.draw_obs_graph import build_graph


class GraphObsForRailEnv(ObservationBuilder):

    Node = collections.namedtuple('Node',
                                  'cell_position ' # Cell position (x, y)
                                  'agent_direction ' # Direction with which the agent arrived in this node
                                  'is_target') # whether agent's target is in this cell

    def __init__(self, bfs_depth=2):
        super(GraphObsForRailEnv, self).__init__()
        self.bfs_depth = bfs_depth

    def set_env(selfself, env: Environment):
        super().set_env(env)

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> {}:
        observations = {}

        for a in handles:
            observations[a] = self.get(a)
        return observations

    '''
    Return a graph (dict) of nodes, where nodes are identified by ids, graph is directed, depends on agent direction
    (that are tuples that represent the cell position, eg (11, 23))
    '''
    def get(self, handle: int = 0) -> {}:
        obs_graph = defaultdict(list)  # dict
        visited_nodes = set()  # set
        bfs_queue = []
        done = False  # True if agent has reached its target

        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
            done = True
        else:
            return None

        agent_current_direction = agent.direction

        # Push root node into the queue
        root_node_obs = GraphObsForRailEnv.Node(cell_position=agent_virtual_position,
                                                agent_direction=agent_current_direction,
                                                is_target=done)
        bfs_queue.append(root_node_obs)

        # Perform BFS of depth = bfs_depth
        for i in range(1, self.bfs_depth + 1):
            # Temporary queue to store nodes that must be appended at the next pass
            tmp_queue = []
            while not len(bfs_queue) == 0:
                current_node = bfs_queue.pop(0)
                agent_position = current_node[0]
                agent_current_direction = current_node[1]
                # Get cell transitions given agent direction
                possible_transitions = self.env.rail.get_transitions(*agent_position, agent_current_direction)

                orientation = agent_current_direction
                possible_branch_directions = []
                # Build list of possible branching directions from cell
                for j, branch_direction in enumerate([(orientation + j) % 4 for j in range(-1, 3)]):
                    if possible_transitions[branch_direction]:
                        possible_branch_directions.append(branch_direction)
                for branch_direction in possible_branch_directions:
                    # Gets adjacent cell and start exploring from that for possible fork points
                    neighbour_cell = get_new_position(agent_position, branch_direction)
                    adj_node = self._explore_path(handle, neighbour_cell, branch_direction)
                    if not (*adj_node[0], adj_node[1]) in visited_nodes:
                        # For now I'm using as key the agent_position tuple
                        obs_graph[agent_position].append(adj_node)
                        visited_nodes.add((*adj_node[0], adj_node[1]))
                        tmp_queue.append(adj_node)
            # Add all the nodes of the next level to the BFS queue
            for el in tmp_queue:
                bfs_queue.append(el)
            # At the last pass add adj nodes to the obs graph wih empty lists
            if i == self.bfs_depth:
                for el in bfs_queue:
                    if not el[0] in obs_graph.keys():
                        obs_graph[el[0]] = []


        # For obs rendering
        # self.env.dev_obs_dict[handle] = [(node[0], node[1]) for node in visited_nodes]

        # Build graph with graph-tool library for visualization
        g = build_graph(obs_graph)

        return obs_graph

    # Find next branching point
    def _explore_path(self, handle, position, direction):

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        # 4 different cases to have a branching point:
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell or cycle
        last_is_target = False  # target was reached
        agent = self.env.agents[handle]
        visited = OrderedSet()
        
        while True:

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
                    break

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
            
            elif num_transitions > 1:
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
        node = GraphObsForRailEnv.Node(cell_position=position,
                                       agent_direction=direction,
                                       is_target=last_is_target) # TODO

        return node
