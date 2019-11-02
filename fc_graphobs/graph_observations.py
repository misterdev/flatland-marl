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
                                  'cell_position '  # Cell position (x, y)
                                  'agent_direction '  # Direction with which the agent arrived in this node
                                  'is_target')  # Whether agent's target is in this cell

    def __init__(self, bfs_depth, predictor):
        super(GraphObsForRailEnv, self).__init__()
        self.bfs_depth = bfs_depth
        self.predictor = predictor
        self.prediction_dict = {}


    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            # Use set_env available in PredictionBuilder (parent class)
            self.predictor.set_env(self.env)

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> {}:
        observations = {}
        self.prediction_dict = self.predictor.get()

        for a in handles:
            observations[a] = self.get(a)
        return observations

    '''
    Return a graph (dict) of nodes, where nodes are identified by ids, graph is directed, depends on agent direction
    (that are tuples that represent the cell position, eg (11, 23))
    '''
    def get(self, handle: int = 0) -> {}:

        bfs_graph = self._bfs_graph(handle)
        # TODO For the moment: computes the obs_graph (dunno what to do with it exactly) but return the next_cell obs
        prediction_dict = self.predictor.get()
        cells_sequence = self.predictor.compute_cells_sequence(prediction_dict)

        # Debug
        # Visualize paths that are overlapping (use graph tool?) or print to file
        for a in self.env.agents:
            print(str(a.handle) + ": ", end='')
            for cell in cells_sequence[a.handle]:
                print(str(cell) + " ", end='')
            print()

        occupancy_dict = {}
        for a in self.env.agents:
            occupancy_dict[a.handle] = self._fill_occupancy(a.handle)

        # TODO This should return the tot cells sequence to use as state to feed into the network
        return bfs_graph

    def _bfs_graph(self, handle: int = 0) -> {}:
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

                # Init node in the obs_graph (if first time)
                if not agent_position in obs_graph.keys():
                    obs_graph[agent_position] = []

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

        # After the last pass add adj nodes to the obs graph wih empty lists
        for el in bfs_queue:
            if not el[0] in obs_graph.keys():
                obs_graph[el[0]] = []
                # visited_nodes.add((*el[0], el[1]))
        # For obs rendering
        # self.env.dev_obs_dict[handle] = [(node[0], node[1]) for node in visited_nodes]

        # Build graph with graph-tool library for visualization
        g = build_graph(obs_graph, handle)


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

    # TODO Generalize to multiple agents later and also definition of possible conflict must be improved
    # Should consider a span of conflicts, not just one
    '''
    Function that given predicted cells sequence for two different agents (as list of tuples) and a specific time step
    returns 1 if a possible conflict is predicted at that ts.
    ts must be < prediction_depth - 1 e > 0
    '''
    def _possible_conflict(self, sequence, possible_overlapping_sequence, ts):

            cell_pos = sequence[ts]
            if cell_pos == possible_overlapping_sequence[ts] or cell_pos == possible_overlapping_sequence[ts-1] or cell_pos == possible_overlapping_sequence[ts+1]:
                return 1
            return 0

    '''
    Returns one-hot encoding of 
    0: no other agent in this cell at this ts
    1: (probably) other agents here at the same ts, so conflict (for now, check with any other agent, but we could use several "layers")
    NOW RETURNS THE SUM 
    Update using idea of branch and direction
    '''
    def _fill_occupancy(self, handle):
        agents = self.env.agents
        agent = agents[handle]
        other_agents = []
        for a in agents:
            if not a == agent:
                other_agents.append(a)

        prediction_depth = self.predictor.get_prediction_depth()
        occupancy = np.zeros(prediction_depth)
        cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)
        agent_sequence = cells_sequence[handle]
        for a in other_agents:
            possible_overlapping_sequence = cells_sequence[a.handle]
            for ts in range(1, prediction_depth - 1):  # TODO Check if it works of first and last ts
                occupancy[ts] += self._possible_conflict(agent_sequence, possible_overlapping_sequence, ts)

        return occupancy
