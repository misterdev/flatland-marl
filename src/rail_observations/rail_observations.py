"""
A class to build observations for agents.
"""
import collections
from typing import Optional, List, Dict, Tuple, NamedTuple

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position

CardinalNode = \
	NamedTuple('CardinalNode', [('id_node', int), ('cardinal_point', int)])

"""
Things to figure out:
- The observation acts considering that ALL agents start at once at the beginning, so at ts=0 they already have a position on the grid 
Could be useful to have a limited number of agents contemporary active to control the number of bitmaps to feed to the network.
- How to consider a switch: part of a rail, include it in the bitmap or not, etc. now there are rails connecting two switches 
 but they have length = 0 so switches are not considered in the bitmap.
- The path to draw the bitmap must be recomputed at every ts, so maybe prediction depth so high (2000) is inefficient and unnecessary
- TODO How to return the actions (for all the agents at once?)

Current implementation:
- Truncating the prediction at the point where target is reached (all 0s in the bitmap after target)
- Prediction now doesn't consider if the agent is currently moving or not (so the bitmap still show all the path even though the agent is stopped)
If it did, then we would have a row of 1/-1 in the bitmap. Both choices give wrong info about future moves.
"""

class RailObsForRailEnv(ObservationBuilder):
	"""
	This class returns an observation of rails occupancy in time in a bitmap.

			--- timesteps --->
	rail 0: 1 1 1       -1-1
	rail 1:      1 1
	rail 2:         -1-1
	.
	.
	rail n:
	
	where rails are edges and 1/-1 indicates the traversal direction on the rail.
	"""

	def __init__(self, predictor):
		"""
	
		:param predictor: class that predicts the path.
		"""
		super(RailObsForRailEnv, self).__init__()
		self.predictor = predictor
		self.prediction_dict = {}  # Dict handle : list of tuples representing prediction steps
		self.cells_sequence = {} # Dict handle : list of tuples representing cell positions
		
		self.num_rails = None # Depends on the map, must be computed in reset()
		self.max_time_steps = self.predictor.max_depth
		# Not all of them are necessary
		self.cell_to_id_node = {} # Map cell position : id_node
		self.id_node_to_cell = {} # Map id_node to cell position
		self.connections = {} # Map id_node : connections(node)
		self.info = {} # Map id_edge : tuple (CardinalNode1, CardinalNode2, edge_length)
		self.id_edge_to_cells = {} # Map id_edge : list of tuples (cell pos, crossing dir) in rail (nodes are not counted)
		self.nodes = set() # Set of node ids
		self.edges = set() # Set of edge ids

	def set_env(self, env: Environment):
		super().set_env(env)
		if self.predictor:
			# Use set_env available in PredictionBuilder (parent class)
			self.predictor.set_env(self.env)

	def reset(self):
		"""
		
		:return: 
		"""
		self._map_to_graph() # Fill member variables
		
		
	def get(self, handle: int = 0) -> np.ndarray:
		"""
		Currently this implementation of bitmap has 'holes' of zeros when agent lies on a switch. TODO
		:param handle: 
		:return: Bitmap of rail obs for agent handle.
		"""
		
		rail_obs = np.zeros((self.num_rails, self.max_time_steps + 1), dtype=int) # Max steps in the future + current ts
		agent = self.env.agents[handle]
		path = self.cells_sequence[handle]
		# Truncate path in the future, after reaching target
		target_index = [i for i, pos in enumerate(path) if pos[0] == agent.target[0] and pos[1] == agent.target[1]]
		if len(target_index) != 0:
			target_index = target_index[0]
			path = path[:target_index+1]
		
		# Fill rail occupancy according to predicted position at ts
		for ts in range(len(path)):
			cell = path[ts]
			# Find rail associated to cell
			rail, dist = self.get_edge_from_cell(cell)
			# Find crossing direction
			if rail != -1: # Means agent is not on a switch
				direction = self.id_edge_to_cells[rail][dist][1]
				crossing_dir = 1 if direction == agent.direction else -1 # Direction saved is considered as crossing_dir = 1
				
				rail_obs[rail, ts] = crossing_dir
	
		return rail_obs
		
		
	def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
		"""
		
		:param handles: 
		:return: 
		"""
		self.prediction_dict = self.predictor.get()
		self.cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)
		
		observations = {}
		for a in handles:
			observations[a] = self.get(a)
		return observations
	
	# Slightly modified wrt to the other
	def _map_to_graph(self):
		"""
		Build the representation of the map as a graph.
		:return: 
		"""
		id_node_counter = 0
		# targets = [agent.target for agent in self.env.agents]

		# Identify cells hat are nodes (switches or diamond crossings)
		for i in range(self.env.height):
			for j in range(self.env.width):

				is_switch = False
				is_crossing = False
				# is_target = False
				connections_matrix = np.zeros((4, 4))  # Matrix NESW x NESW

				# Check if diamond crossing
				transitions_bit = bin(self.env.rail.get_full_transitions(i, j))
				if int(transitions_bit, 2) == int('1000010000100001', 2):
					is_crossing = True
					connections_matrix[0, 2] = connections_matrix[2, 0] = 1
					connections_matrix[1, 3] = connections_matrix[3, 1] = 1

				else:
					# Check if target
					# if (i, j) in targets:
					#	is_target = True
					# Check if switch
					for direction in (0, 1, 2, 3):  # 0:N, 1:E, 2:S, 3:W
						possible_transitions = self.env.rail.get_transitions(i, j, direction)
						for t in range(4):  # Check groups of bits
							if possible_transitions[t]:
								inv_direction = (direction + 2) % 4
								connections_matrix[inv_direction, t] = connections_matrix[t, inv_direction] = 1
						num_transitions = np.count_nonzero(possible_transitions)
						if num_transitions > 1:
							is_switch = True

				if is_switch or is_crossing: #or is_target:
					# Add node - keep info on cell position
					# Update only for nodes that are switches
					self.connections.update({id_node_counter: connections_matrix})
					self.id_node_to_cell.update({id_node_counter: (i, j)})
					self.cell_to_id_node.update({(i, j): id_node_counter})
					id_node_counter += 1

		# Enumerate edges from these nodes
		id_edge_counter = 0
		# Start from connections of one node and follow path until next switch is found
		nodes = self.connections.keys()  # ids
		visited = set()  # Keeps set of CardinalNodes that were already visited
		for n in nodes:
			for cp in range(4):  # Check edges from the 4 cardinal points
				if np.count_nonzero(self.connections[n][cp, :]) > 0:
					visited.add(CardinalNode(n, cp))  # Add to visited
					cells_sequence = []
					node_found = False
					edge_length = 0
					# Keep going until another node is found
					direction = cp
					pos = self.id_node_to_cell[n]
					while not node_found:
						neighbour_pos = get_new_position(pos, direction)
						cells_sequence.append((neighbour_pos, direction))
						if neighbour_pos in self.cell_to_id_node:  # If neighbour is a node
							# node_found = True
							# Build edge, mark visited
							id_node1 = n
							cp1 = cp
							id_node2 = self.cell_to_id_node[neighbour_pos]
							cp2 = self._reverse_dir(direction)
							if CardinalNode(id_node2, cp2) not in visited:
								self.info.update({id_edge_counter:
									                  (CardinalNode(id_node1, cp1),
									                   CardinalNode(id_node2, cp2),
									                   edge_length)})
								cells_sequence.pop()  # Don't include this node in the edge
								self.id_edge_to_cells.update({id_edge_counter: cells_sequence})
								id_edge_counter += 1
								visited.add(CardinalNode(id_node2, cp2))
							break
						edge_length += 1  # Not considering switches in the count
						# Update pos and dir
						pos = neighbour_pos
						exit_dir = self._reverse_dir(direction)
						possible_transitions = np.array(self.env.rail.get_transitions(pos[0], pos[1], direction))
						possible_transitions[exit_dir] = 0  # Don't consider direction from which I entered
						# t = 2
						t = np.argmax(
							possible_transitions)  # There's only one possible transition except the one that I took to get in
						temp_pos = get_new_position(pos, t)
						if 0 <= temp_pos[0] < self.env.height and 0 <= temp_pos[1] < self.env.width:  # Patch - check if this cell is a rail
							# Entrance dir is always opposite to exit dir
							direction = t

		edges = self.info.keys()

		self.nodes = nodes # Set of nodes
		self.edges = self.info.keys() # Set of edges
		self.num_rails = len(self.edges)

	def get_edge_from_cell(self, cell):
		"""

		:param cell: Cell for which we want to find the associated rail id.
		:return: A tuple (id rail, dist) where dist is the distance as offset from the beginning of the rail.
		"""

		for edge in self.id_edge_to_cells.keys():
			cells = [cell[0] for cell in self.id_edge_to_cells[edge]] 
			dist = 0
			if cell in cells:
				return edge, cells.index(cell)
				dist += 1

		return -1, -1  # Node

	@staticmethod
	def _reverse_dir(direction):
		"""
		Invert direction (int) of one agent.
		:param direction: 
		:return: 
		"""
		return int((direction + 2) % 4)
		pass