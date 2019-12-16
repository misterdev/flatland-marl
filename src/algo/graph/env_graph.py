import numpy as np
from typing import NamedTuple
from flatland.core.grid.grid4_utils import get_new_position, get_direction, direction_to_point
from src.algo.shortest_path import scheduling

CardinalNode = \
	NamedTuple('CardinalNode', [('id_node', int), ('cardinal_point', int)])
'''
Train = \
	NamedTuple('Train', [
		('handle', int), 
		(('id_edge', int), ('crossing_dir', bool), ('dist', int)), 
		(('id_node', int), ('cp', int)), 
		('speed', int)
	])
'''

class EnvGraph():
	"""
	This class provides a simplified representation of the environment.
	The map is a graph G = (V, E) where V are switches and E are rails connecting these switches.
	"""

	def __init__(self, env, strategy):
		self.env = env
		self.strategy = strategy # TODO 
		self.cell_to_id_node = {} # Map cell position : id_node
		self.id_node_to_cell = {} # Map id_node to cell position
		self.connections = {} # Map id_node : connections(node)
		self.info = {} # Map id_edge : tuple (CardinalNode1, CardinalNode2, edge_length)
		self.id_edge_to_cells = {} # Map id_edge : list of tuples (cell pos, crossing dir) in rail (nodes are not counted)
		self.nodes = set() # Set of node ids
		self.edges = set() # Set of edge ids
		self.trains = [] # List of trains in this env
	
	def reset(self):
		"""
		TODO Reset members when episode changes
		:return: 
		"""
		pass
	
	def map_to_graph(self):
		"""
		
		:return: 
		"""
		id_node_counter = 0
		targets = [agent.target for agent in self.env.agents]
		
		# Identify cells hat are nodes (have switches)
		for i in range(self.env.height):
			for j in range(self.env.width):
	
				is_switch = False	
				is_crossing = False
				is_target = False
				connections_matrix = np.zeros((4, 4)) # Matrix NESW x NESW
				
				# Check if diamond crossing
				transitions_bit = bin(self.env.rail.get_full_transitions(i, j))
				if int(transitions_bit, 2) == int('1000010000100001', 2):
					is_crossing = True
					connections_matrix[0, 2] = connections_matrix[2, 0] = 1
					connections_matrix[1, 3] = connections_matrix[3, 1] = 1
	
				else:
					# Check if target
					if (i, j) in targets:
						is_target = True
					# Check if switch
					for direction in (0, 1, 2, 3):	# 0:N, 1:E, 2:S, 3:W
						possible_transitions = self.env.rail.get_transitions(i, j, direction)
						for t in range(4): # Check groups of bits
							if possible_transitions[t]:
								inv_direction = (direction + 2) % 4
								connections_matrix[inv_direction, t] = connections_matrix[t, inv_direction] = 1
						num_transitions = np.count_nonzero(possible_transitions)
						if num_transitions > 1:
							is_switch = True	
							
				if is_switch or is_crossing or is_target:
					# Add node - keep info on cell position
					# Update only for nodes that are switches
					self.connections.update({id_node_counter: connections_matrix})
					self.id_node_to_cell.update({id_node_counter: (i, j)})
					self.cell_to_id_node.update({(i, j): id_node_counter})
					id_node_counter += 1
				
		# Enumerate edges from these nodes
		id_edge_counter = 0
		# Start from connections of one node and follow path until next switch is found
		nodes = self.connections.keys() # ids
		visited = set()  # Keeps set of CardinalNodes that were already visited
		for n in nodes:
			for cp in range(4): # Check edges from the 4 cardinal points
				if np.count_nonzero(self.connections[n][cp, :]) > 0:
					visited.add(CardinalNode(n, cp)) # Add to visited
					cells_sequence = []
					node_found = False
					edge_length = 0
					# Keep going until another node is found
					direction = cp
					pos = self.id_node_to_cell[n]
					while not node_found:
						neighbour_pos = get_new_position(pos, direction)
						cells_sequence.append((neighbour_pos, direction))
						if neighbour_pos in self.cell_to_id_node: # If neighbour is a node
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
								cells_sequence.pop() # Don't include this node in the edge
								self.id_edge_to_cells.update({id_edge_counter: cells_sequence})
								id_edge_counter += 1
								visited.add(CardinalNode(id_node2, cp2))
							break
						edge_length += 1 # Not considering switches in the count
						# Update pos and dir
						pos = neighbour_pos
						exit_dir = self._reverse_dir(direction) 
						possible_transitions = np.array(self.env.rail.get_transitions(pos[0], pos[1], direction))
						possible_transitions[exit_dir] = 0 # Don't consider direction from which I entered
						# t = 2
						t = np.argmax(possible_transitions) # There's only one possible transition except the one that I took to get in
						temp_pos = get_new_position(pos, t)
						if 0 <= temp_pos[0] < self.env.height and 0 <= temp_pos[1] < self.env.width:	# Patch - check if this cell is a rail
							# Entrance dir is always opposite to exit dir
							direction = t
								
		# Build graph object made of vertices and edges
		edges = self.info.keys()
		
		# Create trains
		self.trains = [self._train_on_graph(a, nodes, edges) for a in range(self.env.get_num_agents())]
		'''
		for a in self.env.get_num_agents():
			trains.append(self._train_on_graph(a, nodes, edges))
		'''
		self.nodes = nodes
		self.edges = edges
		
		return self.cell_to_id_node.keys(), edges # Graph as a tuple (list of vertices pos, list of edges) TODO Change return and change in GraphObs

	def _reverse_dir(self, direction):
		"""
		Invert direction (int) of one agent.
		:param direction: 
		:return: 
		"""
		return int((direction + 2) % 4)

	#ogni treno ha un suo id, una posizione definita da
	# - un binario (un arco)
	# - una direzione di attraversamento (0,1)
	# - la distanza percorsa sul binario (int)
	# una velocita
	# una destinazione (id del nodo)
	# altre info

	def _train_on_graph(self, handle, nodes, edges):
		"""
		Given an agent/a train in the env returns a representation of it that is suitable for the EnvGraph.
		:param: handle: agent id
		:return: ((id_edge, crossing_dir, dist) , (id_target_node, {possible cps}), speed) where
		id_edge: id of the rail where agent currently lies
		crossing_dir: direction in which the agent is crossing the rail (1 if from node1 to node2, 0 otherwise)
		dist: total distance walked from the beginning of the rail
		id_target_node: id of the node where agent's target lies
		possible_cps: set of possible cardinal points from which the agent may enter its target
		speed: agent speed
		"""
		agent = self.env.agents[handle]
		agent_pos = agent.initial_position 
		dist = 0 # Distance already walked along the rail
		id_edge = None
		id_target_node = None
		crossing_dir = None
		for e in edges:
			# la dist Considera anche il tempo? TODO
			index = [x for x, y in enumerate(self.id_edge_to_cells[e]) if y[0][0] == agent_pos[0] and y[0][1] == agent_pos[1]]
			if index: # If index is not empty
				dist = index[0]
				dir = self.id_edge_to_cells[e][dist][1]
				crossing_dir = 1 if dir == agent.direction else 0 # Direction saved is considered as crossing_dir = 1
				id_edge = e
				break
		
		# Determine target id
		for n in nodes:
			if self.id_node_to_cell[n] == agent.target:
				id_target_node = n
				break
		
		speed = agent.speed_data['speed']
		
		# Possible cps to enter target node depend on connections available
		target_connections = self.connections[id_target_node]
		cps = []
		for i in range(4):
			if np.any(target_connections[i]):
				cps.append(i)
				
		return handle, (id_edge, crossing_dir, dist), (id_target_node, cps), speed
	
	# def run_strategy...

	def run_shortest_path(self):
		"""
		
		:return: 
		"""
		paths, available_at = scheduling((self.nodes, self.edges), self.trains, self.info, self.connections)
		return paths, available_at

