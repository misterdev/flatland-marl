import numpy as np
from typing import NamedTuple
from flatland.core.grid.grid4_utils import get_new_position, get_direction, direction_to_point
from flatland.core.grid.rail_env_grid import RailEnvTransitions

from src.algo.flat import scheduling 
# TODO Refactor into a class

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
def map_to_graph(env):
	
	id_node_counter = 0
	cell_to_id_node = {} # Map cell position : id_node
	id_node_to_cell = {} # Map id_node to cell position
	connections = {} # Map id_node : connections(node)
	targets = [agent.target for agent in env.agents]
	
	# Identify cells hat are nodes (have switches)
	for i in range(env.height):
		for j in range(env.width):

			is_switch = False	
			is_crossing = False
			is_target = False
			connections_matrix = np.zeros((4, 4)) # Matrix NESW x NESW
			
			# Check if diamond crossing
			transitions_bit = bin(env.rail.get_full_transitions(i, j))
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
					possible_transitions = env.rail.get_transitions(i, j, direction)
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
				connections.update({id_node_counter: connections_matrix})
				id_node_to_cell.update({id_node_counter: (i, j)})
				cell_to_id_node.update({(i, j): id_node_counter})
				id_node_counter += 1
			
	# Enumerate edges from these nodes
	info = {} # Map id_edge : tuple (CardinalNode1, CardinalNode2, edge_length)
	id_edge_to_cells = {} # Map id_edge : list of tuples (cell pos, crossing dir) in rail (nodes are not counted)
	id_edge_counter = 0
	# Start from connections of one node and follow path until next switch is found
	nodes = connections.keys() # ids
	visited = set()  # Keeps set of CardinalNodes that were already visited
	for n in nodes:
		for cp in range(4): # Check edges from the 4 cardinal points
			if np.count_nonzero(connections[n][cp, :]) > 0:
				visited.add(CardinalNode(n, cp)) # Add to visited
				cells_sequence = []
				node_found = False
				edge_length = 0
				# Keep going until another node is found
				direction = cp
				pos = id_node_to_cell[n]
				while not node_found:
					neighbour_pos = get_new_position(pos, direction)
					cells_sequence.append((neighbour_pos, direction))
					if neighbour_pos in cell_to_id_node: # If neighbour is a node
						# node_found = True
						# Build edge, mark visited
						id_node1 = n
						cp1 = cp
						id_node2 = cell_to_id_node[neighbour_pos]
						cp2 = reverse_dir(direction)
						if CardinalNode(id_node2, cp2) not in visited: 
							info.update({id_edge_counter: 
											 (CardinalNode(id_node1, cp1), 
											  CardinalNode(id_node2, cp2), 
											  edge_length)})
							cells_sequence.pop() # Don't include this node in the edge
							id_edge_to_cells.update({id_edge_counter: cells_sequence})
							id_edge_counter += 1
							visited.add(CardinalNode(id_node2, cp2))
						break
					edge_length += 1 # Not considering switches in the count
					# Update pos and dir
					pos = neighbour_pos
					exit_dir = reverse_dir(direction) 
					possible_transitions = np.array(env.rail.get_transitions(pos[0], pos[1], direction))
					possible_transitions[exit_dir] = 0 # Don't consider direction from which I entered
					# t = 2
					t = np.argmax(possible_transitions) # There's only one possible transition except the one that I took to get in
					temp_pos = get_new_position(pos, t)
					if 0 <= temp_pos[0] < env.height and 0 <= temp_pos[1] < env.width:	# Patch - check if this cell is a rail
						# Entrance dir is always opposite to exit dir
						direction = t
							
	# Build graph object made of vertices and edges
	edges = info.keys()
	
	train = train_on_graph(env, 0, nodes, edges, id_node_to_cell, id_edge_to_cells)
	paths, available_at = scheduling((nodes, edges), [train], info, connections, len(edges)) # TODO Rivedi params
	
	return nodes, edges # Graph as a tuple (list of vertices, list of edges)

def reverse_dir(direction):
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

# Tutti questi params andranno salvati nella classe TODO
def train_on_graph(env, handle, nodes, edges, id_node_to_cell, id_edge_to_cells):
	"""
	:param: handle: agent id
	:return: structure that represents a train on this graph ((id_edge, crossing_dir, dist) , (id_target_node, cp), speed)
	"""
	agent = env.agents[handle]
	agent_pos = agent.initial_position 
	dist = 0 # Distance already walked along the rail
	id_edge = None
	id_target_node = None
	crossing_dir = None
	for e in edges:
		# la dist Considera anche il tempo?
		index = [x for x, y in enumerate(id_edge_to_cells[e]) if y[0][0] == agent_pos[0] and y[0][1] == agent_pos[1]]
		if index: # If index is not empty
			dist = index[0]
			dir = id_edge_to_cells[e][dist][1]
			crossing_dir = 0 if dir == agent.direction else 1 # Direction saved is considered as crossing_dir = 0
			id_edge = e
			break
	
	# Determine target id
	for n in nodes:
		if id_node_to_cell[n] == agent.target:
			id_target_node = n
			break
	
	speed = agent.speed_data['speed']
	
	# cp depends on connections available in the target node
	return handle, (id_edge, crossing_dir, dist), (id_target_node, 1), speed # Prova