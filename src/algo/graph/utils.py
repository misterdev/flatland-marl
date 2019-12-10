import numpy as np
from typing import NamedTuple
from flatland.core.grid.grid4_utils import get_new_position, get_direction, direction_to_point
from flatland.core.grid.rail_env_grid import RailEnvTransitions

CardinalNode = \
	NamedTuple('CardinalNode',[('id_node', int), ('cardinal_point', int)])

def map_to_graph(env):
	
	# Retrieve nodes
	id_node_counter = 0
	cell_to_id_node = {} # Map cell position : id_node
	id_node_to_cell = {} # Map id_node to cell position
	# Get bitmap and identify cells hat are nodes (have switches)
	bitmap = np.zeros((env.height, env.width, 16))
	connections = {} # Map id_node : connections(node)
	for i in range(bitmap.shape[0]):
		for j in range(bitmap.shape[1]):
			'''
			* This is another way to do it *
			
			bitlist = [int(digit) for digit in bin(env.rail.get_full_transitions(i, j))[2:]]
			bitlist = [0] * (16 - len(bitlist)) + bitlist
			bitmap[i, j] = np.array(bitlist)
			
			orientation = 0 # North
			for k in range(bitmap.shape[2] - 4): # 16 - 4
				if k > 3 and k % 4 == 0: # Change to following group of bits
					orientation += 1
				if bitmap[i, j, k] == 1 # esiste una connessione SN
			'''
			is_switch = False	
			is_crossing = False
			connections_matrix = np.zeros((4, 4)) # Matrix NESW x NESW
			
			# Check if diamond crossing
			transitions_bit = bin(env.rail.get_full_transitions(i, j))
			if int(transitions_bit, 2) == int('1000010000100001', 2):
				is_crossing = True
				connections_matrix[0, 2] = connections_matrix[2, 0] = 1
				connections_matrix[1, 3] = connections_matrix[3, 1] = 1
			else:
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
						
			if is_switch or is_crossing:
				# Add node - keep info on cell position
				# Update only for nodes that are switches
				connections.update({id_node_counter: connections_matrix})
				id_node_to_cell.update({id_node_counter: (i, j)})
				cell_to_id_node.update({(i, j): id_node_counter})
				id_node_counter += 1
			
	# Enumerate edges from these nodes
	info = {} # Map id_edge : tuple (CardinalNode1, CardinalNode2, edge_length)
	id_edge_counter = 0
	# Start from connections of one node and follow path until next switch is found
	nodes = connections.keys() # ids
	visited = set()  # Keeps set of CardinalNodes that were already visited
	for n in nodes:
		for cp in range(4): # Check edges from the 4 cardinal points
			if np.count_nonzero(connections[n][cp, :]) > 0:	
				node_found = False
				edge_length = 0
				# Keep going until another node is found
				direction = cp
				pos = id_node_to_cell[n]
				while not node_found:
					neighbour_pos = get_new_position(pos, direction)
					if neighbour_pos in cell_to_id_node:
						# node_found = True
						# Build edge, mark visited
						# id_edge, (id_node1, cp1), (id_node2, cp2), length 
						info.update({id_edge_counter: 
										 (CardinalNode(n, cp), 
										  CardinalNode(cell_to_id_node[neighbour_pos], reverse_dir(direction)), 
										  edge_length)})
						id_edge_counter += 1
						visited.add(CardinalNode(n, cp))
						visited.add(CardinalNode(cell_to_id_node[neighbour_pos], reverse_dir(direction)))
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
							
	# Build graph object made of vertices and edges TODO
	print('End')

def reverse_dir(direction):
	"""
	Invert direction (int) of one agent.
	:param direction: 
	:return: 
	"""
	return int((direction + 2) % 4)