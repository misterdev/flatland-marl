"""
A class to build observations for agents.
"""
import collections
from typing import Optional, List, Dict, Tuple, NamedTuple

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position, direction_to_point
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus

import src.utils.debug as debug


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
- Agents that are not departed yet see anyway all their path to the target on the bitmap.
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
		
		self.bitmaps = None
		self.recompute_bitmap = True

	def set_env(self, env: Environment):
		"""
	
		:param env: 
		:return: 
		"""
		super().set_env(env)
		if self.predictor:
			# Use set_env available in PredictionBuilder (parent class)
			self.predictor.set_env(self.env)

	def reset(self):
		"""
		
		:return: 
		"""
		self.cell_to_id_node = {}
		self.id_node_to_cell = {}
		self.connections = {}
		self.info = {}
		self.id_edge_to_cells = {}
		self.nodes = set()
		self.edges = set()
		self._map_to_graph()
		self.recompute_bitmap = True
		
	def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
		"""
		
		:param handles: 
		:return: 
		"""
		
		# Compute bitmaps from shortest paths
		if self.recompute_bitmap:
			self.prediction_dict = self.predictor.get()
			self.paths = self.predictor.shortest_paths
			self.cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)
			self.bitmaps = self._get_many_bitmap(handles=[a for a in range(self.env.get_num_agents())])
			self.recompute_bitmap = False
		
		observations = {}
		return observations

	def get_altmaps(self, handle):
		agent = self.env.agents[handle]
		altpaths, cells_seqs = self.predictor.get_altpaths(handle, self.cell_to_id_node)
		bitmaps = []
		for i in range(len(cells_seqs)):
			bitmap = self._bitmap_from_cells_seq(handle, cells_seqs[i])
			# We should have only 1
			# steps = int(1 / agent.speed_data['speed']) - 1
			# if steps > 0:
			# 	for s in range(steps):
			# 		bitmap[:, 0] = 0
			# 		bitmap = np.roll(bitmap, -1)

			# If agent not departed, add 0 at the beginning
			if agent.status == RailAgentStatus.READY_TO_DEPART:
				bitmap[:, -1] = 0
				bitmap = np.roll(bitmap, 1)

			bitmaps.append(bitmap)

		return bitmaps, altpaths

	def get_initial_bitmaps(self, print):
		"""
		Getter for bitmaps
		:return: 
		"""
		bitmaps = np.roll(self.bitmaps, 1)
		bitmaps[:, :, 0] = 0
		if print:
			debug.print_rails(self.env.height, self.env.height, self.id_node_to_cell, self.id_edge_to_cells)
			debug.print_cells_sequence(self.env.height, self.env.width, self.cells_sequence)
		return bitmaps

	def unroll_bitmap(self, handle, bitmaps):
		bitmaps[handle, :, 0] = 0
		bitmaps[handle] = np.roll(bitmaps[handle], -1)
		return bitmaps

	def get_agent_action(self, handle):
		agent = self.env.agents[handle]
		
		if agent.status == RailAgentStatus.READY_TO_DEPART:
			action = RailEnvActions.MOVE_FORWARD

		elif agent.status == RailAgentStatus.ACTIVE:
			# This can return None when rails are disconnected or there was an error in the DistanceMap
			if self.paths[handle] is None:  # Railway disrupted
				#TODO check if is None when rail disrupted
				action = RailEnvActions.STOP_MOVING
			else:
				# Get action
				step = self.paths[handle][0]
				next_action_element = step.next_action_element.action  # Get next_action_element

				assert step.position == agent.position
				# Just to use the correct form/name
				if next_action_element == 1:
					action = RailEnvActions.MOVE_LEFT
				elif next_action_element == 2:
					action = RailEnvActions.MOVE_FORWARD
				elif next_action_element == 3:
					action = RailEnvActions.MOVE_RIGHT
				
				self.paths[handle] = self.paths[handle][1:]

		else:  # If status == DONE
			action = RailEnvActions.DO_NOTHING

		return action

	def next_cell_occupied(self, handle):
		occupied = False
		
		if self.paths[handle][0]:
			next_pos = self.paths[handle][0].next_action_element.next_position
			for agent in self.env.agents:
				if agent.handle != handle and agent.position == next_pos:
					occupied = True
					break
		else:
			print('WHHHHAAAAT')

		return occupied

	def should_generate_altmaps(self, handle):
		if len(self.paths[handle]) > 0:
			curr_pos = self.env.agents[handle].position
			next_pos = self.paths[handle][0].next_action_element.next_position
			curr_rail, _ = self.get_edge_from_cell(curr_pos)
			next_rail, _ = self.get_edge_from_cell(next_pos)
			return curr_rail != -1 and next_rail == -1
		else:
			# This shouldn't happen, but it may happen
			return True

	def delay(self, handle, bitmaps, rail, direction, delay):
		bitmaps[handle] = np.roll(bitmaps[handle], delay)
		bitmaps[handle, rail, 0:delay] = direction
		return bitmaps

	# This is called when a train stops, every other train behind should be delayed
	def delay_schedule(self, handle, bitmaps, rail, direction):
		if bitmaps[handle, rail, 0] != 0:  # If agent is active
			others = self._get_trains_on_rails(bitmaps, rail, handle)
			# The current agent is before a switch
			first_time = 1
			for other in others:
				o, o_exit = other  #  other train (id), other exit
				o_speed = int(1 / self.env.agents[o].speed_data['speed'])
				if o_exit < first_time + o_speed: # If it's going to surpass this
					# Always delay of 1 cell (* speed)
					delay = first_time + o_speed - o_exit
					bitmaps = self.delay(o, bitmaps, rail, direction, delay)
					first_time += o_speed

		return bitmaps

	def update_bitmaps(self, a, network_action, bitmaps):
		curr_rail = np.argmax(np.absolute(bitmaps[a, :, 0]))
		curr_dir = bitmaps[a, curr_rail, 0]
		action = None
		crash = False

		agent = self.env.agents[a]
		if agent.status == RailAgentStatus.READY_TO_DEPART:
			is_occupied = False
			for i in range(len(self.env.agents)):
				if i != a and self.env.agents[i].position == agent.initial_position:
					is_occupied = True
			# If its initial position is occupied or he shouldn't depart 
			if ( network_action == 1 and is_occupied) or network_action == 0:
				action = RailEnvActions.STOP_MOVING
				return action, bitmaps, crash

		if network_action == 1:  # Go
			# print("Advancing", a)
			bitmaps[a, :, 0] = 0
			bitmaps[a] = np.roll(bitmaps[a], -1)
			# Find next rail and dir
			next_rail = np.argmax(np.absolute(bitmaps[a, :, 0]))
			next_dir = bitmaps[a, next_rail, 0]

			if bitmaps[a, next_rail, 0] == 0:
				print("Train {} has reached its target".format(a))
				assert agent.position == agent.target
				# TODO! adding action_required in main, you will never come here again
			else:
				# Check if rail is already occupied to compute new exit time
				last, last_exit = self._last_train_on_rail(bitmaps, next_rail, a)
				if last_exit > 0:
					last_dir = bitmaps[last, next_rail, 0]
					crash = last_dir != next_dir
					if crash:
						# print("{} CRASH with {}".format(a, last))
						bitmaps = self.delay_schedule(a, bitmaps, curr_rail, curr_dir)
						bitmaps[a] = np.roll(bitmaps[a], 1)
						bitmaps[a, curr_rail, 0] = curr_dir
						action = RailEnvActions.STOP_MOVING
					else:
						curr_exit_time = np.argmax(bitmaps[a, next_rail, :] == 0)
						if curr_exit_time <= last_exit:
							delay = last_exit + int(1 / self.env.agents[a].speed_data['speed']) - curr_exit_time
							bitmaps = self.delay(a, bitmaps, next_rail, next_dir, delay)

			if action == None:
				action = self.get_agent_action(a)
		else:
			# print("Waiting")
			action = RailEnvActions.STOP_MOVING
			bitmaps = self.delay_schedule(a, bitmaps, curr_rail, curr_dir)

		return action, bitmaps, crash

	def _bitmap_from_cells_seq(self, handle, path) -> np.ndarray:
		"""
		Compute bitmap for agent handle, given a selected path.
		:param handle: 
		:return: 
		"""
		bitmap = np.zeros((self.num_rails, self.max_time_steps + 1), dtype=int)  # Max steps in the future + current ts
		agent = self.env.agents[handle]
		# Truncate path in the future, after reaching target
		target_index = [i for i, pos in enumerate(path) if pos[0] == agent.target[0] and pos[1] == agent.target[1]]
		if len(target_index) != 0:
			target_index = target_index[0]
			path = path[:target_index + 1]

		# Add 0 at first ts - for 'not departed yet'
		rail, _ = self.get_edge_from_cell(path[0])

		# Agent's cardinal node, where it entered the last edge
		agent_entry_node = None
		# Calculate initial edge entry point
		i = 0
		rail, _ = self.get_edge_from_cell(path[i])
		if rail != -1: # If it's on an edge
			initial_rail = rail
			# Search first switch
			while rail != -1:
				i += 1
				rail, _ = self.get_edge_from_cell(path[i])

			src, dst, _ = self.info[initial_rail]
			node_id = self.cell_to_id_node[path[i]]
			# Reversed because we want the switch's cp
			entry_cp = self._reverse_dir(direction_to_point(path[i-1], path[i]))
			# If we reach the dst node
			if (node_id, entry_cp) == dst:
				# We entered from the src node (cross_dir = 1)
				agent_entry_node = src
			# Otherwise the opposite
			elif (node_id, entry_cp) == src: 
				agent_entry_node = dst
		else:
			#Handle the case you call this while on a switch before a rail
			node_id = self.cell_to_id_node[path[i]]
			# Calculate exit direction (that's the entry cp for the next edge)
			cp = direction_to_point(path[0], path[1]) # it's ok
			# Not reversed because it's already relative to a switch
			agent_entry_node = CardinalNode(node_id, cp)


		holes = 0
		# Fill rail occupancy according to predicted position at ts
		for ts in range(0, len(path)):
			cell = path[ts]
			# Find rail associated to cell
			rail, _ = self.get_edge_from_cell(cell)
			# Find crossing direction
			if rail == -1: # Agent is on a switch
				holes += 1
				# Skip duplicated cells (for agents with fractional speed)
				if cell != path[ts+1]:
					node_id = self.cell_to_id_node[cell]
					# Calculate exit direction (that's the entry cp for the next edge)
					cp = direction_to_point(cell, path[ts+1])
					# Not reversed because it's already relative to a switch
					agent_entry_node = CardinalNode(node_id, cp)
			else: # Agent is on a rail
				crossing_dir = None
				src, dst, _ = self.info[rail]
				if agent_entry_node == dst:
					crossing_dir = 1
				elif agent_entry_node == src: 
					crossing_dir = -1

				assert crossing_dir != None

				bitmap[rail, ts] = crossing_dir

				if holes > 0:
					bitmap[rail, ts-holes:ts] = crossing_dir
					holes = 0

		assert(holes == 0, "All the cells of the bitmap should be filled")

		temp = np.any(bitmap[:, 1:(len(path)-1)] != 0, axis=0)
		assert(np.all(temp), "Thee agent's bitmap shouldn't have holes ")
		return bitmap
		
	def _get_many_bitmap(self, handles: Optional[List[int]] = None) -> np.ndarray:
		"""
		This function computes the bitmaps and returns them, bitmaps are *strictly not* observations.
		:return: 
		"""
		bitmaps = np.zeros((len(handles), self.num_rails, self.max_time_steps + 1), dtype=int)
		# Stack bitmaps
		for a in range(len(handles)):
			bitmaps[a, :, :] = self._bitmap_from_cells_seq(a, self.cells_sequence[a])

		return bitmaps
	
	
	def _last_train_on_rail(self, maps, rail, handle):
		"""
		Find train preceding agent 'handle' on rail.
		:param maps: 
		:param rail: 
		:param handle: 
		:return: 
		"""
		l, l_exit = 0, 0 # Final train, its expected exit time
		
		for a in range(self.env.get_num_agents()):
			if not maps[a, rail, 0] == 0 and not a == handle:
				it = np.argmax(maps[a, rail, :] == 0)
				if it > l_exit: # If exit time of train a > my exit time
					l, l_exit = a, it
		return l, l_exit
	
	def _get_trains_on_rails(self, maps, rail, handle):
		"""

		:param maps: 
		:param rail: 
		:param handle: 
		:return: 
		"""
		trains = []
		for a in range(self.env.get_num_agents()):
			if not (maps[a, rail, 0] == 0 or a == handle):
				expected_exit_time = np.argmax(maps[a, rail, :] == 0) # Takes index/ts of last bit in a row
				trains.append((a, expected_exit_time))
		trains.sort()
		
		return trains
		
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
						t = np.argmax(possible_transitions)  # There's only one possible transition except the one that I took to get in
						temp_pos = get_new_position(pos, t)
						if 0 <= temp_pos[0] < self.env.height and 0 <= temp_pos[1] < self.env.width:  # Patch - check if this cell is a rail
							# Entrance dir is always opposite to exit dir
							direction = t
						else:
							break

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
			if cell in cells:
				return edge, cells.index(cell)

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