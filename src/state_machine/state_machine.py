import numpy as np
from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

# TODO Add check for status
# TODO Add 2nd shortest path
# Only for active agents
def act(args, state):
	
	# Case 1 - Path is free, go
	if state[0] == 0:
		# Go	
		return 0
	# Case 2 - in front of an overlapping span
	elif state[0] == 1 and state[args.prediction_depth] == 0:
		# Go if has priority, should consider prio in first span? prio like this doesn't work because of wrong conflicting agents TODO 
		if state[args.prediction_depth*4] < state[args.prediction_depth*4 + 1]:
			return 0
		else:		
			if state[args.prediction_depth*2] == 1: # Means I'm on a fork - check possible alternatives to shortest path
				return np.random.choice((1, 2, 3)) # TODO
			else:
				return 1
			
	# Case 3 
	elif state[0] == 1 and state[args.prediction_depth] == 1:
		# Stop
		return 1
