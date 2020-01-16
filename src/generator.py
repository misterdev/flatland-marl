import time
import numpy as np

#we consider a simple network with three rails and two switches 
#        ________
#       /        \
#      0----------1
#       \________/
#
# we consider three trains.
# for each train we generate a random path composed of a random
# number of rails, between 3 and 5
# at the switch the train can invert direction along any rail
# the speed of the trains is fixed: 1,2 and 4 (inverse of the speed)
# the length of the rails is fixed too: 2, 3 and 4

# Generate random path along edges 3,4,5
def gen_path():
	n = np.random.randint(3,high=6)
	path = []
	path.append(np.random.randint(3))
	for i in range(0,n-1):
		new = np.random.randint(3)
		while new == path[-1]: # Until it doesn't change rail
			new = np.random.randint(3)
		path.append(new)
	return path

no_trains = 3
speeds = [1,2,4]
blengths = [2,3,4]

paths = []
# Generate paths from source station/node (random) for each agent
for i in range(no_trains):
	station = np.random.randint(2)
	paths.append((station,gen_path()))

print("Paths from station (0/1) for each agent:")
print(paths)

# generate the initial map of size trains_no x rails_no x time
# in our case 3x3x80

def gen_maps():
	maps = np.zeros((no_trains,3,80)).astype(int)
	for i in range(no_trains):
		speed = speeds[i]
		station,path = paths[i]
		if station == 1: # Defines direction along edges according to initial direction
			dir = -1
		else:
			dir = 1
		pos = 1
		while not path==[]:
			bin = path[0] # Get first rail in the path
			blen = blengths[bin]
			path = path[1:] # Cut path
			last_pos = pos+blen*speed
			maps[i,bin,pos:last_pos]=dir
			pos = last_pos
			dir = -dir
	return maps

# find the train preceding me on rail r
def last_train_on_rail(maps,r,me):
	print("check last train on rail")
	ft,tt = 0,0 #train, expected exit time
	print(maps[:,r,0]) # Trains that are on this rail at ts=0
	for i in range(0,no_trains):
		# Loop through trains on rail (except myself)
		if not (maps[i,r,0] == 0) and not(i==me):
			#print(i) # argmax o argmin ???
			it = np.argmax(maps[i,r,:]==0) # Exit time of train i on this rail
			#print("argmax {}".format(it))
			print("Train {} exits at time {}".format(i, it))
			if it > tt: # If exit time of i is greater than mine, takes it
				ft,tt = i,it
	return (ft,tt)

#find all trains others than me on rail r ordered by expected exit time
def all_trains_on_rail(maps,r,me):
	all = []
	print("Trains on rail {} at ts=0".format(r))
	print(maps[:,r,0]) # Trains on r at ts=0
	for i in range(0,no_trains):
		if not (maps[i,r,0] == 0 or i==me):
			exp_exit = np.argmax(maps[i,r,:]==0) # Takes index=ts of last bit in a row
			all.append((exp_exit,i))
	all.sort() # Sort in ascending order
	return all

def random_move(maps, t):
	"""
	Perform random move and update the map.
	:param maps: (no_trains x no_rails x max_time_steps)
	:param t: train
	:return: 
	"""
	#a train may decide its move only at the end of a rail;
	#in the middle of a rail it can only advance
	current_rail = np.argmax(np.absolute(maps[t,:,0]))
	current_dir = maps[t,current_rail,0]
	if maps[t,current_rail,0]==0: # The first el is 0 for an agent READY_TO_DEPART
		print("train {} ready to start".format(t))
	else:
		print("train {} on rail {} in direction {}".format(t,current_rail,current_dir))
		assert(maps[t,current_rail,1]==0)
	
	# My idea: (not sure where to insert the alternative path selection)
	# Feeding the network with a subset of bitmaps relative to me and conflicting agents
	# The network can choose among the 0/1
	# If this move (1) led to a crash, we undo move on bitmap and select another path to feed to the network
	######## OR #########
	# We feed the network already with bitmaps from alternative paths (following the dir 1/2/3)
	# The network chooses in [0, 3] and that's it, if we crash, game over
	a = np.random.randint(2)
	if a == 1:
		print("advancing")
		#we just roll the map relative to train t
		maps[t,:,0]=0
		# Roll and not shift so at the end we have 0s at the beginning (target reached)
		# This assumes that 2000 ts are enough to compute paths that lead to targets (??)
		maps[t] = np.roll(maps[t],-1)
		new_rail = np.argmax(np.absolute(maps[t,:,0]))
		new_dir = maps[t,new_rail,0]
		if maps[t,new_rail,0]==0:
			print("train {} completed".format(t))
		else:
			print("now on rail {} in direction {}".format(new_rail,new_dir))
			# Check if rail is already occupied - to compute new exit time
			lt,tt = last_train_on_rail(maps,new_rail,t) # Get preceding trail and its exit time
			#print("found {}".format(lt,tt))
			if tt>0:
				dir2 = maps[lt,new_rail,0]
				if not(dir2==new_dir):
					print("CRASH with {}".format(lt))
					print("undo move")
					#we undo our move and unroll the map
					maps[t] = np.roll(maps[t],1)
					maps[t,current_rail,0]=current_dir
				#t is not the only one
				else:
					t_time = np.argmax(maps[t,new_rail,:]==0)
					if t_time <= tt:
						delay = tt+speeds[t]-t_time
						maps[t] = np.roll(maps[t],delay)
						maps[t,new_rail,0:delay] = new_dir
						print("following {} with delay {}".format(lt,delay))
					else:
						print("following {} with no delay".format(lt))
			else:
				print("new rail is free")
	else:
		print("waiting")
		#if t has not yet started or already completed its path there is
		#nothing to do
		#otherwise, if t decided to wait it could delay other trains on
		#the same rail
		if not(maps[t,current_rail,0]==0):
			others = all_trains_on_rail(maps,current_rail,t)
			print("other trains on rail {}: {}".format(current_rail,others))
			first_time = 1
			for other in others:
				oe,ot = other # other exit, other time ???
				ospeed = speeds[ot]
				if oe < first_time + ospeed:
					delay = first_time + ospeed - oe
					maps[ot] = np.roll(maps[ot],delay)
					maps[ot,current_rail,0:delay] = current_dir
					print("train {} delayed of {}".format(ot,delay))
					first_time += ospeed
	return maps

# Generate bitmaps for every train
maps = gen_maps()
print("Bitmaps at the very beginning")
print(maps[:,:,0:16])

step=0
# Until all maps are empty (all targets have been reached) or max time steps was reached
while not np.all(maps==0) and step<2000:
	print("############################ step {} ############################".format(step))
	for t in range(0,no_trains):
		#print("Train {}".format(t))
		if np.all(maps[t,:,0]==maps[t,:,1]): # Two consecutive bits are the same, I'm on a rail
			print("train {} is advancing".format(t))
			# Roll map when train advances
			maps[t,:,0]=0
			maps[t] = np.roll(maps[t],-1)
		else:
			maps = random_move(maps,t)
	print("Bitmap at the end of this step:")
	print(maps[:,:,0:16])
	#time.sleep(5)
	step += 1


"""
Comments:
- move should be computed by the network for each agent, starting from the first (not all together)
generating alternative paths we can evaluate each move and take the best (greedy).
Considering only one agent at a time we can feed to the network also info regarding a small subset of trains
that are conflicting with that agent (feed paths that are 'highly conflicting' so that the network can learn to solve
the most problematic situations);
- in this map switches are not considered at all;

"""


    
