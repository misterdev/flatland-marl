import numpy as np
import time
#import bisect

#un nodo e' definito da un id. Una mappa connections associa all'id un array di 0 o 1
#con le posisibili connessione NE, NS, NO, ES, EO, SO (array 4*4)

#un arco (non orientato) e' un id, con informazioni relative a ((id1,c1),(id2,c2),lunghezza)
# dove c1 e c2 sono N,E,SU,O

#ogni treno ha un suo id, una posizione definita da
# - un binario (un arco)
# - una direzione di attraversamento (0,1)
# - la distanza percorsa sul binario (int)
# una velocita
# una destinazione (id del nodo)
# altre info

def insert(a, frontier, visited):
    """
    
    :param a: 
    :param frontier: 
    :param visited: 
    :return: 
    """
    (t, ct), length, tim, path = a
    if (t,ct) in visited:
        #print("visited : {}".format((t,ct)))
        return frontier
    elif frontier==[]:
        return([a])
    else:
        new_frontier = []
        (x,xc), xlength, xtime, xpath = frontier[0]
        tail = frontier[1:]
        while xlength < length and not(tail==[]):
            new_frontier.append(((x,xc), xlength, xtime, xpath))
            (x,xc), xlength, xtime, xpath = tail[0]
            tail = tail[1:]
        if xlength < length: 
            new_frontier.append(((x,xc), xlength, xtime, xpath))
        else:
            tail.append(((x,xc), xlength, xtime, xpath))  #put back in tail
        new_frontier.append(((t,ct), length, tim, path))
        new_frontier = new_frontier + tail
        
        return new_frontier

def insert_wrt_time(a, frontier, visited):
    """
    
    :param a: 
    :param frontier: 
    :param visited: 
    :return: 
    """
    (t,ct), length, tim, path = a
    if (t,ct) in visited:
        #print("visited : {}".format((t,ct)))
        return frontier
    elif frontier==[]:
        return [a]
    else:
        new_frontier = []
        (x,xc), xlength, xtime, xpath = frontier[0]
        tail = frontier[1:]
        while xtime < tim and not(tail==[]):
            new_frontier.append(((x,xc), xlength, xtime, xpath))
            (x,xc), xlength, xtime, xpath = tail[0]
            tail = tail[1:]
        if xtime < tim: 
            new_frontier.append(((x,xc),xlength,xtime,xpath))
        else:
            tail.append(((x,xc), xlength, xtime, xpath))  #put back in tail
        new_frontier.append(((t,ct),length,tim,path))
        new_frontier = new_frontier + tail
        return new_frontier


def find_shortest_paths(G, train, available_at, info, connections):

    # available_at is a map associating to each rail the timestep
    # at which it will be available and the current traversal direction.
    # We keep the invariant that if a
    # rail is available at time t0, it will be available at any time
    # t > t0 (with the current scheduling)
    # The greedy policy books, for each train, ALL rails along its path,
    # until its transit on the rail

    V, E = G
    shortest_paths = []
    id_train, pos, target, speed = train
    for cp in target[1]:  # Build a shortest path for each possible entry point of the target node
        # print("train id: {}".format(id_train))
        rail, direction, dist = pos
        (s, cs), (t, ct), l = info[rail]

        rail_exit_time, rail_dir = available_at[rail]
        # if the train already on a rail, the two directions must coincide
        assert (direction == rail_dir)

        current_length = l - dist
        # current_time = int(current_length/speed) + available_at[rail]
        # rail_exit_time is the maximum exit time of all trains currently on the rail
        current_time = rail_exit_time
        current_path = [(
            rail,  # id del binario
            direction,  # direzione di percorrenza
            current_time,  # at exit time the rail will be availbale again
        )]
        visited = []
        frontier = []
        if direction == 1:
            current_node, current_c = t, ct
        else:
            current_node, current_c = s, cs

        # invariante: mantengo la frontier ordinata rispetto alle lunghezze correnti
        # todo = [n for n in V if not(n==target_node)]
        # print("ct before while {}",current_time)
        while not (current_node, current_c) == (target[0], cp):
            # print(frontier)
            # time.sleep(1)
            visited.append((current_node, current_c))
            # cerco i binari adiacenti al current_node
            for e in E:
                (s, cs), (t, ct), l = info[e]
                if ((s == current_node and connections[current_node][current_c, cs] == 1) or
                        (t == current_node and connections[current_node][current_c, ct] == 1)):
                    if s == current_node:
                        train_dir = 1
                        d, cd = t, ct  # destination is target                      
                    else:
                        train_dir = 0
                        d, cd = s, cs  # destination is source      
                    new_length = current_length + l
                    new_path = list(current_path)  # make a copy of the current path
                    rail_exit_time, rail_dir = available_at[e]
                    # if we want to use this rail, we need to wait until it is available;
                    # the exit time depend from the "current" rail direction
                    if train_dir == rail_dir:
                        # time at which I could reac the exit
                        potential_transit_time = new_current_time + int(l / speed)
                        # we keep the maximum between the potential_tyransit_time and
                        # the current trial exit time +1, in case we need to follow a line
                        transit_time = max(potential_transit_time, rail_exit_time + 1)
                        # destination and its connection
                    else:
                        # wait until available in my direction
                        new_current_time = max(current_time, rail_exit_time)
                        transit_time = new_current_time + int(l / speed)
                    new_path.append((e, dir, transit_time))
                    insert_wrt_time(((d, cd), new_length, transit_time, new_path), frontier, visited)
            if not frontier:
                # print("No path found")
                current_path = []  # empty path means failure
                break
            else:
                (current_node, current_c), current_length, current_time, current_path = frontier[0]
                # print("current from frontier = {}", current_time)
                frontier = frontier[1:]

        shortest_paths.append((current_length, current_path))
        
    # Return path with minimum length
    index = 0
    min_length = shortest_paths[0][0]
    for i in range(1, len(shortest_paths)):
        if shortest_paths[i][0] < min_length:
            min_length = shortest_paths[i][0]
            index = i

    return shortest_paths[index]

def update_availability(available_at, path):
    """
    
    :param available_at: 
    :param path: 
    :return: 
    """
    for (id_edge, crossing_dir, transit_time) in path[1]:
        available_at[id_edge] = (transit_time, crossing_dir)
        
    return available_at

def scheduling(G, trains, info, connections):
    """
    
    :param G: 
    :param trains: 
    :param info: 
    :param connections: 
    :return: paths: , available_at: list of times at which edges will be available again (index in list corresponds to edge id)
    """
    paths = []
    num_rails = len(G[1]) # G[0] = vertices, G[1] = edges
    available_at = np.zeros((num_rails, 2), dtype=int) # Contains num_rails tuples of 2 elements (transit_time, crossing_dir)
    # Initialize available_at considering initial positions of trains
    for t in trains:
        id_edge = t[1][0]
        crossing_dir = t[1][1]
        dist = t[1][2]
        speed = t[3]
        # Compute total edge length to know how much time left till exit
         edge_length = # TODO 
        exit_time = edge_length -  dist / speed 
        available_at[id_edge] = (exit_time, crossing_dir)
    
    # TODO Agents order matters
    for train in trains:
        path = find_shortest_paths(G, train, available_at, info, connections)
        paths.append(path)
        available_at = update_availability(available_at, path)
        
    return paths, available_at

'''
train = (0, #id del treno
         (0,1,0), #pos: id del binario, direzione, distanza percorsa
         (2,2), #target: nodo e punto cardinale
         1 #speed
         )

train2 = (1, #id del treno
         (0,1,0), #pos: id del binario, direzione, distanza percorsa
         (0,0), #target: nodo e punto cardinale
         .25 #speed
         )
'''