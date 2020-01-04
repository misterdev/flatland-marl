import numpy as np

def fill_padding(obs, max_rails):
    """
    
    :param obs: Agent state 
    :param max_rails: Maximum number of rails in environment 
    :return: Observation padded with 0s along first axis (until max_rails)
    
    """
    prediction_depth = obs.shape[1]
    
    pad_agent_obs = np.zeros((max_rails, prediction_depth))
    pad_agent_obs[:obs.shape[0], :obs.shape[1]] = obs
    
    return pad_agent_obs

def choose_subset(handle, bitmaps, max_num):
    """
    Select a subset of bitmaps belonging to 'most conflicting agents'.
    :param: handle: agent's handle
    :param bitmaps: all the bitmaps
    :param max_num: max number of bitmaps/agents to consider in the subset
    :return: 
    """
    num_agents = bitmaps.shape[0]
    num_rails = bitmaps.shape[1]
    num_steps = bitmaps.shape[2]
    conflicts_per_agents = {a: 0 for a in range(num_agents)} # Add here count of conflicts per agent
    # Find 'most' conflicting agents - naive way
    # Debug
    #for a in range(num_agents):
        #print(bitmaps[a, :, :])
    for ts in range(1, num_steps):
        rail_ts = np.argmax(np.absolute(bitmaps[handle, : , ts])) # Find agent current rail
        handle_dir_ts = bitmaps[handle, rail_ts, ts] # Find current direction on this rail
        conflicting_agents = np.where(bitmaps[:, rail_ts, ts] == handle_dir_ts)
        
        for a in conflicting_agents[0]:
            if a != handle:
                conflicts_per_agents[a] += 1
    # Order dicts per number of conflicts
    sorted_dict = {k: v for k, v in sorted(conflicts_per_agents.items(), key=lambda item: item[1])}
    max_num_keys = [key for key, i in enumerate(sorted_dict.keys(), 0) if i < max_num]
    # Return bitmaps relative to max_num most conflicting agents - the max_num top of the dictionary
    # bitmaps_subset = np.empty((max_num, num_rails, num_steps), dtype=int)
    bitmaps_subset = np.stack([bitmaps[a, :, :] for a in max_num_keys], axis=0)
    
    return bitmaps_subset

def preprocess_obs(handle, bitmaps, max_conflicting_agents, max_rails):
    """
    
    :param handle: 
    :param bitmaps: 
    :param max_conflicting_agents: 
    :param max_rails: 
    :return: 
    """
    # Select subset of conflicting paths in bitmap
    state = choose_subset(handle, bitmaps, max_conflicting_agents)
    # Pad until max rails
    # Stack to add an axis
    # state = np.stack([fill_padding(state[a,:, :], max_rails) for a in range(max_conflicting_agents)], axis=0)
    # Or simply concatenate
    state = np.concatenate([fill_padding(state[a,:, :], max_rails) for a in range(max_conflicting_agents)], axis=0)
    # PyTorch CNN - BCHW
    #state = np.transpose(state, (1, 0))
    
    return state # (prediction_depth + 1, max_cas * max_rails)