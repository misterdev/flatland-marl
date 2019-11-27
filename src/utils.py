import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid_utils import coordinate_to_position, distance_on_rail
'''
Function that assigns a priority to each agent in the environment, following a specified criterion
e.g. random with all distinct priorities 0..num_agents - 1
Input: number of agents in the current env
Output: np.array of priorities

Ideas:
    - give priority to faster agents
    - give priority to agents closer to target
    - give priority to agent that have the shortest path "free", aka with less possible conflicts

'''

def assign_random_priority(num_agents):
    """
    
    :param num_agents: 
    :return: 
    """
    priorities = np.random.choice(range(num_agents), num_agents, replace=False)

    return priorities

def assign_speed_priority(agent):
    """
    Priority is assigned according to agent speed and it is fixed.
    max_priority: 1 (fast passenger train)
    min_priority: 1/4 (slow freight train)
    :param agent: 
    :return: 
    """
    priority = agent.speed_data.speed
    return priority

def assign_id_priority(handle):
    """
    Assign priority according to agent id (lower id means higher priority).
    :param agent: 
    :return: 
    """
    return handle

def assign_target_priority(handle):
    """
    Assign priority according to agent initial distance to target.
    :param handle: 
    :return: 
    
    return 
    """

def assign_priority(env, env_agent, conflict):
    """
    Assign priority in this way:
    - if agent is READY_TO_DEPART or DONE, return 0
    - if agent is ACTIVE:
        - if no conflict was predicted, return 0 (max prio)
        - if conflict was predicted, return priority as function of distance to target and speed
    :param env_agent: 
    :param conflict:
    :return: 
    """
    if env_agent.status is not RailAgentStatus.ACTIVE:
        return 0

    if not conflict:
        return 0
    else:
        max_distance = distance_on_rail((0,0), (env.height - 1, env.width - 1))
        min_distance = 0
        min_speed = 0.25
        priority = distance_on_rail(env_agent.position, env_agent.target) # Use Euclidean distance
        priority /= env_agent.speed_data['speed']
        # Normalize priority
        priority = priority / (max_distance / min_speed)
        return priority
    

def preprocess_obs(obs):
    """Preprocess local observations before feeding to the conv network"""
    # Concatenate info about rail, agent and targets
    agent_obs = np.concatenate((obs[0], obs[1], obs[2]), axis=2)
    # Reshape for PyTorch CNN - BCHW
    # from (env_width, env_height, in_channels=22) to (batch_size, in_channels=22, env_height, env_width)
    # agent_obs[a] = np.expand_dims(np.transpose(agent_obs[a], (2, 1, 0)), axis=0)
    agent_obs = np.transpose(agent_obs, (2, 1, 0))
    
    return agent_obs

