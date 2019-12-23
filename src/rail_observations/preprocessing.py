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