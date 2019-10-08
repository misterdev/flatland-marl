import numpy as np


# Given observations of one agent, perform preprocessing necessary before feeding to the NN
def preprocess_obs(obs):
    agent_obs = np.concatenate((obs[0], obs[1], obs[2]), axis=2)
    # Reshape for PyTorch CNN - BCHW
    # from (env_width, env_height, channels=22) to (batch_size, channels=22, env_height, env_width)
    # agent_obs[a] = np.expand_dims(np.transpose(agent_obs[a], (2, 1, 0)), axis=0)
    agent_obs = np.transpose(agent_obs, (2, 1, 0))
    # Pad the array to have HxW = 200X200
    pad_agent_obs = np.zeros((22, 200, 200))
    pad_agent_obs[:agent_obs.shape[0], :agent_obs.shape[1], :agent_obs.shape[2]] = agent_obs

    return pad_agent_obs
