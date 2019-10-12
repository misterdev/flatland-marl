import numpy as np
from flatland.utils import graphics_pil


# Given observations of one agent, perform preprocessing necessary before feeding to the NN
def preprocess_obs(obs, input_channels, env_width, env_height):
    agent_obs = np.concatenate((obs[0], obs[1], obs[2]), axis=2)
    # Reshape for PyTorch CNN - BCHW
    # from (env_width, env_height, channels=22) to (batch_size, channels=22, env_height, env_width)
    # agent_obs[a] = np.expand_dims(np.transpose(agent_obs[a], (2, 1, 0)), axis=0)
    agent_obs = np.transpose(agent_obs, (2, 1, 0))
    # Pad the array to have HxW = 200X200
    pad_agent_obs = np.zeros((input_channels, env_width, env_height)) # TODO check if order env_width/env_height is correct
    pad_agent_obs[:agent_obs.shape[0], :agent_obs.shape[1], :agent_obs.shape[2]] = agent_obs

    return pad_agent_obs

def convert_pil_to_nparray():

    pass

    transition_list = [int('0000000000000000', 2),  # empty cell - Case 0
                       int('1000000000100000', 2),  # Case 1 - straight
                       int('1001001000100000', 2),  # Case 2 - simple switch
                       int('1000010000100001', 2),  # Case 3 - diamond drossing
                       int('1001011000100001', 2),  # Case 4 - single slip
                       int('1100110000110011', 2),  # Case 5 - double slip
                       int('0101001000000010', 2),  # Case 6 - symmetrical
                       int('0010000000000000', 2),  # Case 7 - dead end
                       int('0100000000000010', 2),  # Case 1b (8)  - simple turn right
                       int('0001001000000000', 2),  # Case 1c (9)  - simple turn left
                       int('1100000000100010', 2)]  # Case 2b (10) - simple switch mirrored
    
# Given np.array of shape (env_width_ env_height, 16) convert to (env_width, env_height, 2) where
# the first channel encodes cell_types (0,.. 10) and the second channel orientation (0, 90, 180, 270 as 0 1 2 3)
# see rail_env_grid.py
def convert_transitions_map(transitions_map):
    
    new_transitions_map = np.zeros((transitions_map.shape[0], transitions_map.shape[1], 2))
    for i in range(transitions_map.shape[0]):
        for j in range(transitions_map.shape[1]):
            transition_bitmap = transitions_map[i, j]
            # Empty cell - cell_type 0, rotation 0
            if np.array_equal(transition_bitmap, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
                new_transitions_map[i, j] = np.array([0, 0])
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Straight - cell type 1
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Simple switch - cell type 2
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Diamond crossing - cell type 3
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Single slip - cell type 4
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Double slip - cell type 5
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Symmetrical - cell type 6
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Dead-end - cell type 7
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Simple turn right - cell type 8
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Simple turn left - cell type 9
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            # Simple switch mirrored - cell type 10
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())
            elif np.array_equal(transition_bitmap, np.array())


    pass
