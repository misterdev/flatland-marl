import numpy as np
from flatland.utils import graphics_pil
from flatland.core.grid.rail_env_grid import RailEnvTransitions

# TODO Do the conversion inside an ObserverObject directly...
# Given observations of one agent, perform preprocessing necessary before feeding to the NN
def preprocess_obs(obs, input_channels, max_env_width, max_env_height):
    
    # Convert 16-channel for rail obs to 2-channel
    two_channel_obs = convert_transitions_map(obs[0])
    # Concatenate info about rail, agent and targets
    agent_obs = np.concatenate((two_channel_obs, obs[1], obs[2]), axis=2)
    # Reshape for PyTorch CNN - BCHW
    # from (env_width, env_height, in_channels=22) to (batch_size, in_channels=22, env_height, env_width)
    # agent_obs[a] = np.expand_dims(np.transpose(agent_obs[a], (2, 1, 0)), axis=0)
    agent_obs = np.transpose(agent_obs, (2, 1, 0))
    # Pad the array to have HxW = 200X200
    pad_agent_obs = np.zeros((input_channels, max_env_width, max_env_height)) # TODO check if order env_width/env_height is correct
    pad_agent_obs[:agent_obs.shape[0], :agent_obs.shape[1], :agent_obs.shape[2]] = agent_obs

    return pad_agent_obs

def convert_pil_to_nparray():

    pass


# Given transitions list considering cell types outputs all possible transitions bitmap considering cell rotations too
def compute_all_possible_transitions():
    
    # Bitmaps are read in decimal numbers
    transitions = RailEnvTransitions()
    transition_list = transitions.transition_list
    '''
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
    '''

    transitions_with_rotation_dict = {}
    rotation_degrees = [0, 90, 180, 270]

    for i in range(len(transition_list)):
        for r in rotation_degrees:
            t = transition_list[i]
            rot_transition = transitions.rotate_transition(t, r)
            if rot_transition not in transitions_with_rotation_dict:
                transitions_with_rotation_dict[rot_transition] = np.array([i, r])

    return transitions_with_rotation_dict


# Given np.array of shape (env_width_ env_height, 16) convert to (env_width, env_height, 2) where
# the first channel encodes cell_types (0,.. 10) and the second channel orientation (0, 90, 180, 270 as 0 1 2 3)
# see rail_env_grid.py
def convert_transitions_map(obs_transitions_map):

    new_transitions_map = np.zeros((obs_transitions_map.shape[0], obs_transitions_map.shape[1], 2))
    possible_transitions_dict = compute_all_possible_transitions()

    for i in range(obs_transitions_map.shape[0]):
        for j in range(obs_transitions_map.shape[1]):
            transition_bitmap = obs_transitions_map[i, j]
            # Convert bitmap to int binario
            int_transition_bitmap = int(transition_bitmap.dot(2 ** np.arange(transition_bitmap.size)[::-1]))
            new_transitions_map[i, j] = possible_transitions_dict[int_transition_bitmap]

    return new_transitions_map
