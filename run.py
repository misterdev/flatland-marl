import numpy as np
import time
import sys
import torch
from pathlib import Path
from importlib_resources import path

from flatland.evaluators.client import FlatlandRemoteClient  # For evaluation

from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv
from src.dueling_double_dqn import Agent
import src.nets

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

remote_client = FlatlandRemoteClient()  # Init remote client for eval

prediction_depth = 40
observation_builder = GraphObsForRailEnv(bfs_depth=4, predictor=ShortestPathPredictorForRailEnv(max_depth=prediction_depth))


state_size = prediction_depth + 3
network_action_size = 2
controller = Agent(state_size, network_action_size)
railenv_action_dict = dict()


with path(src.nets, "avoid_checkpoint100NEW.pth") as file_in:
    controller.qnetwork_local.load_state_dict(torch.load(file_in))
    
evaluation_number = 0
while True:
    
    evaluation_number += 1
    time_start = time.time()
    
    obs, info = remote_client.env_create(obs_builder_object=observation_builder)
    if not obs:
        break
    
    env_creation_time = time.time() - time_start
    
    print("Evaluation Number : {}".format(evaluation_number))

    local_env = remote_client.env
    number_of_agents = len(local_env.agents)
    
    time_taken_by_controller = []
    time_taken_per_step = []
    steps = 0
    
    while True:
        # Evaluation of a single episode
    
        time_start = time.time()
        # Pick actions
        for a in range(number_of_agents):
            network_action = controller.act(obs[a])
            railenv_action = observation_builder.choose_railenv_action(a, network_action)
            railenv_action_dict.update({a: railenv_action})
            
        time_taken = time.time() - time_start
        time_taken_by_controller.append(time_taken)

        time_start = time.time()
        # Perform env step
        obs, all_rewards, done, info = remote_client.env_step(railenv_action_dict)
        steps += 1
        time_taken = time.time() - time_start
        time_taken_per_step.append(time_taken)
        
        if done['__all__']:
            print("Reward : ", sum(list(all_rewards.values())))

            break

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("=" * 100)
    print("=" * 100)
    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)
    print("Number of Steps : ", steps)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(),
          np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("=" * 100)

print("Evaluation of all environments complete...")

print(remote_client.submit())