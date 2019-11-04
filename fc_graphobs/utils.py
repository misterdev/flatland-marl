import numpy as np

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
# TODO Improve


def assign_random_priority(num_agents):
    priorities = np.random.choice(range(num_agents), num_agents, replace=False)

    return priorities


'''
Priority is assigned according to agent speed and it is fixed.
max_priority: 1 (fast passenger train)
min_priority: 1/4 (slow freight train)

'''
# TODO ???

def assign_speed_priority(agent):

    priority = agent.speed_data.speed
    return priority


if __name__ == "__main__":
    prios = assign_random_priority(5)
    print(prios)

