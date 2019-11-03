import numpy as np

'''
Function that assigns a priority to each agent in the environment, following a specified criterion
e.g. random with all distinct priorities 0..num_agents - 1
Input: number of agents in the current env
Output: np.array of priorities

'''
# TODO Improve


def assign_priority(num_agents):
    priorities = np.random.choice(range(num_agents), num_agents, replace=False)

    return priorities


if __name__ == "__main__":
    prios = assign_priority(5)
    print(prios)

