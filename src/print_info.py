def print_info(env):
    print("\n Agents in the environment have to solve the following tasks: \n")
    for agent_idx, agent in enumerate(env.agents):
        print(
            "The agent with index {} has the task to go from its initial position {}, facing in the direction {} to its target at {}.".format(
                agent_idx, agent.initial_position, agent.direction, agent.target))

    # Agent status: READY_TO_DEPART, ACTIVE, or DONE
    print("\n Their current statuses are:")
    print("============================")

    for agent_idx, agent in enumerate(env.agents):
        print("Agent {} status is: {} with its current position being {}".format(agent_idx, str(agent.status),
                                                                                 str(agent.position)))

    # The agent needs to take any action [1,2,3] except do_nothing or stop to enter the level
    # If the starting cell is free they will enter the level
    # If multiple agents want to enter the same cell at the same time the lower index agent will enter first.

    # Let's check if there are any agents with the same start location
    agents_with_same_start = set()
    print("\n The following agents have the same initial position:")
    print("=====================================================")
    for agent_idx, agent in enumerate(env.agents):
        for agent_2_idx, agent2 in enumerate(env.agents):
            if agent_idx != agent_2_idx and agent.initial_position == agent2.initial_position:
                print("Agent {} as the same initial position as agent {}".format(agent_idx, agent_2_idx))
                agents_with_same_start.add(agent_idx)

    # Lets try to enter with all of these agents at the same time
    action_dict = dict()

    for agent_id in agents_with_same_start:
        action_dict[agent_id] = 1  # Try to move with the agents

    # Do a step in the environment to see what agents entered:
    obs, rewards, done, infos = env.step(action_dict)

    # Speed
    print("\n The speed information of the agents are:")
    print("=========================================")

    for agent_idx, agent in enumerate(env.agents):
        print(
            "Agent {} speed is: {:.2f} with the current fractional position being {}".format(
                agent_idx, agent.speed_data['speed'], agent.speed_data['position_fraction']))

    # Malfunctions
    print("\n The malfunction data of the agents are:")
    print("========================================")

    for agent_idx, agent in enumerate(env.agents):
        print(
            "Agent {} is OK = {}".format(
                agent_idx, agent.malfunction_data['malfunction'] < 1))
        
    '''

    # Which agents needs to pick and action
    print("\n The following agents can register an action:")
    print("========================================")
    for info in infos['action_required']:
        print("Agent {} needs to submit an action.".format(info))

    '''