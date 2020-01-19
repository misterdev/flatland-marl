import math
from typing import Optional, Dict, List

from flatland.envs.distance_map import DistanceMap
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions, RailEnvNextAction
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_

from src.utils.types import WalkingElement

def get_shortest_paths(distance_map: DistanceMap, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) \
        -> Dict[int, Optional[List[WalkingElement]]]:
    """
    Computes the shortest path for each agent to its target and the action to be taken to do so.
    The paths are derived from a `DistanceMap`.

    If there is no path (rail disconnected), the path is given as None.
    The agent state (moving or not) and its speed are not taken into account

    example:
            agent_fixed_travel_paths = get_shortest_paths(env.distance_map, None, agent.handle)
            path = agent_fixed_travel_paths[agent.handle]

    Parameters
    ----------
    distance_map : reference to the distance_map
    max_depth : max path length, if the shortest path is longer, it will be cutted
    agent_handle : if set, the shortest for agent.handle will be returned , otherwise for all agents

    Returns
    -------
        Dict[int, Optional[List[WalkingElement]]]

    """
    shortest_paths = dict()

    def _shortest_path_for_agent(agent):
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            position = agent.target
        else:
            shortest_paths[agent.handle] = None
            return
        direction = agent.direction
        shortest_paths[agent.handle] = []
        distance = math.inf
        depth = 0

        while (position != agent.target and (max_depth is None or depth < max_depth)):
            next_actions = get_valid_move_actions_(
                direction, position, distance_map.rail)
            best_next_action = None
            for next_action in next_actions:
                next_action_distance = distance_map.get()[
                    agent.handle, next_action.next_position[0], next_action.next_position[
                        1], next_action.next_direction]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            shortest_paths[agent.handle].append(
                WalkingElement(position, direction, best_next_action))
            depth += 1

            # if there is no way to continue, the rail must be disconnected!
            # (or distance map is incorrect)
            if best_next_action is None:
                shortest_paths[agent.handle] = None
                return

            position = best_next_action.next_position
            direction = best_next_action.next_direction
        if max_depth is None or depth < max_depth:
            shortest_paths[agent.handle].append(
                WalkingElement(position, direction,
                                RailEnvNextAction(RailEnvActions.STOP_MOVING, position, direction)))

    if agent_handle is not None:
        _shortest_path_for_agent(distance_map.agents[agent_handle])
    else:
        for agent in distance_map.agents:
            _shortest_path_for_agent(agent)

    return shortest_paths
