import math
from typing import Optional, Dict, List, Tuple

from flatland.envs.distance_map import DistanceMap
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions, RailEnvNextAction
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_

from src.utils.types import WalkingElement

def get_shortest_paths(distance_map: DistanceMap, max_depth: Optional[int] = None, agent_handle: Optional[int] = None,
        position: Optional[Tuple[int, int]] = None, direction: Optional[int] = None) -> Dict[int, Optional[List[WalkingElement]]]:
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

    def _shortest_path_for_agent(agent, a_position, a_direction):
        if a_position == None:
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                a_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                a_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                a_position = agent.target
            else:
                shortest_paths[agent.handle] = None
                return
        if a_direction == None:
            a_direction = agent.direction

        shortest_paths[agent.handle] = []
        distance = math.inf
        depth = 0

        while (a_position != agent.target and (max_depth is None or depth < max_depth)):
            next_actions = get_valid_move_actions_(
                a_direction, a_position, distance_map.rail)
            best_next_action = None
            for next_action in next_actions:
                next_action_distance = distance_map.get()[
                    agent.handle, next_action.next_position[0], next_action.next_position[
                        1], next_action.next_direction]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            shortest_paths[agent.handle].append(
                WalkingElement(a_position, a_direction, best_next_action))
            depth += 1

            # if there is no way to continue, the rail must be disconnected!
            # (or distance map is incorrect)
            if best_next_action is None:
                shortest_paths[agent.handle] = None
                return

            a_position = best_next_action.next_position
            a_direction = best_next_action.next_direction
        if max_depth is None or depth < max_depth:
            shortest_paths[agent.handle].append(
                WalkingElement(a_position, a_direction,
                                RailEnvNextAction(RailEnvActions.STOP_MOVING, a_position, a_direction)))

    if agent_handle is not None:
        _shortest_path_for_agent(distance_map.agents[agent_handle], position, direction)
    else:
        for agent in distance_map.agents:
            _shortest_path_for_agent(agent, position, direction)

    return shortest_paths

def get_altpaths(handle, distance_map, max_depth, cell_to_id_node): # TODO max_depth
    agent = distance_map.agents[handle]

    if agent.status == RailAgentStatus.READY_TO_DEPART:
        position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        position = agent.target
    else: #Agent arrived
        return []

    direction = agent.direction
    distance = math.inf
    depth = 0
    
    assert position != agent.target
    
    def get_altpaths_from(a_pos, a_dir): # TODO depth
        next_actions = get_valid_move_actions_(
            a_dir, a_pos, distance_map.rail)

        if a_pos in cell_to_id_node: # if this is a switch, recursive case
            paths = []
            for action in next_actions:
                first_step = WalkingElement(a_pos, a_dir,
                    RailEnvNextAction(action.action, action.next_position, action.next_direction))
                recursive_altpaths = get_altpaths_from(action.next_position, action.next_direction)

                for path in recursive_altpaths:
                    paths.append([first_step] + path)
            return paths
        else: #Move Forward, I'm on a rail
            assert len(next_actions) == 1
            
            # get shortest path
            for action in next_actions:
                first_step = WalkingElement(a_pos, a_dir,
                    RailEnvNextAction(action.action, action.next_position, action.next_direction))
            
            ret = []
            if a_pos == agent.target:
                ret = [[first_step]]
            else:
                shortest = get_shortest_paths(
                            distance_map=distance_map,
                            max_depth=max_depth,
                            agent_handle=handle,
                            position=action.next_position,
                            direction=action.next_direction)

                if shortest[handle] != None:
                    ret = [[first_step] + shortest[handle]]

            return ret

    next_actions = get_valid_move_actions_(
        direction, position, distance_map.rail)

    ret = []
    for action in next_actions:
        new_position = action.next_position
        new_direction = action.next_direction
        first_action = WalkingElement(position, direction,
            RailEnvNextAction(RailEnvActions.MOVE_FORWARD, new_position, new_direction))

        altpaths = get_altpaths_from(new_position, new_direction)

        for altpath in altpaths:
            ret.append([first_action] + altpath)

    return ret