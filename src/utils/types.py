
from typing import NamedTuple, Tuple
from flatland.envs.rail_env import RailEnvNextAction

WalkingElement = \
    NamedTuple('WalkingElement',
               [('position', Tuple[int, int]), ('direction', int), ('next_action_element', RailEnvNextAction)])

# Flatland v3
# A way point is the entry into a cell defined by
# - the row and column coordinates of the cell entered
# - direction, in which the agent is facing to enter the cell.
# This induces a graph on top of the FLATland cells:
# - four possible way points per cell
# - edges are the possible transitions in the cell.
Waypoint = NamedTuple(
    'Waypoint', [('position', Tuple[int, int]), ('direction', int)])