"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
import collections
from typing import Optional, List

import numpy as np


class GraphObsForRailEnv(ObservationBuilder):

    Node = collections.namedtuple('Node', 'node_position')

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> Node:
        return True

    def get(self, handle: int = 0) -> Node:
        return True

