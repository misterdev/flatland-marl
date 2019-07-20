from flatland.envs.generators import rail_from_manual_specifications_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
# cell_type is in [0..10] where
# 0: empty cell (e.g. a building)
# 1: straight
# 2: simple switch
# 3: diamond crossing
# 4: single slip
# 5: double slip
# 6: symmetrical
# 7: dead end
# 8: turn left
# 9: turn right
# 10: mirrored switch
specs = [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
         [(6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]

env = RailEnv(width=6,
              height=4,
              rail_generator=rail_from_manual_specifications_generator(specs),
              number_of_agents=1,  # agent position is chosen randomly
              obs_builder_object=TreeObsForRailEnv(max_depth=2))  # red squares mark the obs tree

env.reset()

env_renderer = RenderTool(env)
env_renderer.renderEnv(show=True)
env_renderer.renderEnv(show=True)


input("Press Enter to continue...")
