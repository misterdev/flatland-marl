import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.seed import seed as set_seed

from flatland.envs.rail_generators import sparse_rail_generator, random_rail_generator
from flatland.envs.rail_env import RailEnv


class RailEnvRLLibWrapper(MultiAgentEnv):

    def __init__(self, config):

        super(MultiAgentEnv, self).__init__()

        # Environment ID if num_envs_per_worker > 1
        if hasattr(config, "vector_index"):
            vector_index = config.vector_index
        else:
            vector_index = 1

        self.predefined_env = False

        if config['rail_generator'] == "sparse_rail_generator":
            self.rail_generator = sparse_rail_generator(max_num_cities=config['max_num_cities'],
                                                        seed=config['seed'],
                                                        grid_mode=config['grid_mode'],
                                                        max_rails_between_cities=config['max_rails_between_cities'],
                                                        max_rails_in_city=config['max_rails_in_city'])

        elif config['rail_generator'] == "load_env":
            self.predefined_env = True
            self.rail_generator = random_rail_generator() # Just a placeholder
        else:
            raise (ValueError, f'Unknown rail generator: {config["rail_generator"]}')

        set_seed(config['seed'] * (1 + vector_index))
        # Initialize the "usual" RailEnv from flatland
        # TODO Update
        self.env = RailEnv(width=config["width"], 
                           height=config["height"],
                           rail_generator=self.rail_generator,
                           schedule_generator=self.schedule_generator, # config schedule_generator TODO
                           number_of_agents=config["number_of_agents"],
                           obs_builder_object=config['obs_builder'],
                           max_episode_steps=config['max_episode_steps'],
                           stochastic_data=config["stochastic_data"],
                           remove_agents_at_target=config['remove_agents_at_targets']
                           )
        # TODO Change here
        if self.predefined_env:
            self.env.load_resource('torch_training.railway', 'complex_scene.pkl')

        self.width = self.env.width
        self.height = self.env.height
        self.step_memory = config["step_memory"]

        # Needed for the renderer
        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict

    def get_agent_handles(self):
        return self.env.get_agent_handles()

    def get_num_agents(self, static=True):
        return self.env.get_num_agents()

    def add_agent_static(self, agent_static):
        pass

    def set_agent_active(self, handle: int):
        pass

    def restart_agents(self):
        pass
    
    # Taken from MultiAgentEnv
    def reset(self):
        self.agents_done = []
        if self.predefined_env:
            obs = self.env.reset(False, False)
        else:
            obs = self.env.reset()

        # RLLib only receives observation of agents that are not done.
        o = dict()

        for a in range(len(self.env.agents)):
            data, distance, agent_data = self.env.obs_builder.split_tree(tree=np.array(obs[a]),
                                                                         current_depth=0)
            o[a] = [data, distance, agent_data]

        # Needed for the renderer
        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict

        # If step_memory > 1, we need to concatenate it the observations in memory, only works for
        # step_memory = 1 or 2 for the moment
        if self.step_memory < 2:
            return o
        else:
            self.old_obs = o
            oo = dict()

            for a in range(len(self.env.agents)):
                oo[a] = [o[a], o[a]]
            return oo


    # Taken from MultiAgentEnv
    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)

        d = dict()
        r = dict()
        o = dict()

        for a in range(len(self.env.agents)):
            if a not in self.agents_done:
                data, distance, agent_data = self.env.obs_builder.split_tree(tree=np.array(obs[a]),
                                                                             current_depth=0)

                o[a] = [data, distance, agent_data]
                r[a] = rewards[a]
                d[a] = dones[a]

        d['__all__'] = dones['__all__']

        if self.step_memory >= 2:
            oo = dict()

            for a in range(len(self.env.agents)):
                if a not in self.agents_done:
                    oo[a] = [o[a], self.old_obs[a]]

            self.old_obs = o

        for agent, done in dones.items():
            if done and agent != '__all__':
                self.agents_done.append(agent)

        if self.step_memory < 2:
            return o, r, d, infos
        else:
            return oo, r, d, infos

