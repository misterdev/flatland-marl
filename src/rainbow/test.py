# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objects import Scatter
from plotly.graph_objs.scatter import Line
import torch
import random
import numpy as np
from tqdm import trange

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

from src.algo.graph.utils import map_to_graph

# Test DQN
def test(args, T, ep, dqn, val_mem, metrics, results_dir, evaluate=False):
    
    # Init env and set in evaluation mode
    # Maps speeds to % of appearance in the env
    speed_ration_map = {1.: 0.25,    # Fast passenger train
                        1. / 2.: 0.25,    # Fast freight train
                        1. / 3.: 0.25,    # Slow commuter train
                        1. / 4.: 0.25}    # Slow freight train

    schedule_generator = sparse_schedule_generator(speed_ration_map)

    observation_builder = GraphObsForRailEnv(
                                            bfs_depth=args.bfs_depth,
                                            predictor=ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth))

    env = RailEnv(
                width=args.width,
                height=args.height,
                rail_generator=sparse_rail_generator(
                            max_num_cities=args.max_num_cities,
                            seed=ep, # Use episode as seed when evaluation is performed during training
                            grid_mode=args.grid_mode,
                            max_rails_between_cities=args.max_rails_between_cities,
                            max_rails_in_city=args.max_rails_in_city,
                            ),
                schedule_generator=schedule_generator,
                number_of_agents=args.num_agents,
                obs_builder_object=observation_builder,
                malfunction_generator_and_process_data=malfunction_from_params(
                    parameters={
                        'malfunction_rate': args.malfunction_rate,
                        'min_duration': args.min_duration,
                        'max_duration': args.max_duration
                    }),
                )
    
    if args.render:
        env_renderer = RenderTool(
                env,
                gl="PILSVG",
                agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                show_debug=True,
                screen_height=1080,
                screen_width=1920)

    #max_time_steps = env.compute_max_episode_steps(env.width, env.height)
    max_time_steps = 150 # TODO Debug
    metrics['steps'].append(T)
    T_rewards = [] # List of episodes rewards
    T_Qs = [] # List
    T_num_done_agents = [] # List of number of done agents for each episode
    T_all_done = [] # If all agents completed in each episode
    network_action_dict = dict()
    railenv_action_dict = dict()

    # Test performance over several episodes
    for ep in range(args.evaluation_episodes):
        # Reset info
        state, info = env.reset()
        reward_sum, all_done = 0, False  # reward_sum contains the cummulated reward obtained as sum during the steps
        num_done_agents = 0
        if args.render:
            env_renderer.reset()
            
        # Choose first action - decide entering of agents into the environment TODO Now random
        for a in range(env.get_num_agents()):
            action = np.random.choice((0,2))
            railenv_action_dict.update({a: action})
        state, reward, done, info = env.step(railenv_action_dict) # Env step
        reward_sum += sum(reward[a] for a in range(env.get_num_agents()))

        if args.render:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=True)
            
        for step in range(max_time_steps - 1):
            # TODO
            #if step == 10:
            #    map_to_graph(env)
            # Choose actions
            for a in range(env.get_num_agents()):
                if info['action_required'][a]:
                    network_action = dqn.act_e_greedy(state[a])    # Choose an action ε-greedily
                    railenv_action = observation_builder.choose_railenv_action(a, network_action)
                else:
                    network_action = 0
                    railenv_action = 0 # DO_NOTHING

                railenv_action_dict.update({a: railenv_action})
                network_action_dict.update({a: network_action})
                
            if args.debug:
                for a in range(env.get_num_agents()):
                    print('#########################################')
                    print('Info for agent {}'.format(a))
                    print('Obs: {}'.format(state[a]))
                    print('Status: {}'.format(info['status'][a]))
                    print('Position: {}'.format(env.agents[a].position))
                    print('Moving? {} at speed: {}'.format(env.agents[a].moving, info['speed'][a]))
                    print('Action required? {}'.format(info['action_required'][a]))
                    print('Network action: {}'.format(network_action_dict[a]))
                    print('Railenv action: {}'.format(railenv_action_dict[a]))
            
            # Breakpoint for debugging here
            state, reward, done, info = env.step(railenv_action_dict)  # Env step
            if args.render:
                env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

            reward_sum += sum(reward[a] for a in range(env.get_num_agents()))

            if done['__all__']:
                all_done = True
                break
        # No need to close the renderer since env parameter sizes stay the same
        T_rewards.append(reward_sum)
        # Compute num of agents that reached their target
        for a in range(env.get_num_agents()):
            if done[a]:
                num_done_agents += 1 
        T_num_done_agents.append(num_done_agents / env.get_num_agents()) # In proportion to total
        T_all_done.append(all_done)

    # Test Q-values over validation memory
    
    for state in val_mem:    # Iterate over valid states
        T_Qs.append(dqn.evaluate_q(state))
    #if args.debug:
    print('T_Qs: {}'.format(T_Qs))
    
    avg_done_agents = sum(T_num_done_agents) / len(T_num_done_agents) # Average number of agents that reached their target
    avg_reward = sum(T_rewards) / len(T_rewards)
    avg_norm_reward = avg_reward / (max_time_steps / env.get_num_agents())
    
    # avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
    if not evaluate:
        # Save model parameters if improved
        if avg_done_agents > metrics['best_avg_done_agents']:
            metrics['best_avg_done_agents'] = avg_done_agents
            dqn.save(results_dir)

        # Append to results and save metrics
        metrics['rewards'].append(T_rewards)
        #metrics['Qs'].append(T_Qs)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Plot
        _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)  # Plot rewards in episodes
        #_plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)

    # Return average number of done agents (in proportion) and average reward
    return avg_done_agents, avg_reward, avg_norm_reward


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)
