# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objects import Scatter
from plotly.graph_objs.scatter import Line
import torch
import random
import numpy as np


# TODO Non so se è corretto calcolare la stddev sulle moving average
# From https://github.com/Kaixhin/Rainbow
def plot_metric(xs, ys_population, title, path=''):
	'''
	Plots min, max and mean + standard deviation bars of a population over time, used to plot average reward.
	:param xs: 
	:param ys_population: 
	:param title: 
	:param path: 
	:return: 
	'''
	
	max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

	ys = torch.tensor(ys_population, dtype=torch.float32)
	ys_min, ys_max, ys_mean, ys_std = ys.min(), ys.max(), ys.mean(), ys.std()
	
	ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

	trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
	trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
	#trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
	trace_mean = Scatter(x=xs, y=ys.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean') 
	trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
	trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')
	
	plotly.offline.plot({
	    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
	    'layout': dict(title=title, xaxis={'title': 'episode'}, yaxis={'title': title})
	}, filename=os.path.join(path, title + '.html'), auto_open=False)