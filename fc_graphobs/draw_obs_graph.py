from graph_tool.all import *
from collections import defaultdict

'''
Given the graph of observations as a dict, where a key is the cell position and the value is a Node, returns a simple dict
that decodes only edge properties, so of type:
{
id : id2, id3,
id2 : id4,
}
'''


def get_id_graph(obs_graph, id_to_cell_map):

    id_graph = defaultdict(list)
    for cell_pos in obs_graph.keys():
        # Values are Node structures, not cell pos
        for adj_cell_node in obs_graph[cell_pos]:
            v_id = id_to_cell_map[cell_pos]
            adj_v_id = id_to_cell_map[adj_cell_node[0]]
            id_graph[v_id].append(adj_v_id)

    return id_graph


def build_id_cell_map(bifurcation_cells):
    # Assign ids to cell, vertex ids start from 0
    id_to_cell_map = {}
    v_id = 0
    for cell_pos in bifurcation_cells:
        id_to_cell_map[cell_pos] = v_id
        v_id += 1
    return id_to_cell_map


def build_graph(obs_graph):

    g = Graph()
    bifurcation_cells = obs_graph.keys()
    num_vertices = len(bifurcation_cells)

    # Add vertices to graph
    vertices = g.add_vertex(num_vertices)

    id_to_cell_map = build_id_cell_map(bifurcation_cells)
    # Add edges to g using the id_graph
    id_graph = get_id_graph(obs_graph, id_to_cell_map)
    # Add properties to vertices (e.g. cell positions)
    #vprop_cell_pos = g.new_vertex_property("object")
    #vprop_cell_pos[]

    for v_id in id_graph.keys():
        for adj_v_id in id_graph[v_id]:
            # Add edge using vertices ids
            g.add_edge(g.vertex(v_id), g.vertex(adj_v_id))

    

    graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=14, output_size=(800, 800), output="graphobs.png")

    return g

