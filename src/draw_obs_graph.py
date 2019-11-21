#from graph_tool.all import *
from collections import defaultdict

'''
Given the graph of observations as a dict, where a key is the cell position and the value is a Node, returns a simple dict
that decodes only edge properties, so of type:
{
id : id2, id3,
id2 : id4,
}
'''


def get_id_graph(obs_graph, cell_id_map):

    id_graph = defaultdict(list)
    for cell_pos in obs_graph.keys():
        # Values are Node structures, not cell pos
        for adj_cell_node in obs_graph[cell_pos]:
            v_id = cell_id_map[cell_pos]
            adj_v_id = cell_id_map[adj_cell_node[0]]
            id_graph[v_id].append(adj_v_id)

    return id_graph


'''
Given list of cell positions (tuples) return a dict of type { cell_pos : id }
'''


def _cell_id_map(cells):
    # Assign ids to cell, vertex ids start from 0
    cell_id_map = {}
    v_id = 0
    for cell_pos in cells:
        cell_id_map[cell_pos] = v_id
        v_id += 1
    return cell_id_map


'''
Given graph of observations for one agent (handle), build a graph with graph-tool to visualize the network and draw it.
'''


def build_graph(obs_graph, handle):

    g = Graph()
    bifurcation_cells = obs_graph.keys()
    num_vertices = len(bifurcation_cells)

    # Add vertices to graph
    vertices = g.add_vertex(num_vertices)

    cell_to_id_map = _cell_id_map(bifurcation_cells)
    # Add edges to g using the id_graph
    id_graph = get_id_graph(obs_graph, cell_to_id_map)
    # Add properties to vertices (e.g. cell positions)
    # Create a new property map
    #vprop_cell_pos = g.new_vertex_property("object")
    vprop_cell_pos = g.new_vertex_property("vector<float>")
    # Associate property map to name in the dict
    g.vertex_properties["cell position"] = vprop_cell_pos
    for v in g.vertices():
        # Get v id
        v_id = int(v)  # or g.vertex_index[v]
        # Retrieve cell pos (basically revert dict) 
        for cell_pos in cell_to_id_map.keys():
            if cell_to_id_map[cell_pos] == v_id:
                v_cell_pos = cell_pos
                # Fill the value for current property map
                vprop_cell_pos[v] = [v_cell_pos[1], v_cell_pos[0]]  # convert from tuple to vec

    # Add edges
    for v_id in id_graph.keys():
        for adj_v_id in id_graph[v_id]:
            # Add edge using vertices ids
            g.add_edge(g.vertex(v_id), g.vertex(adj_v_id))

    # TODO Split drawing from graph building
    graph_draw(g,
               pos=g.vertex_properties["cell position"],
               bg_color=[1, 1, 1, 1],  # white
               vertex_text=g.vertex_properties["cell position"],
               vertex_size=1,
               vertex_font_size=10,
               vertex_text_color=[1, 1, 1, 1],
               vertex_fill_color=[0.640625, 0, 0, 0.5],  # red almost transparent
               edge_pen_width=5,
               edge_marker_size=20,
               output_size=(1000, 1000),
               output="graphobs{}.png".format(handle))

    return g

