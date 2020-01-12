import numpy as np
import math

def print_rails(height, width, id_node_to_cell, id_edge_to_cells):
    switch = -1
    rail = 1
    map = np.zeros((height, width), dtype=int)
    ref = np.zeros((height, width), dtype=int)

    for i in id_node_to_cell:
        (x, y) = id_node_to_cell[i]
        map[x][y] = i
        ref[x][y] = switch

    for i in id_edge_to_cells:
        edges = id_edge_to_cells[i]
        for e in edges:
            ((x, y), _) = e
            map[x][y] = i
            ref[x][y] = rail

    f = open("./temp/map.txt","w+")
    for r in range(height):
        for c in range(width):
            char = map[r][c]
            cellType = ref[r][c]
            if cellType == switch:
                char = '[{}]'.format(char)
            elif char == 0 and cellType != rail:
                char = "·"
            f.write("{:^4} ".format(char))
        f.write("\r\n")
    f.close()
    print('MAP SAVED AS "temp/map.txt"')