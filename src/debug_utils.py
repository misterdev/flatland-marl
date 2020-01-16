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

def print_bitmaps(maps, id=0):
    shape = maps.shape
    f = open("./temp/z-bitmaps-{}.txt".format(id),"w+")

    for a in range(shape[0]): # agent
        f.write('AGENT {} \n'.format(a))
        f.write('#COL:  ')
        for t in range(shape[2]): # time
            f.write("{:^3}".format(t))
        f.write('\n')
        for r in range(shape[1]): # rail
            f.write('>{:^3}:  '.format(r))
            for t in range(shape[2]): # time
                char = maps[a, r, t]
                if char == 1:
                    char = '[+]'
                elif char == -1:
                    char = '[-]'
                elif char == 0:
                    char = "·"
                f.write("{:^3}".format(char))
            f.write('\n')
        f.write('\n\n\n')

    f.close()
    print('BITMAP SAVED AS "temp/z-bitmaps-{}.txt"'.format(id))

def print_cells_sequence(height, width, cells_sequence):
    f = open("./temp/cell_seq.txt","w+")
    
    for a in cells_sequence:
        map = np.full((height, width), -1, dtype=int)
        cells = cells_sequence[a]
        step = 0
        for (x, y) in cells:
            map[int(x), int(y)] = step
            step += 1

        f.write('AGENT {} \nCELLSEQ: {} \n'.format(a, cells))
        for r in range(height):
            for c in range(width):
                char = map[r][c]
                if char == -1:
                    char = "·"
                f.write("{:^4} ".format(char))
            f.write("\r\n")
        f.write("\n\n")

    f.close()
    print('CELL SEQ SAVED AS "temp/cell_seq.txt"')