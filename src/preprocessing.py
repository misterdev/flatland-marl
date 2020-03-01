import numpy as np

class ObsPreprocessor:
    def __init__(self, max_rails, reorder_rails, cut_bitmap):
        self.max_rails = max_rails
        self.reorder_rails = reorder_rails
        self.cut_bitmap = cut_bitmap
        
    def _fill_padding(self, obs, max_rails):
        """
        
        :param obs: Agent state 
        :param max_rails: Maximum number of rails in environment 
        :return: Observation padded with 0s along first axis (until max_rails)
        
        """
        prediction_depth = obs.shape[1]
        
        pad_agent_obs = np.zeros((max_rails, prediction_depth))
        pad_agent_obs[:obs.shape[0], :obs.shape[1]] = obs
        
        return pad_agent_obs

    # (agents x rails x depth)
    def _get_heatmap(self, handle, bitmaps, max_rails):
        temp_bitmaps = np.copy(bitmaps)
        temp_bitmaps[handle, :, :] = 0
        pos_dir = np.sum(np.where(temp_bitmaps > 0, temp_bitmaps, 0), axis=0)
        neg_dir = np.abs(np.sum(np.where(temp_bitmaps < 0, temp_bitmaps, 0), axis=0))
        
        return pos_dir, neg_dir

    def _swap_rails(self, bitmap, swap):
        bitmap[range(len(swap))] = bitmap[swap]
        bitmap[len(swap):, :] = 0 # Reset to 0 rows that were swapped
        return bitmap
    
    def _find_neighbors(self, rail, dir_on_rail, info, connections):
        """
        
        :param rail: id
        :param dir_on_rail: 1 or -1
        :return: List containing ids of rail that exit from the destination node
        """
        node = self.info[rail][1] if dir_on_rail == 1 else self.info[rail]
        # Take all rails exiting that node
        neighbors = [rail for rail in self.connections[node]] # NESW ?
            

    def _reorder_rails(self, bitmap, pos_map, neg_map, info, connections):
        swap = np.array([], dtype=int) # Keep the list of rails that are in the path, that must go on top

        ts = 0 
        rail = np.argmax(np.absolute(bitmap[:, ts])) # Take starting rail
        # If agent not departed
        if bitmap[rail, ts] == 0:
            ts = 1
            rail = np.argmax(np.absolute(bitmap[:, ts]))
        
        # While the bitmap is not empty
        
        while bitmap[rail, ts] != 0:
            swap = np.append(swap, rail) if rail not in swap else swap
            ts += np.argmax(bitmap[rail, ts:] == 0) # Take ts at start of next rail
            rail = np.argmax(np.absolute(bitmap[:, ts])) # Go to next rail
        
        '''
        while bitmap[rail, ts] != 0:
            swap = np.append(swap, rail) if rail not in swap else swap
            neighbor_rails = self._find_neighbors(rail, 1, info, connections)

            ts += np.argmax(bitmap[rail, ts:] == 0) # Take ts at start of next rail
            rail = np.argmax(np.absolute(bitmap[:, ts])) # Go to next rail
        '''
        
        if len(swap) > 0:
            bitmap = self._swap_rails(bitmap, swap)
            pos_map = self._swap_rails(pos_map, swap)
            neg_map = self._swap_rails(neg_map, swap)
        
        return bitmap, pos_map, neg_map
    
    def _cut_map(self, map, cutoff):
        new_map = map[:cutoff, :map.shape[1]]
        return new_map

    
    def get_obs(self, handle, bitmap, maps, info, connections):
        # Select subset of conflicting paths in bitmap
        pos_map, neg_map = self._get_heatmap(handle, maps, self.max_rails)
        if self.reorder_rails:
            bitmap, pos_map, neg_map = self._reorder_rails(bitmap, pos_map, neg_map, info, connections)
        
        if self.cut_bitmap:
            state = np.concatenate([
                self._cut_map(bitmap, 15), # 15 is cutoff threshold...
                self._cut_map(pos_map, 15),
                self._cut_map(neg_map, 15)
            ])
        
        else: 
            state = np.concatenate([
                self._fill_padding(bitmap, self.max_rails),
                self._fill_padding(pos_map, self.max_rails),
                self._fill_padding(neg_map, self.max_rails)
            ])
        
        return state # (max_cas * max_rails, prediction_depth + 1)