import numpy as np

class ObsPreprocessor:
    def __init__(self, max_rails, keep_rail_order):
        self.max_rails = max_rails
        self.keep_rail_order = keep_rail_order

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

    def get_obs(self, handle, bitmap, maps):
        # Select subset of conflicting paths in bitmap
        pos_dir, neg_dir = self._get_heatmap(handle, maps, self.max_rails)

        if self.keep_rail_order:
            state = np.concatenate([
                self._fill_padding(bitmap, self.max_rails),
                self._fill_padding(pos_dir, self.max_rails),
                self._fill_padding(neg_dir, self.max_rails)
            ])
        else:
            print('self.keep_rail_order')
        
        return state # (prediction_depth + 1, max_cas * max_rails)