import numpy as np
from .hopper_avg import HopperAvgEnv


class HopperOracleEnv(HopperAvgEnv):
    def __init__(self):
        super(HopperOracleEnv, self).__init__()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10),
            self.scale,
        ])
