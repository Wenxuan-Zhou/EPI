import numpy as np
from .striker_avg import StrikerAvgEnv


class StrikerOracleEnv(StrikerAvgEnv):
    def __init__(self):
        super(StrikerOracleEnv, self).__init__()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
            self.scale,
        ])
