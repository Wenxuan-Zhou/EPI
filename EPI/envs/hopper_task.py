import numpy as np
from .hopper_avg import HopperAvgEnv
import EPI


class HopperTaskEnv(HopperAvgEnv):
    def __init__(self, reset):
        self.epi_reset = reset
        self.env_vec = np.zeros(EPI.EMBEDDING_DIMENSION)
        super(HopperTaskEnv, self).__init__()
        self.interactive_policy = None

    def load_interaction_policy(self, p):
        self.interactive_policy = p

    def reset_model(self):
        self.change_env()
        self.raw_reset_model()
        self.env_vec = self.interactive_policy.do_interaction(self)
        if self.epi_reset:
            self.raw_reset_model()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10),
            self.env_vec,
        ])

    def get_raw_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10),
        ])
