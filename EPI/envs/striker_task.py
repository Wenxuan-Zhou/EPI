import numpy as np
from .striker_avg import StrikerAvgEnv
import EPI


class StrikerTaskEnv(StrikerAvgEnv):
    def __init__(self, reset):
        self.epi_reset = reset
        self.env_vec = np.zeros(EPI.EMBEDDING_DIMENSION)
        super(StrikerTaskEnv, self).__init__()
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
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
            self.env_vec,
        ])

    def get_raw_obs(self):
        # obs for interaction
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])

