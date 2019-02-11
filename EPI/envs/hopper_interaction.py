import numpy as np
from .hopper_avg import HopperAvgEnv
import pandas as pd
import os
import EPI


class HopperInteractionEnv(HopperAvgEnv):
    def __init__(self):
        self.scale_list = pd.read_csv(os.path.dirname(__file__) + '/hopper_env_list.csv').values
        super(HopperInteractionEnv, self).__init__()

    def _step(self, a):
        ob, reward, done, d = super(HopperInteractionEnv, self)._step(a)
        # alive_bonus = 1.0
        # reward = alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        d['dead'] = done
        if done:
            reward = 0
        d['reset_dist'] = np.linalg.norm(self.init_qpos - self.model.data.qpos.ravel().copy())
        return ob, reward, False, d

    def reset_model(self):
        if EPI.NUM_OF_PARAMS == 8:
            env_id = np.random.randint(EPI.NUM_OF_ENVS)
            scale = self.scale_list[env_id, 1:]
            self.change_env(scale=scale, env_id=env_id)
        else:
            self.change_env()
        return self.raw_reset_model()
