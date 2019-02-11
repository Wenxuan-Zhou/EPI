import numpy as np
from .striker_avg import StrikerAvgEnv


class StrikerInteractionEnv(StrikerAvgEnv):
    def __init__(self):
        super(StrikerInteractionEnv, self).__init__()

    def _step(self, a):
        ob, reward, done, d = super(StrikerInteractionEnv, self)._step(a)
        # if ball out of range (0.3~0.7, -0.4~0)
        ball_pos = self.model.data.qpos[-9:-7]
        reward = 0.1 * d['reward_ctrl'] + 0.5 * d['reward_near']
        if not (-0.4 < ball_pos[0] < 0 and 0.3 < ball_pos[1] < 0.7):
            d['dead'] = True
        else:
            d['dead'] = False
        d['reset_dist'] = np.linalg.norm(ball_pos - self.ball)
        return ob, reward, False, d

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])