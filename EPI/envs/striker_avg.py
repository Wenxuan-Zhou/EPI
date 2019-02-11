import numpy as np
from .striker_original import StrikerEnv
import EPI


class StrikerAvgEnv(StrikerEnv):
    def __init__(self):
        # default values for additional properties
        self.scale = np.zeros(EPI.NUM_OF_PARAMS, dtype=float)
        self.env_id = 0
        super(StrikerAvgEnv, self).__init__()
        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.original_damping = np.copy(self.model.dof_damping)

    def _step(self, a):
        ob, reward, done, d = super(StrikerAvgEnv, self)._step(a)
        d['scale'] = np.copy(self.scale)
        d['env_id'] = np.copy(self.env_id)
        return ob, reward, done, d

    def raw_reset_model(self):
        return super(StrikerAvgEnv, self).reset_model()

    def reset_model(self):
        self.change_env()
        return self.raw_reset_model()

    def change_env(self, scale=None):
        mass = np.copy(self.original_mass)
        inertia = np.copy(self.original_inertia)
        damping = np.copy(self.original_damping)

        if scale is None:
            self.scale = np.random.randint(0, 5, EPI.NUM_OF_PARAMS)*0.1+0.05  # 0~0.4
        else:
            self.scale = scale

        self.env_id = int((self.scale[0] * 5 + self.scale[1]) * 10)

        mass[11] = ((self.scale[0]-0.1)*8+1) * mass[11]  # 0.2~4.2*mass
        inertia[11, :] = ((self.scale[0]-0.1)*8+1) * inertia[11, :]

        damping[7] = (self.scale[1] - 0.2) * 2 + 0.5  # default 0.5: 0.1~1.1
        damping[8] = (self.scale[1] - 0.2) * 2 + 0.5

        self.model.body_mass = mass
        self.model.body_inertia = inertia
        self.model.dof_damping = damping
        return

    def force_reset_model(self, qpos, qvel):
        self.set_state(qpos, qvel)
        return self._get_obs()
