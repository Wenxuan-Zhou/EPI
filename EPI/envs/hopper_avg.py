import numpy as np
from .hopper_original import HopperEnv
import EPI


class HopperAvgEnv(HopperEnv):
    def __init__(self):
        # default values for additional properties
        self.scale = np.zeros(EPI.NUM_OF_PARAMS, dtype=float)
        self.env_id = 0
        super(HopperAvgEnv, self).__init__()
        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.original_friction = np.copy(self.model.geom_friction)
        self.original_damping = np.copy(self.model.dof_damping)

    def _step(self, a):
        ob, reward, done, d = super(HopperAvgEnv, self)._step(a)
        d['scale'] = np.copy(self.scale)
        d['env_id'] = np.copy(self.env_id)
        return ob, reward, done, d

    def reset_model(self):
        self.change_env()
        return self.raw_reset_model()

    def raw_reset_model(self):
        return super(HopperAvgEnv, self).reset_model()

    def change_env(self, scale=None, env_id=None):
        mass = np.copy(self.original_mass)
        inertia = np.copy(self.original_inertia)
        friction = np.copy(self.original_friction)
        damping = np.copy(self.original_damping)

        if scale is None:
            self.scale = np.random.randint(0, 5, EPI.NUM_OF_PARAMS)*0.1  # 0~0.4
            self.env_id = 0
        else:
            self.scale = scale
            self.env_id = env_id

        if EPI.NUM_OF_PARAMS == 8:
            mass[1] = ((self.scale[0]-0.1)*5+1) * mass[1]  # 0.5~2.5*mass
            mass[2] = ((self.scale[1]-0.1)*5+1) * mass[2]  # 0.5~2.5*mass
            mass[3] = ((self.scale[2]-0.1)*5+1) * mass[3]  # 0.5~2.5*mass
            mass[4] = ((self.scale[3]-0.1)*5+1) * mass[4]  # 0.5~2.5*mass

            inertia[1, :] = ((self.scale[0]-0.1)*5+1) * inertia[1, :]
            inertia[2, :] = ((self.scale[1]-0.1)*5+1) * inertia[2, :]
            inertia[3, :] = ((self.scale[2]-0.1)*5+1) * inertia[3, :]
            inertia[4, :] = ((self.scale[3]-0.1)*5+1) * inertia[4, :]

            friction[4, 0] = (self.scale[4]-0.2)*3 + 2  # 1.4~2.6

            damping[3] = (self.scale[5] - 0.1) * 3 + 1  # 0.7~2.2
            damping[4] = (self.scale[6] - 0.1) * 3 + 1  # 0.7~2.2
            damping[5] = (self.scale[7] - 0.1) * 3 + 1  # 0.7~2.2

        elif EPI.NUM_OF_PARAMS == 2:
            self.env_id = int((self.scale[0] * 5 + self.scale[1]) * 10)
            mass = ((self.scale[0] - 0.1) * 5 + 1) * mass  # 0.5~2.5*mass
            inertia = ((self.scale[0] - 0.1) * 5 + 1) * inertia
            friction[4, 0] = (self.scale[1] - 0.2) * 3 + 2  # 1.4~2.6

        self.model.body_mass = mass
        self.model.body_inertia = inertia
        self.model.geom_friction = friction
        self.model.dof_damping = damping
        return

    def force_reset_model(self, qpos, qvel):
        self.set_state(qpos, qvel)
        return self._get_obs()
