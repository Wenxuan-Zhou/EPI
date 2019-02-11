import sys
sys.path.append("/home/sawyer/wenxuan/180521_EPI_clean/")
import numpy as np
import pickle
from IPython import embed
import os
import tensorflow as tf
import pandas as pd
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
import EPI


def main():

    name = 'Exp180512_simple_baseline_striker'

    EPI.init('striker', num_of_params=2)

    sess = tf.Session()
    sess.__enter__()
    algo = pickle.load(open(os.getcwd()+"/"+name+"/pickle.p", "rb"))

    env = TfEnv(normalize(GymEnv('StrikerAvg-v0')))
    core_env = env.wrapped_env.wrapped_env.env.env

    target_sample_size = 1000
    egreedy = 0.2

    data = []
    rollouts = []
    while len(rollouts) < target_sample_size:
        observation = env.reset()
        core_env.change_env(np.array([0.1, 0.1]))
        old_ball_pos = core_env.model.data.qpos[-9:-7]
        for i in range(200):
            if np.random.rand() < egreedy:
                action = env.action_space.sample()
            else:
                action, d = algo.policy.get_action(observation)

            ball_pos = core_env.model.data.qpos[-9:-7]
            if np.linalg.norm(ball_pos-old_ball_pos) > 0.005:
                full_state = core_env.state_vector()
                rollouts.append([full_state, action])
                # env.render()
            next_observation, reward, terminal, reward_dict = env.step(action)
            # env.render()
            observation = next_observation
            old_ball_pos = ball_pos
            if terminal or len(rollouts) == target_sample_size:
                print(reward_dict['reward_dist'])
                break

    print('Rollout...')
    for i in range(5):
        for j in range(5):
            env_id = int((i * 5 + j))  # default: 1, 2
            core_env.change_env(scale=np.array([i*0.1, j*0.1]))
            print(core_env.env_id)
            print(core_env.scale)

            for rollout in rollouts:
                state = rollout[0]
                observation = core_env.force_reset_model(qpos=state[:16], qvel=state[16:])
                action = rollout[1]
                before = np.concatenate([core_env.model.data.qpos[7:9, 0], core_env.model.data.qvel[7:9, 0], core_env.get_body_com("tips_arm")])
                next_observation, reward, terminal, reward_dict = env.step(action)
                after = np.concatenate([core_env.model.data.qpos[7:9, 0], core_env.model.data.qvel[7:9, 0], core_env.get_body_com("tips_arm")])
                data.append(np.concatenate([before, after, np.array([core_env.env_id]), core_env.scale]))
                observation = next_observation

    data = np.array(data)

    g = lambda s, num: [s + str(i) for i in range(num)]
    columns = g('obs', 7)+g('next_obs', 7)+g('env_id', 1)+g('env_vec', 2)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('data_vine_striker.csv')


if __name__ == '__main__':
    main()
