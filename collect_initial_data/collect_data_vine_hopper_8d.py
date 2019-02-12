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

    name = 'Exp180418_simple_baseline_hopper'

    EPI.init('hopper', num_of_params=8)

    sess = tf.Session()
    sess.__enter__()
    algo = pickle.load(open(os.getcwd()+"/"+name+"/pickle.p", "rb"))

    env = TfEnv(normalize(GymEnv('HopperAvg-v0')))
    core_env = env.wrapped_env.wrapped_env.env.env

    target_sample_size = 500
    egreedy = 0.2

    data = []
    rollouts = []
    sample_size = 0
    while sample_size < target_sample_size:
        observation = env.reset()
        core_env.change_env(scale=np.array([0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1]), env_id=0)
        episode_size = 0
        while True:
            if np.random.rand() < egreedy:
                action = env.action_space.sample()
            else:
                action, d = algo.policy.get_action(observation)
            full_state = core_env.state_vector()
            rollouts.append([full_state, action])
            next_observation, reward, terminal, reward_dict = env.step(action)
            episode_size += 1
            sample_size += 1
            observation = next_observation
            if terminal or sample_size == target_sample_size:
                break

    print('Rollout...')
    scale_list = pd.read_csv('../EPI/envs/hopper_env_list.csv').values
    for i in range(100):
        env_id = i
        core_env.change_env(scale=scale_list[i, 1:], env_id=i)
        print(core_env.env_id)
        print(core_env.scale)
        for rollout in rollouts:
            state = rollout[0]
            observation = core_env.force_reset_model(qpos=state[0:6], qvel=state[6:12])
            action = rollout[1]
            next_observation, reward, terminal, reward_dict = env.step(action)
            data.append(np.concatenate([observation, action, next_observation, np.array([env_id]), core_env.scale, np.array([reward, terminal * 1])]))
            sample_size += 1
            observation = next_observation

    data = np.array(data)

    g = lambda s, num: [s + str(i) for i in range(num)]
    columns = g('obs', len(observation))+g('ac', len(action))+g('next_obs', len(observation))+g('env_id', 1)+g('env_vec', 8)+['reward']+['terminal']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('../EPI/envs/hopper_data_vine.csv')


if __name__ == '__main__':
    main()
