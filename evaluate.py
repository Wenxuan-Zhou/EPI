import numpy as np
import pickle
import os
import tensorflow as tf
import argparse
import EPI
from EPI.interaction_policy.epi_policy import EPIPolicy


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--epi_folder', default=None)
    parser.add_argument('--epi_itr', default=None)
    args = parser.parse_args()
    name = args.name

    if 'Hopper' in name:
        EPI.init('hopper', num_of_params=8)
    elif 'Striker' in name:
        EPI.init('striker')

    sess = tf.Session()
    sess.__enter__()
    algo = pickle.load(open(os.getcwd()+"/"+name+"/pickle.p", "rb"))
    env = algo.env
    if 'Task' in args.name:
        assert(args.epi_folder and args.epi_itr, "Please specify epi folder and epi iteration")
        interaction_policy = EPIPolicy(env.spec, mode='policy',
                                       policy_file=args.epi_folder + '/policy_itr_' + args.epi_itr + '.p',
                                       encoder_file=args.epi_folder + '/mlp_i_itr_' + args.epi_itr + '.h5',
                                       scaler_file=args.epi_folder + '/i_scaler_itr_' + args.epi_itr + '.pkl',
                                       embedding_scaler_file=args.epi_folder + '/embedding_scaler_itr_' + args.epi_itr + '.pkl', )

        env.wrapped_env.wrapped_env.env.env.load_interaction_policy(interaction_policy)
        interaction_policy.load_models()

    result = []
    for j in range(100):
        observation = env.reset()
        print(env.wrapped_env.wrapped_env.env.env.scale)
        total_reward = 0
        terminal = False
        env_dict = None
        while not terminal:
            action_rand, d = algo.policy.get_action(observation)
            action = d['mean']
            next_observation, reward, terminal, env_dict = env.step(action)
            observation = next_observation
            total_reward += reward
            # env.render()

        if 'Hopper' in name:
            result.append(total_reward)
            print(total_reward)
        elif 'Striker' in name:
            result.append((env_dict['reward_dist']))
            print(env_dict['reward_dist'])

    print('Results:' + str(np.mean(np.array(result))))


if __name__ == '__main__':
    main()
