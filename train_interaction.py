from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
import pickle
import rllab.misc.logger as logger
import os.path as osp
import datetime
import tensorflow as tf
import argparse
from EPI.interaction_policy.trpo import TRPO
from EPI.interaction_policy.prediction_model import PredictionModel
import EPI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('-n')
    parser.add_argument('-e')
    parser.add_argument('-r')
    parser.add_argument('-l')
    args = parser.parse_args()

    if args.l is None:
        loss_type = 'interaction_separation'
    else:
        loss_type = args.l

    if args.r is None:
        reward_scale = 0.5
    else:
        reward_scale = float(args.r)

    name = args.name

    if 'Striker' in name:
        if args.r is None:
            reward_scale = 0.1
        EPI.init('striker', prediction_reward_scale=reward_scale, embedding_dimension=int(args.e), loss_type=loss_type)
    elif 'Hopper' in name:
        if args.n == '2':
            EPI.init('hopper', prediction_reward_scale=reward_scale, num_of_envs=25, num_of_params=2,
                     embedding_dimension=int(args.e), loss_type=loss_type)
        elif args.n == '8':
            EPI.init('hopper', prediction_reward_scale=reward_scale, num_of_envs=100, num_of_params=8,
                     embedding_dimension=int(args.e), loss_type=loss_type)
        else:
            raise ValueError
    else:
        EPI.init((name.replace("Interaction", "")).lower(), prediction_reward_scale=reward_scale,
                 embedding_dimension=int(args.e), loss_type=loss_type)

    log_dir = setup_logger(exp_name=name)
    logger.log("EPI:DEFAULT_REWARD_SCALE:" + str(EPI.DEFAULT_REWARD_SCALE))
    logger.log("EPI:PREDICTION_REWARD_SCALE:" + str(EPI.PREDICTION_REWARD_SCALE))
    logger.log("EPI:NUM_OF_PARAMS:" + str(EPI.NUM_OF_PARAMS))
    logger.log("EPI:EMBEDDING_DIMENSION:" + str(EPI.EMBEDDING_DIMENSION))
    logger.log("EPI:LOSS_TYPE:" + str(EPI.LOSS_TYPE))

    env = TfEnv(normalize(GymEnv(name + '-v0')))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(32, 32),
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=10,
        n_itr=500,
        discount=0.99,
        step_size=0.01,
    )

    predicion_model = PredictionModel(log_dir)

    sess = tf.Session()
    sess.__enter__()
    algo.train(sess=sess, pred_model=predicion_model)
    pickle.dump(algo, open(log_dir + "/algo.p", "wb"))  # need sess
    sess.close()
    close_logger(log_dir)


def setup_logger(exp_name=''):
    # Logging info
    now = datetime.datetime.now()
    exp_name = 'Exp' + now.strftime("%y%m%d_") + exp_name
    n = 0
    while osp.exists('./data/' + exp_name + '_' + str(n)):
        n = n + 1
    exp_name = exp_name + '_' + str(n)
    log_dir = './data/' + exp_name
    logger.add_text_output(osp.join(log_dir, 'debug.log'))
    logger.add_tabular_output(osp.join(log_dir, 'progress.csv'))
    logger.push_prefix("[%s] " % exp_name)
    return log_dir


def close_logger(log_dir):
    logger.remove_tabular_output(osp.join(log_dir, 'progress.csv'))
    logger.remove_text_output(osp.join(log_dir, 'debug.log'))
    logger.pop_prefix()


if __name__ == '__main__':
    main()
