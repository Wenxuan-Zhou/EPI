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
from sandbox.rocky.tf.algos.trpo import TRPO as BasicTRPO
from EPI.interaction_policy.epi_policy import EPIPolicy
from EPI.task_policy.trpo import TRPO as TaskTRPO
import EPI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--epi_folder')
    parser.add_argument('--epi_itr')
    parser.add_argument('--params', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--itr', type=int)
    parser.add_argument('--gae_lambda', type=float)
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--discount', type=float)
    args = parser.parse_args()

    # Defaults
    TRPO = BasicTRPO
    batch_size = 100000
    max_path_length = 1000
    n_itr = 1000
    hidden_sizes = (32, 32)
    gae_lambda = 1
    step_size = 0.01
    discount = 0.99

    if 'Striker' in args.name:
        EPI.init('striker')
        max_path_length = 200
    elif 'Hopper' in args.name:
        EPI.init('hopper', num_of_params=args.params)
    else:
        assert()

    log_dir = setup_logger(exp_name=args.name)
    env = TfEnv(normalize(GymEnv(args.name + "-v0")))
    optimizer = None
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # Task Policy
    if 'Task' in args.name:
        TRPO = TaskTRPO
        interaction_policy = EPIPolicy(env.spec, mode='policy',
                                       policy_file=args.epi_folder + '/policy_itr_' + args.epi_itr + '.p',
                                       encoder_file=args.epi_folder + '/mlp_i_itr_' + args.epi_itr + '.h5',
                                       scaler_file=args.epi_folder + '/i_scaler_itr_' + args.epi_itr + '.pkl',
                                       embedding_scaler_file=args.epi_folder + '/embedding_scaler_itr_' + args.epi_itr + '.pkl', )
        env.wrapped_env.wrapped_env.env.env.load_interaction_policy(interaction_policy)

    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.itr is not None:
        n_itr = args.itr
    if args.gae_lambda is not None:
        gae_lambda = args.gae_lambda
    if args.step_size is not None:
        step_size = args.step_size
    if args.discount is not None:
        discount = args.discount

    logger.log("batch_size:" + str(batch_size))
    logger.log("n_itr:" + str(n_itr))
    logger.log("gae_lambda:" + str(gae_lambda))
    logger.log("discount:" + str(discount))
    logger.log("step_size:" + str(step_size))

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=discount,
        step_size=step_size,
        gae_lambda=gae_lambda,
        optimizer=optimizer,
    )

    sess = tf.Session()
    sess.__enter__()
    if 'Task' in args.name:
        algo.train(sess=sess, interaction_policy=interaction_policy, log_dir=log_dir)
    else:
        algo.train(sess=sess)

    pickle.dump(algo, open(log_dir+"/pickle.p", "wb"))  # need sess
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
