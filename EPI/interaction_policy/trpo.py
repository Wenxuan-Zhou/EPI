from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.trpo import TRPO as OriginalTRPO
from IPython import embed
import time
import rllab.misc.logger as logger
import tensorflow as tf
from rllab.sampler.utils import rollout
import numpy as np
import pickle
import EPI


class TRPO(OriginalTRPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(self, **kwargs):
        super(TRPO, self).__init__(**kwargs)

    @overrides
    def train(self, sess=None, pred_model=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker()
        start_time = time.time()
        pred_model.load_models(logger=logger)

        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):

                # Update prediction model
                if itr % pred_model.update_freq == 0:
                    pickle.dump(self.policy, open(pred_model.filepath + "/policy_itr_"+str(itr)+".p", "wb"))  # need sess
                    logger.log('Policy saved:{}'.format(pred_model.filepath + "/policy_itr_"+str(itr)+".p"))
                    sample_count = 0
                    while sample_count < pred_model.update_batch_size:
                        sample = self.obtain_samples(itr)
                        for path in sample:
                            traj = np.concatenate([path['observations'], path['actions']], axis=1).reshape(-1)
                            pred_model.save_trajectory(traj, path['env_infos']['env_id'][0])
                        sample_count += len(sample)

                    logger.log("Training prediction model...")
                    pred_model.update(itr, logger=logger)
                    logger.log("Prediction model saved.")

                # Process path for prediction model:
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                logger.log("Modifying reward using prediction model...")
                predicion_score_list = []
                reset_score_list = []
                default_score_list = []
                for k in range(len(paths)):
                    path = paths[k]
                    traj = np.concatenate([path['observations'], path['actions']], axis=1).reshape(-1)
                    env_vec = path['env_infos']['env_id'][0]
                    predicion_score = pred_model.get_score(traj, env_vec)
                    predicion_score_list.append(predicion_score)
                    reset_score = -path['env_infos']['reset_dist'][-1]
                    reset_score_list.append(reset_score)
                    default_score_list.append(path['rewards'])
                    if not path['env_infos']['dead'][-1]:
                        path['rewards'] = path['rewards'] * EPI.DEFAULT_REWARD_SCALE + predicion_score * EPI.PREDICTION_REWARD_SCALE
                    if itr % pred_model.saving_freq == 0:
                        pred_model.save_embedding(traj, env_vec)
                if itr % pred_model.saving_freq == 0:
                    pred_model.evaluate_embedding('itr'+str(itr))

                logger.log("Average default reward: %.3f" % np.array(default_score_list).mean())
                logger.log("Average prediction reward: %.3f" % np.array(predicion_score_list).mean())
                logger.log("Average reset reward: %.3f" % np.array(reset_score_list).mean())

                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

        self.shutdown_worker()
        if created_session:
            sess.close()
