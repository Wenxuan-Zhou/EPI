from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.trpo import TRPO as OriginalTRPO
import time
import rllab.misc.logger as logger
import tensorflow as tf
from rllab.sampler.utils import rollout
from .sampler import VectorizedSampler
import pickle


# TRPO for goal policy
# Load interaction models after tf.global_variables_initializer()
class TRPO(OriginalTRPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(self, **kwargs):
        super(TRPO, self).__init__(sampler_cls=VectorizedSampler, **kwargs)

    @overrides
    def train(self, sess=None, interaction_policy=None, log_dir=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        interaction_policy.load_models()
        self.start_worker()

        # Load tf models in interaction policy

        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
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
            if itr % 200 == 0 and log_dir is not None:
                pickle.dump(self, open(log_dir + "/algo_itr_"+str(itr)+".p", "wb"))
        self.shutdown_worker()
        if created_session:
            sess.close()
