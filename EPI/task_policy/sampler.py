import pickle
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler as OriginalVectorizedSampler
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor


# Push the same interaction policy into vectorized environments
class VectorizedSampler(OriginalVectorizedSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo, n_envs=None)

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]

            original_env = self.algo.env.wrapped_env.wrapped_env.env.env
            for env in envs:
                env.wrapped_env.wrapped_env.env.env.load_interaction_policy(original_env.interactive_policy)

            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec