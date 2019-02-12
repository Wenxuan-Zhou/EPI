import numpy as np
import tensorflow as tf
import pickle
import rllab.misc.logger as logger
from keras.models import load_model
from keras import Model
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .prediction_model import separation_loss
import keras.losses
keras.losses.separation_loss = separation_loss
import EPI
import pandas as pd
import os


class EPIPolicy(object):
    def __init__(self,
                 env_spec,
                 dir='.',
                 policy_file=None,
                 encoder_file=None,
                 scaler_file=None,
                 embedding_scaler_file=None,
                 num_of_actions=10,
                 ):

        self.filepath = dir
        self.env_spec = env_spec
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.num_of_actions = num_of_actions
        self.policy_file = policy_file
        self.encoder_file = encoder_file
        self.scaler_file = scaler_file
        if os.path.isfile(embedding_scaler_file):
            self.embedding_scaler_file = embedding_scaler_file
        else:
            self.embedding_scaler_file = None
        self.policy = None
        self.encoder_scaler = None
        self.embedding_scaler = None
        self.encoder = None
        self.embedding_list = []

    def load_models(self):
        if self.policy_file is not None:
            with tf.variable_scope('interaction_policy'):
                self.policy = pickle.load(open(self.policy_file, "rb"))
            logger.log('Interaction Policy Loaded:{}'.format(self.policy_file))

        if self.encoder_file is not None and self.scaler_file is not None:
            self.encoder_scaler = pickle.load(open(self.scaler_file, 'rb'))
            model = load_model(self.encoder_file)
            self.encoder = Model(inputs=model.get_layer('env_input').input,
                                 outputs=model.get_layer('encoder_output').output)
            logger.log('Encoder Loaded:{}'.format(self.policy_file))
            logger.log('Scaler Loaded:{}'.format(self.scaler_file))

        if self.embedding_scaler_file is not None:
            self.embedding_scaler = pickle.load(open(self.embedding_scaler_file, 'rb'))
            logger.log('Embedding Scaler Loaded:{}'.format(self.embedding_scaler_file))

    def get_action(self, obs=None):
        assert(self.policy, 'Policy action mode but no policy was given.')
        return self.policy.get_action(obs)

    def do_interaction(self, env):
        paths = []
        obs = env.get_raw_obs()
        running_path = dict(
            observations=[],
            actions=[],
            rewards=[],
            env_infos=[],
            agent_infos=[],
        )

        for i in range(self.num_of_actions):
            action, agent_info = self.get_action(obs)

            ub = env.action_space.high
            lb = env.action_space.low
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)

            _, reward, done, env_info = env.step(scaled_action)

            running_path["observations"].append(obs)
            running_path["actions"].append(action)
            running_path["rewards"].append(reward)
            running_path["env_infos"].append(env_info)
            running_path["agent_infos"].append(agent_info)
            if i == self.num_of_actions-1:
                paths.append(dict(
                    observations=self.env_spec.observation_space.flatten_n(running_path["observations"]),
                    actions=self.env_spec.action_space.flatten_n(running_path["actions"]),
                    rewards=running_path["rewards"],
                    env_infos=running_path["env_infos"],
                    agent_infos=running_path["agent_infos"],
                ))
                running_path = None
            obs = env.get_raw_obs()

        path = paths[0]
        traj = np.concatenate([path['observations'], path['actions']], axis=1).reshape(-1)
        env_vec = path['env_infos'][0]['env_id']

        embedding = self.encoder.predict({'env_input': self.encoder_scaler.transform(np.vstack([traj]))}).reshape(-1)

        if self.embedding_scaler is not None:
            embedding = self.embedding_scaler.transform(embedding.reshape(1, -1)).reshape(-1)

        self.embedding_list.append(np.concatenate([embedding, np.array([env_vec])]))

        return np.copy(embedding)
