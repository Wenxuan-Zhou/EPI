import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython import embed
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.models import load_model
import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
import matplotlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import EPI


def separation_loss(y_true, y_pred):

    y_true = tf.squeeze(y_true)
    env_id, _ = tf.unique(y_true)

    mu = []
    sigma = []
    for i in range(EPI.NUM_OF_ENVS):
        idx = tf.where(tf.equal(y_true, env_id[i]))
        traj = tf.gather(y_pred, idx)
        mu.append(tf.squeeze(K.mean(traj, axis=0)))
        this_sigma = tf.maximum(K.mean(K.std(traj, axis=0))-0.1, 0)
        sigma.append(this_sigma)

    mu = tf.stack(mu)
    r = tf.reduce_sum(mu * mu, 1)
    r = tf.reshape(r, [-1, 1])
    D = (r - 2 * tf.matmul(mu, tf.transpose(mu)) + tf.transpose(r))/tf.constant(EPI.EMBEDDING_DIMENSION, dtype=tf.float32)
    D = tf.sqrt(D + tf.eye(EPI.NUM_OF_ENVS, dtype=tf.float32))
    distance = K.mean(tf.reduce_sum(0.1 - tf.minimum(D, 0.1)))

    sigma = tf.stack(sigma)

    return (distance + K.mean(sigma))*0.01


def train(name, data, model_type='avg', plot=False, model=None, verbose=1, epochs=500, logger=None):

    def print_note(note_arr):
        if logger is not None:
            logger.log(note_arr)
        else:
            print(name+':'+note_arr)
        return

    x_dim = data['XTrain'].shape[1]
    y_dim = data['YTrain'].shape[1]

    main_input = Input(shape=(x_dim,), name='main_input')

    if model_type == 'interaction_separation':
        env_input = Input(shape=(data['ITrain'].shape[1],), name='env_input')
        e = Dense(32, activation='relu')(env_input)
        e = Dense(32, activation='relu')(e)
        extra_output = Dense(EPI.EMBEDDING_DIMENSION, name='encoder_output')(e)

        m = concatenate([main_input, extra_output])
        m = Dense(128, activation='relu')(m)
        m = Dense(128, activation='relu')(m)
        m = Dense(128, activation='relu')(m)
        m = Dense(128, activation='relu')(m)
        state_output = Dense(y_dim, activation='sigmoid', name='state_output')(m)
        model = Model(inputs=[main_input, env_input], outputs=[state_output, extra_output])
        model.compile(optimizer='adam', loss=['mean_squared_error', separation_loss])
        model.summary()
        x = {'main_input': data['XTrain'], 'env_input': data['ITrain']}
        y = {'state_output': data['YTrain'], 'encoder_output': data['ETrain']}
        validation_data = ([data['XValid'], data['IValid']], [data['YValid'], data['EValid']])

    elif model_type == 'avg':
        m = main_input
        m = Dense(128, activation='relu')(m)
        m = Dense(128, activation='relu')(m)
        m = Dense(128, activation='relu')(m)
        m = Dense(128, activation='relu')(m)
        state_output = Dense(y_dim, activation='sigmoid', name='state_output')(m)
        model = Model(inputs=main_input, outputs=state_output)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        model.summary()
        x = {'main_input': data['XTrain']}
        y = {'state_output': data['YTrain']}
        validation_data = (data['XValid'], data['YValid'])

    else:
        print_note('{} is not valid model type.'.format(model_type))

    start_time = datetime.datetime.now()
    history = model.fit(x, y, batch_size=4096, epochs=epochs, verbose=verbose, validation_data=validation_data,
                        shuffle=True)

    val_loss_arr = np.array(history.history['val_loss'])
    note = 'Training loss:{0:.6f} Best validation loss:{1:.6f} at epoch {2}. [{3}]'\
        .format(history.history['loss'][epochs - 1], val_loss_arr.min(), np.argmin(val_loss_arr), datetime.datetime.now() - start_time)
    print_note(note)

    if model_type == 'interaction_plus':
        val_loss_arr_extra = np.array(history.history['val_state_output_loss'])
        note = '-- State output: Training loss:{0:.6f} Lowest validation loss:{1:.6f} at epoch {2}'\
               .format(history.history['state_output_loss'][epochs - 1],
                       val_loss_arr_extra.min(), np.argmin(val_loss_arr_extra))
        print_note(note)

        val_loss_arr_extra = np.array(history.history['val_env_output_loss'])
        note = '-- Env_vec output: Training loss:{0:.6f} Lowest validation loss:{1:.6f} at epoch {2}'\
               .format(history.history['env_output_loss'][epochs - 1],
                       val_loss_arr_extra.min(), np.argmin(val_loss_arr_extra))
        print_note(note)
    elif model_type == 'interaction_separation':
        val_loss_arr_extra = np.array(history.history['val_state_output_loss'])
        note = '-- State output: Training loss:{0:.6f} Lowest validation loss:{1:.6f} at epoch {2}' \
            .format(history.history['state_output_loss'][epochs - 1],
                    val_loss_arr_extra.min(), np.argmin(val_loss_arr_extra))
        print_note(note)

        val_loss_arr_extra = np.array(history.history['val_encoder_output_loss'])
        note = '-- Env_vec output: Training loss:{0:.6f} Lowest validation loss:{1:.6f} at epoch {2}' \
            .format(history.history['encoder_output_loss'][epochs - 1],
                    val_loss_arr_extra.min(), np.argmin(val_loss_arr_extra))
        print_note(note)

    if plot and logger is not None:
        if model_type == 'interaction_plus' or model_type == 'interaction_separation':
            plt.plot(history.history['state_output_loss'][300:], label='Training')
            plt.plot(history.history['val_state_output_loss'][300:], label='Validation')
        else:
            plt.plot(history.history['loss'][300:], label='Training')
            plt.plot(history.history['val_loss'][300:], label='Validation')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.dirname(logger._text_outputs[0]) + '/' + name + '.png')
        plt.close()

    return model, history


class PredictionModel(object):
    def __init__(self, dir):
        self.paths = []
        self.env_vecs = []
        self.init_paths = []
        self.init_env_vecs = []
        self.update_freq = 50
        self.saving_freq = 10
        self.update_batch_size = 10000  # required number of trajectories, 10000 for itr 0, then 5000
        self.filepath = dir

        self.data = dict()
        self.x_dict = dict()
        self.y_dict = dict()

        folder_name = EPI.ENV + '_' + str(EPI.NUM_OF_PARAMS)
        if os.path.exists(os.path.dirname(__file__) + '/models/' + folder_name + '/x_scaler.pkl'):
            self.x_scaler = pickle.load(open(os.path.dirname(__file__) + '/models/' + folder_name + '/x_scaler.pkl', 'rb'))
        else:
            self.x_scaler = None

        if os.path.exists(os.path.dirname(__file__) + '/models/' + folder_name + '/y_scaler.pkl'):
            self.y_scaler = pickle.load(open(os.path.dirname(__file__) + '/models/' + folder_name + '/y_scaler.pkl', 'rb'))
        else:
            self.y_scaler = None

        self.load_data(os.path.dirname(__file__) + '/models/' + folder_name + '/data_vine.csv')

        self.i_scaler = None

        self.baseline_score = dict()
        self.model_baseline = None
        self.model = None
        self.encoder = None
        self.saved_model_path = None
        self.saved_i_scaler_path = None

        self.embedding_list = []
        self.training_history = None

    def load_models(self, logger=None):
        # Load model after tf.global_variables_initializer(), otherwise weights will be cleared.
        folder_name = EPI.ENV + '_' + str(EPI.NUM_OF_PARAMS)
        if os.path.exists(os.path.dirname(__file__) + '/models/' + folder_name + '/mlp_avg.h5'):
            self.model_baseline = load_model(os.path.dirname(__file__) + '/models/' + folder_name + '/mlp_avg.h5')
        else:
            self.model_baseline, self.training_history = train('mlp_avg', self.data, model_type='avg', plot=True, verbose=0, logger=logger)
            self.model_baseline.save(os.path.dirname(__file__) + '/models/' + folder_name + '/mlp_avg.h5')

        for k in range(len(self.data['EValid'])):
            env_vec = self.data['EValid'][k]
            if env_vec not in self.baseline_score.keys():
                idx_filter = (self.data['EValid'] == env_vec)
                x_arr = self.data['XValid'][idx_filter, :]
                y_arr = self.data['YValid'][idx_filter, :]
                pred_raw = self.model_baseline.predict({'main_input': x_arr})
                self.baseline_score[env_vec] = np.mean(np.square(pred_raw - y_arr))

        if self.saved_model_path is not None:
            self.model = load_model(self.saved_model_path)
            self.encoder = Model(inputs=self.model.get_layer('env_input').input,
                                 outputs=self.model.get_layer('encoder_output').output)
            self.i_scaler = pickle.load(open(self.saved_i_scaler_path, 'rb'))

    def save_trajectory(self, interaction_traj, env_vec):
        self.paths.append(interaction_traj)
        self.env_vecs.append(env_vec)

    def load_trajectory(self, filename):
        self.paths, self.env_vecs = pickle.load(open(filename, 'rb'))

    def update(self, itr, logger=None):

        logger.log('Num of paths:'+str(len(self.paths)))
        paths_train, paths_val, env_vec_train, env_vec_val = train_test_split(self.paths, self.env_vecs,
                                                                              test_size=0.2, random_state=5)

        ITrain = self.match_interaction(self.data['ETrain'], paths_train, env_vec_train)
        IValid = self.match_interaction(self.data['EValid'], paths_val, env_vec_val)
        self.i_scaler = preprocessing.StandardScaler().fit(ITrain)
        ITrain = self.i_scaler.transform(ITrain)
        IValid = self.i_scaler.transform(IValid)
        self.data['ITrain'] = ITrain
        self.data['IValid'] = IValid

        self.model, history = train('itr'+str(itr), self.data, model_type=EPI.LOSS_TYPE, plot=True, verbose=0, logger=logger)
        self.encoder = Model(inputs=self.model.get_layer('env_input').input,
                             outputs=self.model.get_layer('encoder_output').output)
        self.model.save(self.filepath+'/mlp_i_itr_' + str(itr) + '.h5')
        self.saved_model_path = self.filepath+'/mlp_i_itr_' + str(itr) + '.h5'
        pickle.dump(self.i_scaler, open(self.filepath+'/i_scaler_itr_' + str(itr) + '.pkl', 'wb'))
        self.saved_i_scaler_path = self.filepath+'/i_scaler_itr_' + str(itr) + '.pkl'

        self.update_batch_size = 5000
        return

    def load_data(self, data_file):
        data = pd.read_csv(data_file, index_col=0).values

        if EPI.ENV == 'striker':
            ob = data[:, 0:7]  # Observation
            obn = data[:, 7:14]  # Next Observation
            env_vec = data[:, 14]

            x = ob
            y = obn - ob
            e = env_vec
        elif EPI.ENV == 'hopper':
            ob = data[:, 0:11]  # Observation
            ac = data[:, 11:14]  # Action
            obn = data[:, 14:25]  # Next Observation
            env_vec = data[:, 25]

            x = np.append(ob, ac, axis=1)
            y = obn - ob
            e = env_vec
        else:
            print('Env not defined in prediction_model.py.')
            raise ValueError

        x_train, x_val, y_train, y_val, e_train, e_val = train_test_split(x, y, e, test_size=0.2, random_state=10)

        folder_name = EPI.ENV + '_' + str(EPI.NUM_OF_PARAMS)
        if self.x_scaler is None:
            self.x_scaler = preprocessing.StandardScaler().fit(x_train)
            pickle.dump(self.x_scaler, open(os.path.dirname(__file__) + '/models/' + folder_name + '/x_scaler.pkl', 'wb'))

        if self.y_scaler is None:
            self.y_scaler = preprocessing.StandardScaler().fit(y_train)
            y_temp = self.y_scaler.transform(y_train)
            self.y_scaler.scale_ = self.y_scaler.scale_ * np.max(abs(y_temp), axis=0) * 2  # additional scaling for sigmoid
            pickle.dump(self.y_scaler, open(os.path.dirname(__file__) + '/models/' + folder_name + '/y_scaler.pkl', 'wb'))

        x_train = self.x_scaler.transform(x_train)
        x_val = self.x_scaler.transform(x_val)
        y_train = self.y_scaler.transform(y_train)+0.5
        y_val = self.y_scaler.transform(y_val)+0.5

        self.data = {'XTrain': x_train, 'XValid': x_val, 'YTrain': y_train, 'YValid': y_val,
                     'ETrain': e_train, 'EValid': e_val}

        return

    @staticmethod
    def match_interaction(data_env_vec, paths, env_vecs):
        # match interaction in the order of the given env_vec
        interaction_data = paths
        intercation_env_vec = env_vecs
        interaction_vec = []
        for i in range(len(data_env_vec)):
            if np.sum(intercation_env_vec == data_env_vec[i]) > 0:
                while True:
                    idx = np.random.randint(0, len(intercation_env_vec))
                    if intercation_env_vec[idx] == data_env_vec[i]:
                        interaction_vec.append(interaction_data[idx])
                        break
            else:
                print('No matching interaction.')
        return np.vstack(interaction_vec)

    # Evaluation
    def get_score(self, traj, env_vec):

        idx_filter = (self.data['EValid'] == env_vec)

        x_arr = self.data['XValid'][idx_filter, :]
        y_arr = self.data['YValid'][idx_filter, :]
        i_arr = np.vstack([traj]*len(x_arr))

        pred_raw = self.model.predict({'main_input': x_arr, 'env_input': self.i_scaler.transform(i_arr)})

        if EPI.LOSS_TYPE == 'interaction':
            score = np.mean(np.square(pred_raw - y_arr))
        elif EPI.LOSS_TYPE == 'interaction_separation':
            score = np.mean(np.square(pred_raw[0] - y_arr))
        else:
            print('prediction_model.py: def get_score.')
            embed()
        score = self.baseline_score[env_vec] - score

        return np.clip(score*1e5, 0, None)

    def save_embedding(self, traj, env_vec):
        # Save trajectory for analysis
        embedding = self.encoder.predict({'env_input': self.i_scaler.transform(np.vstack([traj]))}).reshape(-1)
        self.embedding_list.append(np.concatenate([embedding, np.array([env_vec])]))

    def evaluate_embedding(self, title):
        if EPI.EMBEDDING_DIMENSION == 2:  # Striker(embedding=2)
            # 2-dimensions
            f, (ax1) = plt.subplots(1, sharex=True, sharey=False, figsize=[4, 3])
            plt.gcf().subplots_adjust(left=0.3)
            arr = np.vstack(self.embedding_list)
            cmap = matplotlib.cm.get_cmap('Spectral')
            for i in range(EPI.NUM_OF_ENVS):
                label = 'm:%.f f:%.f' % (i / 5, i % 5)
                ax1.scatter(arr[(arr[:, -1] == i), 0], arr[(arr[:, -1] == i), 1], label=label,
                            color=cmap(i / EPI.NUM_OF_ENVS), s=20)
            plt.savefig(self.filepath + '/' + title + '.png')
            plt.close()

        elif EPI.NUM_OF_PARAMS == 8:  # Hopper
            scale_list = pd.read_csv(os.path.dirname(__file__) + '/../envs/' + EPI.ENV + '_env_list.csv').values

            # tsne
            f, (ax1) = plt.subplots(1, sharex=True, sharey=False, figsize=[4, 3])
            plt.gcf().subplots_adjust(left=0.3)
            arr = np.vstack(self.embedding_list)
            cmap = matplotlib.cm.get_cmap('hsv')
            env_ids = arr[:, -1].copy()

            tsne = TSNE()
            tsne_result = tsne.fit_transform(arr[:, :-1])
            for i in range(EPI.NUM_OF_ENVS):
                scale = scale_list[i, 1:]
                r = (np.mean(scale[0:4])+0.1)*2
                g = (np.mean(scale[4])+0.1)*2
                b = (np.mean(scale[5:])+0.1)*2
                ax1.scatter(tsne_result[(env_ids[:] == i), 0], tsne_result[(env_ids[:] == i), 1], color=(r,g,b), s=20)
            plt.savefig(self.filepath+'/'+title+'_tsne.png')
            plt.close()

            # pca
            f, (ax1) = plt.subplots(1, sharex=True, sharey=False, figsize=[4, 3])
            plt.gcf().subplots_adjust(left=0.3)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(arr[:, :-1])
            for i in range(EPI.NUM_OF_ENVS):
                scale = scale_list[i, 1:]
                r = (np.mean(scale[0:4])+0.1)*2
                g = (np.mean(scale[4])+0.1)*2
                b = (np.mean(scale[5:])+0.1)*2
                ax1.scatter(pca_result[(env_ids[:] == i), 0], pca_result[(env_ids[:] == i), 1], color=(r,g,b), s=20)
            plt.savefig(self.filepath+'/'+title+'_pca.png')
            plt.close()

        elif EPI.NUM_OF_PARAMS == 2:  # Striker(embedding=8)
            # tsne
            f, (ax1) = plt.subplots(1, sharex=True, sharey=False, figsize=[4, 3])
            plt.gcf().subplots_adjust(left=0.3)
            arr = np.vstack(self.embedding_list)
            cmap = matplotlib.cm.get_cmap('Spectral')

            tsne = TSNE()
            tsne_result = tsne.fit_transform(arr[:, :-1])
            for i in range(EPI.NUM_OF_ENVS):
                label = 'm:%.f f:%.f' % (i / 5, i % 5)
                ax1.scatter(tsne_result[(arr[:, -1] == i), 0], tsne_result[(arr[:, -1] == i), 1], label=label,
                            color=cmap(i / EPI.NUM_OF_ENVS), s=20)

            plt.savefig(self.filepath + '/' + title + '_tsne.png')
            plt.close()

            # pca
            f, (ax1) = plt.subplots(1, sharex=True, sharey=False, figsize=[4, 3])
            plt.gcf().subplots_adjust(left=0.3)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(arr[:, :-1])
            for i in range(EPI.NUM_OF_ENVS):
                label = 'm:%.f f:%.f' % (i / 5, i % 5)
                ax1.scatter(pca_result[(arr[:, -1] == i), 0], pca_result[(arr[:, -1] == i), 1], label=label,
                            color=cmap(i / EPI.NUM_OF_ENVS), s=20)
            plt.savefig(self.filepath + '/' + title + '_pca.png')
            plt.close()

        elif EPI.ENV == 'pendulum':
            scale_list = pd.read_csv(os.path.dirname(__file__) + '/../envs/' + EPI.ENV + '_env_list.csv').values

            # tsne
            f, (ax1) = plt.subplots(1, sharex=True, sharey=False, figsize=[4, 3])
            plt.gcf().subplots_adjust(left=0.3)
            arr = np.vstack(self.embedding_list)
            cmap = matplotlib.cm.get_cmap('hsv')
            env_ids = arr[:, -1].copy()

            tsne = TSNE()
            tsne_result = tsne.fit_transform(arr[:, :-1])

            # embed()
            for i in range(EPI.NUM_OF_ENVS):
                scale = scale_list[i, 1:]
                # r = np.clip((np.mean(scale[1:3]) + 0.1) * 2, 0, 1)
                # g = np.clip((np.mean(scale[4:6]) + 0.1) * 2, 0, 1)
                # b = np.clip(((scale[0]+scale[3])/2 + 0.1) * 2, 0, 1)
                r = np.clip((np.mean(scale[0:3])+0.1)*2, 0, 1)
                g = np.clip((np.mean(scale[3:6]) + 0.1) * 2, 0, 1)
                b = 0.3
                ax1.scatter(tsne_result[(env_ids[:] == i), 0], tsne_result[(env_ids[:] == i), 1], color=(r,g,b), s=20)
            plt.savefig(self.filepath+'/'+title+'_tsne.pdf')
            plt.close()

            # pca
            f, (ax1) = plt.subplots(1, sharex=True, sharey=False, figsize=[4, 3])
            plt.gcf().subplots_adjust(left=0.3)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(arr[:, :-1])
            for i in range(EPI.NUM_OF_ENVS):
                scale = scale_list[i, 1:]
                r = np.clip((np.mean(scale[0:3])+0.1)*2, 0, 1)
                g = np.clip((np.mean(scale[3:6]) + 0.1) * 2, 0, 1)
                b = 0.3
                # r = np.clip((np.mean(scale[1:3]) + 0.1) * 2, 0, 1)
                # g = np.clip((np.mean(scale[4:6]) + 0.1) * 2, 0, 1)
                # b = np.clip(((scale[0]+scale[3])/2 + 0.1) * 2, 0, 1)
                ax1.scatter(pca_result[(env_ids[:] == i), 0], pca_result[(env_ids[:] == i), 1], color=(r,g,b), s=20)
            plt.savefig(self.filepath+'/'+title+'_pca.pdf')
            plt.close()

        self.embedding_list = []
        return
