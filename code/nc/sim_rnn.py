import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Concatenate, ZeroPadding1D
from tensorflow_core.python.keras.models import load_model

from util.helper import ensure_dir


def sim_rnn(model_path, data, export_path, file_name, n_cells):
    model = load_model(model_path)
    policies = model([data, np.zeros((data.shape[0], n_cells,), dtype=np.float32)])[1]
    ensure_dir(export_path)
    pd.DataFrame(policies[0].numpy()).to_csv(export_path + file_name)

def get_data(input_path):
    dataset = pd.read_csv(input_path, header=0, sep=',', quotechar='"', keep_default_na=False)
    action = np.zeros((dataset.shape[0], 2))
    action[np.array(dataset['rnn action']) == '[[1 0]]'] = np.array([[1, 0]])
    action[np.array(dataset['rnn action']) == '[[0 1]]'] = np.array([[0, 1]])
    reward = np.concatenate((dataset['r1'][:, np.newaxis], dataset['r2'][:, np.newaxis]), axis=1)
    reward = (reward * action).sum(axis=1)
    action = tf.convert_to_tensor(action[np.newaxis], dtype=tf.float32)
    reward = tf.convert_to_tensor(reward[np.newaxis], dtype=tf.float32)
    # changing action to one-hot encoding
    action_reward = Concatenate(axis=2)([reward[:, :, np.newaxis], action])
    # added dummy zero to the beginning
    action_reward = ZeroPadding1D(padding=[1, 0])(action_reward)
    action_reward = action_reward[:, :-1, :]

    return action_reward


if __name__ == '__main__':
    for i in range(100):
        # for human
        data_path = '../nongit/archive/nc/human/run-cv/RL_human_dqn_vec_sim_sample/nc_sim_600000/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/events_' + str(i) + '.csv'
        model_path = '../models/archive/learner/nc/nc_human_learner_single/cells_10/model-1100.h5'
        output_path = '../nongit/archive/nc/human/run-cv/policies/'
        data = get_data(data_path)
        sim_rnn(model_path, data, output_path, 'policies_' + str(i) + '.csv', 10)


        # for ql
        data_path = '../nongit/archive/nc/ql/run-cv/RL_ql_dqn_vec_sim_rnn/nc_sim_500000/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/events_' + str(i) + '.csv'
        model_path = '../models/archive/learner/nc/nc_ql_learner_single/cells_5/model-46700.h5'
        output_path = '../nongit/archive/nc/ql/run-cv/policies/'
        data = get_data(data_path)
        sim_rnn(model_path, data, output_path, 'policies_' + str(i) + '.csv', 5)
