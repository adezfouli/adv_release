import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Concatenate, ZeroPadding1D
from tensorflow_core.python.keras.models import load_model

from util.helper import ensure_dir


def sim_rnn(model_path, data, export_path, file_name):
    model = load_model(model_path)
    policies = model([data, np.zeros((data.shape[0], 8,), dtype=np.float32)])[1]
    ensure_dir(export_path)
    pols = policies.numpy()[:, range(0, policies.shape[1], 2), :]
    pd.DataFrame(pols[0]).to_csv(export_path + file_name)

def get_data(input_path):
    dataset = pd.read_csv(input_path, header=0, sep=',', quotechar='"', keep_default_na=False)
    action = np.zeros((dataset.shape[0], 2))
    action[np.array(dataset['learner action']) == '[1 0]'] = np.array([1, 0])
    action[np.array(dataset['learner action']) == '[0 1]'] = np.array([0, 1])
    state = np.zeros((dataset.shape[0], 2))
    state[np.array(dataset['learner state']) == '[1. 0.]'] = np.array([1, 0])
    state[np.array(dataset['learner state']) == '[0. 1.]'] = np.array([0, 1])

    reward = np.zeros((dataset.shape[0],))
    reward[np.array(dataset['learner reward']) == 1] = 1
    reward[np.array(dataset['learner reward']) == 0] = 0

    action = tf.convert_to_tensor(action[np.newaxis], dtype=tf.float32)
    reward = tf.convert_to_tensor(reward[np.newaxis], dtype=tf.float32)
    # changing action to one-hot encoding
    action_reward = Concatenate(axis=2)([reward[:, :-1, np.newaxis], action[:, :-1, :]])
    # added dummy zero to the beginning
    action_reward = ZeroPadding1D(padding=[1, 0])(action_reward)
    data = Concatenate(axis=2)([action_reward, state[np.newaxis]])

    data2 = np.zeros((data.shape[0], 2 * data.shape[1], data.shape[2]))
    data2[:, range(0, 2 * data.shape[1], 2), 3:5] = data[:, :, 3:5]
    data2[:, range(1, 2 * data.shape[1] - 1 , 2), 0:3] = data[:, 1:, 0:3]

    return data2


if __name__ == '__main__':
    for i in range(100):
        data_path = '../nongit/archive/gonogo/state-reg/static-sims-1/gonog_sim_2000//RL_gonogo_2layers_0.01ent_256units_lof0.0/events_' + str(i) + '.csv'
        model_path = '../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5'
        output_path = '../nongit/archive/gonogo/state-reg/policies/'
        data = get_data(data_path)
        sim_rnn(model_path, data, output_path, 'policies_' + str(i) + '.csv')
