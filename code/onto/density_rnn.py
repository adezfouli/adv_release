import numpy as np

from data_reader import DataReader
from onto.train_gonogo import data_to_sar
from rnn_learner import RNNAgent
from tensorflow_core.python.keras.saving.save import load_model
from sklearn.neighbors import LocalOutlierFactor

from util import DLogger


class DensRNN:

    @classmethod
    def get_lof(cls, data, model_path):
        DLogger.logger().debug("LOF Load from model path " + model_path)
        DLogger.logger().debug("LOF Data shape " + str(data['state'].shape))
        state, action, reward = data_to_sar(data)
        action, inputs = RNNAgent.make_model_input(action, reward, state, 2)
        model = load_model(model_path)
        return DensRNN.get_density_estm(model, inputs)

    @classmethod
    def get_density_estm(cls, model, data):
        cells = model.get_layer('GRU').units
        rnn_states, rnn_pol = model.predict(
            [data, np.zeros((data.shape[0], cells,), dtype=np.float32)])

        st = np.reshape(rnn_states, (rnn_states.shape[0] * rnn_states.shape[1], rnn_states.shape[2]))

        lof = LocalOutlierFactor(novelty=True, n_neighbors=20)
        lof.fit(st)
        return lof, cells
        # dump(lof, '../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/lof.joblib')
        #
        # for testing
        # lof = load('../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/lof.joblib')
        # print(lof.predict(st))


if __name__ == '__main__':
    data = DataReader.read_gonogo_random_local()
    DensRNN.get_lof(data, '../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5')
