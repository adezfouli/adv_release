import sys

from data_reader import DataReader
from rnn_learner import RNNAgent
from util.helper import fix_seeds
from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd


if __name__ == '__main__':

    # fix_seeds()
    n_cells = 10
    output_path = '/scratch1/dez004/nc_human_learner_single/cells_' + str(n_cells) + '/'
    np.random.seed(1010)
    with LogFile(output_path, 'run.log'):

        data = DataReader.read_nc_data()
        sh = data['action'].shape

        DLogger.logger().debug('Data dims: ' + str(sh))

        # for synthetic data using their Q learning model
        # data = DataReader.read_nc_data()
        model_path = '../models/inits/nc/learner_nc_cells_' + str(n_cells) + '/model-init.h5'
        agent = RNNAgent(2, 0, n_cells, model_path=model_path)

        agent.train(
            data['reward'],
            data['action'],
            data['state'] if 'state' in data else None,
            data['reward'],
            data['action'],
            data['state'] if 'state' in data else None,
            output_path,
            lr=0.005
        )
