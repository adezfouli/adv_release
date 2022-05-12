from data_reader import DataReader
from rnn_learner import RNNAgent
from util.helper import fix_seeds
from util.logger import LogFile, DLogger
import numpy as np
import sys

confs = []
cells = [2,3,4,5, 8, 10, 15]

for c in range(len(cells)):
    confs.append({'cells': cells[c]})

if __name__ == '__main__':

    if len(sys.argv) == 2:
        chunk = int(sys.argv[1])
    else:
        raise Exception("number of cells is required")

    # fix_seeds()

    n_cells = confs[chunk]['cells']
    output_path = '/scratch1/dez004/gonogo_learner_cells_sreg_' + str(n_cells) + '/'
    np.random.seed(1010)
    with LogFile(output_path, 'run.log'):

        data = DataReader.read_gonogo_random_local()

        # saperating state from action
        sh = data['state'].shape

        DLogger.logger().debug('Data dims: ' + str(sh))

        state = np.zeros((sh[0], 2 * sh[1], sh[2]), np.float32)
        reward = np.zeros((sh[0], 2 * sh[1]), np.float32)
        action = -1 * np.ones((sh[0], 2 * sh[1]), np.float32)

        state[:, range(0, 2 * sh[1], 2)] = data['state']
        reward[:, range(0, 2 * sh[1], 2)] = data['reward']
        action[:, range(0, 2 * sh[1], 2)] = data['action']

        model_path = '../models/inits/gonogo/learner_go_nogo_cells_' + str(n_cells) + '/model-init.h5'
        agent = RNNAgent(2, 2, n_cells, reset_after=False, model_path=model_path)

        agent.train(
            reward,
            action,
            state,
            reward,
            action,
            state,
            output_path)
