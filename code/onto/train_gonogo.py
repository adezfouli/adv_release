from data_reader import DataReader
from rnn_learner import RNNAgent
from util.helper import fix_seeds
from util.logger import LogFile, DLogger
import numpy as np
import sys
import pandas as pd

n_folds = 10
confs = []
cells = [2, 3, 4, 5, 8, 10, 15]

for f in range(n_folds):
    for c in range(len(cells)):
        confs.append({'cells': cells[c], 'fold': f})

def data_to_sar(data):
    sh = data['state'].shape

    DLogger.logger().debug('Data dims: ' + str(sh))

    state = np.zeros((sh[0], 2 * sh[1], sh[2]), np.float32)
    reward = np.zeros((sh[0], 2 * sh[1]), np.float32)
    action = -1 * np.ones((sh[0], 2 * sh[1]), np.float32)

    state[:, range(0, 2 * sh[1], 2)] = data['state']
    reward[:, range(0, 2 * sh[1], 2)] = data['reward']
    action[:, range(0, 2 * sh[1], 2)] = data['action']
    return state, action, reward


if __name__ == '__main__':

    if len(sys.argv) == 2:
        chunk = int(sys.argv[1])
    else:
        raise Exception("number of cells is required")

    # fix_seeds()

    n_cells = confs[chunk]['cells']
    fold = confs[chunk]['fold']
    output_path = '/scratch1/dez004/gonogo_learner_sreg/cells_' + str(n_cells) + '/fold_' + str(fold) + '/'
    np.random.seed(1010)
    with LogFile(output_path, 'run.log'):

        data = DataReader.read_gonogo_random_local()

        state, action, reward = data_to_sar(data)

        model_path = '../models/inits/gonogo/learner_go_nogo_cells_' + str(n_cells) + '/model-init.h5'

        agent = RNNAgent(2, 2, n_cells, reset_after=False, model_path=model_path)

        indices = np.random.permutation(reward.shape[0])
        folds = np.array_split(indices, n_folds, axis=0)
        training_idx = np.concatenate(folds[(fold+1):] + folds[:fold])
        test_idx = folds[fold]
        ids =   pd.concat([pd.DataFrame({'type': 'train', 'index': training_idx}),
                pd.DataFrame({'type': 'test', 'index': test_idx})])
        ids.to_csv(output_path + 'indices.csv')

        agent.train(
            reward[training_idx],
            action[training_idx],
            state[training_idx],
            reward[test_idx],
            action[test_idx],
            state[test_idx],
            output_path)
