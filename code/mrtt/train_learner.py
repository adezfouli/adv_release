import sys

from data_reader import DataReader
from rnn_learner import RNNAgent
from util.helper import fix_seeds
from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd

local = False

n_folds = 10
confs = []
cells = [2,3,4,5,8,10,15]

for f in range(n_folds):
    for c in range(len(cells)):
        confs.append({'cells': cells[c], 'fold': f})

if __name__ == '__main__':

    if not local:
        if len(sys.argv) == 2:
            chunk = int(sys.argv[1])
        else:
            raise Exception("number of cells is required")
    else:
        chunk = 0

    # fix_seeds()
    n_cells = confs[chunk]['cells']
    fold = confs[chunk]['fold']
    if not local:
        output_path = '/scratch1/dez004/deep_adv/mrtt_RND_learner/cells_' + str(n_cells) + '/fold_' + str(fold) + '/'
    else:
        output_path = '../nongit/dez004/deep_adv/mrtt_RND_learner/cells_' + str(n_cells) + '/fold_' + str(fold) + '/'

    np.random.seed(1010)
    with LogFile(output_path, 'run.log'):

        data = DataReader.read_MRTT_RND()

        sh = data['action'].shape

        DLogger.logger().debug('Data dims: ' + str(sh))


        # for synthetic data using their Q learning model
        # data = DataReader.read_nc_data()
        model_path = '../models/inits/mrtt/cells_' + str(n_cells) + '/model-init.h5'
        agent = RNNAgent(5, 0, n_cells, model_path=model_path)

        indices = np.random.permutation(sh[0])
        folds = np.array_split(indices, n_folds, axis=0)
        training_idx = np.concatenate(folds[(fold+1):] + folds[:fold])
        test_idx = folds[fold]
        ids =   pd.concat([pd.DataFrame({'type': 'train', 'index': training_idx}),
                pd.DataFrame({'type': 'test', 'index': test_idx})])
        ids.to_csv(output_path + 'indices.csv')
        agent.train(
            data['reward'][training_idx],
            data['action'][training_idx],
            data['state'][training_idx] if 'state' in data else None,
            data['reward'][test_idx],
            data['action'][test_idx],
            data['state'][test_idx] if 'state' in data else None,
            output_path,
            lr=0.005,
        )
