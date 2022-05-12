import os
from multiprocessing.pool import Pool

from tensorflow_core.python.keras.saving.save import load_model

from onto.learner_env import LearnverEnv
from onto.sim_gonogo import sim_gonogo
from util import DLogger

learner_path = '../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5'
base_dirs = '/scratch1/dez004/RL_gonogo_dqn_vec/'
dirs = [[o, os.path.join(base_dirs, o)] for o in os.listdir(base_dirs) if os.path.isdir(os.path.join(base_dirs, o))]
dirs_iter = []
for d in dirs:
    for i in [400000, 300000, 200000, 150000, 100000, 50000]:
        dd = d.copy()
        dd.append(str(i))
        dirs_iter.append(dd)

def run_sim(i):
    try:
        adv_path = dirs_iter[i][1] + '/model-' + dirs_iter[i][2] + '.h5'
        output_path = '/scratch1/dez004/RL_gonogo_dqn_vec_sim/gonog_sim_' + dirs_iter[i][2] + '/' + dirs_iter[i][0] + '/'
        DLogger.logger().debug("Learner model loaded from path {}".format(learner_path))
        learner_model = load_model(learner_path, compile=False)
        le = LearnverEnv(learner_model, 2, 2, 4000)
        DLogger.logger().debug("Adv model loaded from path {}".format(adv_path))
        adv_model = load_model(adv_path, compile=False)
        sim_gonogo(le, adv_model, output_path)

    except:
        DLogger.logger().debug("Exception for " + str(i))


def run(f, n_proc):
    p = Pool(n_proc)
    p.map(f, range(len(dirs_iter)))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':
    run(run_sim, 40)

    # import numpy as np
    # np.random.seed(1010)
    # output_path = '../nongit/temp/'
    # adv_path = '../nongit/archive/gonogo/onto-cv/temp2/RL_nc_dqn_buf_600000_eps_0.1_lr_0.001/model-372000.h5'
    # learner_path = '../models/archive/learner/gonogo/onto/gonogo_learner_single_5cell/model-9300.h5'
    #
    # DLogger.logger().debug("Learner model loaded from path {}".format(learner_path))
    # learner_model = load_model(learner_path, compile=False)
    # le = LearnverEnv(learner_model, 2, 2, 400)
    # DLogger.logger().debug("Adv model loaded from path {}".format(adv_path))
    # adv_model = load_model(adv_path, compile=False)
    # sim_gonogo(le, adv_model, '../nongit/temp/')
    #
