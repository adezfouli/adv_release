import os
from multiprocessing.pool import Pool

from tensorflow_core.python.keras.saving import load_model

from mrtt.learner_env import LearnverEnv
from mrtt.sim_rmrtt import sim_rmrtt
from nc.adv_learner import AdvLearner
from nc.sim_bandit import sim_bandit_env
from util import DLogger
import numpy as np



"""
In this file trained learner in QRL and trained adversary for for that QRL are loaded 
and then simulated in the task each 400 times to assess the power of adversary against the trained RNN.

Note that the learner is a trained RNN to provide state and other information to the adversary, 
but the actual environment used for simulations is a Q learning using the same parameters 
used in NC paper.   
"""

learner_path = '../models/archive/learner/rmrtt/mrtt_RND_learner_single/cells_3/model-700.h5'
base_dirs = '../nongit/archive/mrtt/RND/RL_rmrtt_dqn_RND/'

# for FAIR simulations
# base_dirs = '../nongit/archive/mrtt/RND-fair/RL_rmrtt_dqn_RND_fair_max/'


# uncomment for maximising trust
# base_dirs = '../nongit/archive/mrtt/Read/trust max/RL_rmrtt_dqn_vec/'
dirs = [[o, os.path.join(base_dirs, o)] for o in os.listdir(base_dirs) if os.path.isdir(os.path.join(base_dirs, o))]
dirs_iter = []
for d in dirs:
    for i in [
                200000,
                500000,
                1000000,
                1500000
                # 2000000,
                # 3000000
                # 700000,
                # 1000000,
                # 1500000,
                # 2000000,
                # 3000000,
    ]:
        dd = d.copy()
        dd.append(str(i))
        dirs_iter.append(dd)

def run_sim(i):
    # try:
        adv_path = dirs_iter[i][1] + '/model-' + dirs_iter[i][2] + '.h5'
        output_path = '../nongit/temp/archive/mrtt/RND/RL_rmrtt_dqn_RND_sim/rmrtt_sim_' + dirs_iter[i][2] + '/' + dirs_iter[i][0] + '/'

        # for FAIR case
        # output_path = '../nongit/archive/mrtt/RND-fair/RL_rmrtt_dqn_RND_fair_max_sim/rmrtt_sim_' + dirs_iter[i][2] + '/' + dirs_iter[i][0] + '/'

        # uncomment for maximising trust
        # output_path = '../nongit/archive/mrtt/Read/RL_rmrtt_dqn_vec_sim_total_trust/rmrtt_sim_' + dirs_iter[i][2] + '/' + dirs_iter[i][0] + '/'
        DLogger.logger().debug("Learner model loaded from path {}".format(learner_path))
        learner_model = load_model(learner_path, compile=False)

        # for FAIR case
        le = LearnverEnv(learner_model, 5, 15000, 10000, mode='fair_max')

        DLogger.logger().debug("Adv model loaded from path {}".format(adv_path))
        adv_model = load_model(adv_path, compile=False)
        sim_rmrtt(le, adv_model, output_path)

    # except:
    #     DLogger.logger().debug("Exception for " + str(i))


def run(f, n_proc):
    p = Pool(n_proc)
    p.map(f, range(len(dirs_iter)))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':
    # run(run_sim, 40)
    # for i in range(len(dirs_iter)):
    run_sim(0)


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
