import os
from multiprocessing.pool import Pool
from nc.adv_learner import AdvLearner
from nc.sim_bandit import sim_bandit_env
from util import DLogger



"""
In this file trained learner in QRL and trained adversary for for that QRL are loaded 
and then simulated in the task each 400 times to assess the power of adversary against the trained RNN.

Note that the learner is a trained RNN to provide state and other information to the adversary, 
but the actual environment used for simulations is a Q learning using the same parameters 
used in NC paper.   
"""

learner_path = '../models/archive/learner/nc/nc_ql_learner_single/cells_5/model-46700.h5'
base_dirs = '/scratch1/dez004/RL_ql_dqn_vec/'
dirs = [[o, os.path.join(base_dirs, o)] for o in os.listdir(base_dirs) if os.path.isdir(os.path.join(base_dirs, o))]

dirs_iter = []
for d in dirs:
    for i in [900000, 800000, 700000, 600000, 500000, 400000, 300000, 200000, 100000]:
        dd = d.copy()
        dd.append(str(i))
        dirs_iter.append(dd)


def run_sim(i):
    try:
        adv_path = dirs_iter[i][1] + '/model-' + dirs_iter[i][2] + '.h5'
        output_path = '/scratch1/dez004/RL_ql_dqn_vec_sim_rnn/nc_sim_' + dirs_iter[i][2] + '/' + dirs_iter[i][0] + '/'
        bandit_env = AdvLearner(learner_path, adv_path, output_path, None)
        sim_bandit_env(bandit_env, output_path)
    except:
        DLogger.logger().debug("Exception for " + str(i))


def run(f, n_proc):
    p = Pool(n_proc)
    p.map(f, range(len(dirs_iter)))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':
    run(run_sim, 40)
