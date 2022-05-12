import os
from multiprocessing.pool import Pool
import tensorflow as tf
from tensorflow_core.python.keras.saving.save import load_model
from onto.adv_learner import AdvLearner
from onto.learner_env import LearnverEnv
from onto.sim_gonogo import sim_gonogo
from util import DLogger

learner_path = '../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5'
base_dirs = '../nongit/archive/gonogo/state-reg/RL_gonogo_static-1/'
dirs = [[o, os.path.join(base_dirs, o)] for o in os.listdir(base_dirs) if os.path.isdir(os.path.join(base_dirs, o))]
dirs_iter = []
for d in dirs:
    for i in [0, 500, 1000, 2000, 3000]:
        dd = d.copy()
        dd.append(str(i))
        dirs_iter.append(dd)

def run_sim(i):
    try:
        adv_path = dirs_iter[i][1] + '/p-model-' + dirs_iter[i][2] + '.h5'
        adv_model = load_model(adv_path, compile=False)
        output_path = '../nongit/archive/gonogo/state-reg/static-sims-1/gonog_sim_' + dirs_iter[i][2] + '/' + dirs_iter[i][0] + '/'
        learner_model = load_model(learner_path, compile=False)
        le = LearnverEnv(learner_model, 2, 2, 1500, LOF_weight=0)
        sim_gonogo(le, adv_model, output_path, stochastic='sto')
    except:
        DLogger.logger().debug("Exception for " + str(i))


def run(f, n_proc):
    p = Pool(n_proc)
    p.map(f, range(len(dirs_iter)))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':
    # run_sim(0)
    run(run_sim, 8)

    # folder = 'RL_gonogo_2layers_0.1ent_256units_lof0.0'
    # stochastic = 'sto'
    # adv_path = '../nongit/archive/gonogo/state-reg/RL_gonogo_dqn_vec_with_rew/' + folder + '/p-model-8500.h5'
    # adv_path = '../nongit/archive/gonogo/state-reg/RL_gonogo_dqn_vec_with_rew_cont/RL_gonogo_2layers_0.005725ent_256units_lof0.0/p-model-2500.h5'
    # output_path = '../nongit/temp/'
    # bandit_env = AdvLearner(learner_path, adv_path, output_path, None)
    # adv_model = load_model(adv_path, compile=False)
    # learner_model = load_model(learner_path, compile=False)
    # le = LearnverEnv(learner_model, 2, 2, 400, LOF_weight=0)
    # sim_gonogo(le, adv_model, output_path, stochastic=(stochastic == 'sto'))
