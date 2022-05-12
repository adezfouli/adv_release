import tensorflow as tf
from tensorflow_core.python.keras.models import load_model

from nc.adv_learner import AdvLearner
from nc.learner_env import LearnverEnv
from nc.sim_bandit import sim_bandit_env
from qrl_env import QRL_nc_ENV
from util.helper import fix_seeds

if __name__ == '__main__':
    fix_seeds()
    learner_path = '../models/archive/learner/nc/nc_ql_learner_single/cells_5/model-46700.h5'
    adv_path = '../nongit/archive/nc/ql/run-cv/RL_ql_dqn_vec/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-500000.h5'
    output_path = '../nongit/archive/nc/cross-sims/ql_adv_vs_human_lrn/'
    real_model = LearnverEnv(load_model('../models/archive/learner/nc/nc_human_learner_single/cells_10/model-1100.h5'), 2, 1)
    bandit_env = AdvLearner(learner_path, adv_path, output_path, real_model)
    sim_bandit_env(bandit_env, output_path)

    fix_seeds()
    learner_path = '../models/archive/learner/nc/nc_ql_learner_single/cells_5/model-46700.h5'
    adv_path = '../nongit/archive/nc/ql/run-cv/RL_ql_dqn_vec/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/model-500000.h5'
    output_path = '../nongit/archive/nc/cross-sims/ql_adv_vs_ql_lrn/'
    real_model = LearnverEnv(load_model('../models/archive/learner/nc/nc_ql_learner_single/cells_5/model-46700.h5'), 2, 1)
    bandit_env = AdvLearner(learner_path, adv_path, output_path, real_model)
    sim_bandit_env(bandit_env, output_path)

    fix_seeds()
    learner_path = '../models/archive/learner/nc/nc_human_learner_single/cells_10/model-1100.h5'
    adv_path = '../models/archive/RL/nc/RL_human_dqn_vec/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/model-600000.h5'
    output_path = '../nongit/archive/nc/cross-sims/human_adv_vs_ql_lrn/'
    real_model = LearnverEnv(load_model('../models/archive/learner/nc/nc_ql_learner_single/cells_5/model-46700.h5'), 2, 1)
    bandit_env = AdvLearner(learner_path, adv_path, output_path, real_model)
    sim_bandit_env(bandit_env, output_path)


    fix_seeds()
    learner_path = '../models/archive/learner/nc/nc_human_learner_single/cells_10/model-1100.h5'
    adv_path = '../models/archive/RL/nc/RL_human_dqn_vec/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/model-600000.h5'
    output_path = '../nongit/archive/nc/cross-sims/human_adv_vs_human_lrn/'
    real_model = LearnverEnv(load_model('../models/archive/learner/nc/nc_human_learner_single/cells_10/model-1100.h5'), 2, 1)
    bandit_env = AdvLearner(learner_path, adv_path, output_path, real_model)
    sim_bandit_env(bandit_env, output_path)
