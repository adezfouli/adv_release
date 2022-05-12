import multiprocessing
import os

from data_reader import DataReader
from onto.density_rnn import DensRNN
from onto.learner_env import LearnverEnv
from rl.ddqn import DQNAgent
from util.helper import fix_seeds

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from multiprocessing.pool import Pool
import numpy as np
from tensorflow.python.keras.saving import load_model
from util.logger import LogFile, DLogger

configs = []

for b in [100000]:
    for lr in [0.0001]:
        for eps in [0.02]:
            for l in [0.1]:
                configs.append({'b': b, 'lr': lr, 'eps': eps, 'l': l})


def run_adv(i):
    fix_seeds()

    np.set_printoptions(precision=3)

    buf = configs[i]['b']
    lr = configs[i]['lr']
    eps = configs[i]['eps']
    l = configs[i]['l']

    learner_model_path = '../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5'
    # output_path =  '/scratch1/dez004/RL_gonogo_dqn_vec/RL_nc_dqn_buf_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '_lof_' + str(l) + '/'

    output_path = '../nongit/archive/gonogo/state-reg/RL_gonogo_dqn_vec_static/RL_nc_dqn_buf_' + \
                   str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '_lof_' + str(l) + '/'


    DLogger.logger().debug('config: ')
    np.set_printoptions(precision=3)

    with LogFile(output_path, 'run.log'):
        # lof, cells = DensRNN.get_lof(DataReader.read_gonogo_random_local(), learner_model_path)
        DLogger.logger().debug("Learner model loaded from path {}".format(learner_model_path))
        model = load_model(learner_model_path)
        DLogger.logger().debug("Learner model:")
        model.summary(print_fn=DLogger.logger().debug)
        le = LearnverEnv(model, 2, 2, 100, LOF=None, LOF_weight=l)
        le.reset()
        agent = DQNAgent(le.observation_space.shape[0], 2, buf, epsilon=eps, lr=lr)
        agent.train(env=le, output_path=output_path, batch_size=100, total_episodes=int(1e10))


def run(f, n_proc, chunk):
    p = Pool(n_proc)
    start = min(len(configs), (chunk - 1) * n_proc)
    end = min(len(configs), chunk * n_proc)
    p.map(f, range(start, end))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':

    # run_adv(0)
    # run(run_adv, 1)
    #
    if len(sys.argv) == 2:
        chunk = int(sys.argv[1])
    else:
        raise Exception("invalid processing chunk")
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        print("SLURM_JOB_CPUS_PER_NODE" + " found")
        print("CPU: " +str(ncpus))
    except KeyError:
        ncpus = multiprocessing.cpu_count()
        print("SLURM_JOB_CPUS_PER_NODE" + " not found")
        print("CPU: " +str(ncpus))

    run(run_adv, ncpus, chunk)