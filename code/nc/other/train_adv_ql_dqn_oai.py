import gym

from baselines import deepq, logger

import multiprocessing
import os

from nc.learner_env import LearnverEnv
from util.helper import get_git

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from multiprocessing.pool import Pool
import numpy as np
from tensorflow.python.keras.saving import load_model


class Env:
    def __init__(self, le_env):
        self.observation_space = le_env.observation_space
        self.action_space = le_env.action_space
        self.le_end = le_env

    def reset(self):
        self.le_end.reset()
        state, _, _, _ = self.le_end.step_adv(0)
        return state

    def step(self, action):
        return self.le_end.step_adv(action)


configs = []

for b in [1000, 10000, 50000, 100000]:
    for lr in [0.001, 0.0001, 1e-5]:
        for eps in [0.5, 0.1, 0.2]:
            configs.append({'b': b, 'lr': lr, 'eps': eps})


def run_adv(i):
    # fix_seeds()

    np.set_printoptions(precision=3)

    buf = configs[i]['b']
    lr = configs[i]['lr']
    eps = configs[i]['eps']

    learner_model_path = '../models/archive/learner/nc/nc_ql_learner_single/cells_5/model-46700.h5'
    output_path =  '/scratch1/dez004/RL_ql_dqn_oai/RL_nc_dqn_buf_' + str(buf)+ '_eps_' + str(eps) + '_lr_' + str(lr) + '/'
    logger.configure(dir=output_path)

    logger.debug('config: ')
    np.set_printoptions(precision=3)

    logger.log("version control: " + str(get_git()))
    logger.log("Learner model loaded from path {}".format(learner_model_path))
    model = load_model(learner_model_path)
    logger.log("Learner model:")
    model.summary(print_fn=logger.log)
    le = LearnverEnv(model, 2)
    le.reset()
    deepq.learn(
        Env(le_env=le),
        network='mlp',
        lr=lr,
        total_timesteps=10000000,
        buffer_size=buf,
        exploration_fraction=eps,
        exploration_final_eps=0.02,
        print_freq=10,
        checkpoint_path=output_path + '/model/',
        checkpoint_freq=100000
    )


def run(f, n_proc, chunk):
    p = Pool(n_proc)
    start = min(len(configs), (chunk - 1) * n_proc)
    end = min(len(configs), chunk * n_proc)
    p.map(f, range(start, end))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':

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
