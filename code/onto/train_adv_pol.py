import multiprocessing
import os

from onto.learner_env import LearnverEnv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from multiprocessing.pool import Pool

import numpy as np
from tensorflow.python.keras.saving import load_model

from rl.pol_grad import A2CAgent
from util.logger import LogFile, DLogger

configs = []

for l in [2]:
    for units in [256]:
        for epsilon in [0.01, 0.05]:
            for lof in [0.1, 0.0]:
                configs.append({'l': l, 'u': units, 'eps': epsilon, 'lof': lof})


"""
This file can be used for training the adversary based on a trained RNN.
"""
def run_adv(i):
    # fix_seeds()

    c = configs[i]
    layers = c['l']
    units = c['u']
    eps = c['eps']
    lof = c['lof']

    DLogger.logger().debug('config: ')
    DLogger.logger().debug(c)

    np.set_printoptions(precision=3)

    learner_model_path = '../models/archive/learner/gonogo/sreg/gonogo_learner_cells_sreg_8/model-11600.h5'
    p_adv_model_path = None
    q_adv_model_path = None

    output_path = '../nongit/archive/gonogo/state-reg/RL_gonogo_static-1/RL_gonogo_' + str(layers) + 'layers_' + str(eps) + 'ent_' + str(
        units) + 'units' + '_lof' + str(lof) + '/'

    with LogFile(output_path, 'run.log'):
        DLogger.logger().debug("Learner model loaded from path {}".format(learner_model_path))
        model = load_model(learner_model_path)
        DLogger.logger().debug("Learner model:")
        model.summary(print_fn=DLogger.logger().debug)
        le = LearnverEnv(model, 2, 2, 500, LOF_weight=lof)
        le.reset()
        if p_adv_model_path is not None:
            DLogger.logger().debug("adv model loaded from " + p_adv_model_path)
            p_adv_model = load_model(p_adv_model_path, compile=False)
            q_adv_model = load_model(q_adv_model_path, compile=False)
        else:
            p_adv_model = None
            q_adv_model = None
        agent = A2CAgent(2, le.observation_space.shape[0], layers, units, eps,
                         p_model= p_adv_model,
                         q_model= q_adv_model,
                         lr=0.0001)
        agent.train(le, batch_sz=350, output_path=output_path)


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

    run(run_adv, 1, chunk)
