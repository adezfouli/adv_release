import h5py
import threading
import json
from flask import Flask, jsonify
from flask import request
from tensorflow_core.python.keras.saving import load_model

from mrtt.learner_env import LearnverEnv
from nc.adv_learner import AdvLearner
from util import DLogger
import numpy as np
import pandas as pd
from datetime import datetime
import time

from util.helper import ensure_dir, get_git
from util.logger import LogFile

lock = threading.Lock()
worker_model = {}
worker_data = {}
worker_trial = {}
worker_adv_action = {}
app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

learner_path = '../models/archive/learner/rmrtt/mrtt_RND_learner_single/cells_3/model-700.h5'

# max model
# adv_path = '../models/archive/RL/rnd-mrtt/RL_nc_dqn_buf_200000_eps_0.2_lr_0.001/model-500000.h5'

# fair model
adv_path = '../models/archive/RL/fair-mrt/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05/model-500000.h5'

output_path = '../nongit/results/subject_data/'
adv_model = load_model(adv_path, compile=False)

learner_path = h5py.File(learner_path, 'r')
adv_path = h5py.File(adv_path, 'r')

@app.route('/repay/')
def repay():
    worker_id = request.args.get('worker_id')
    condition = request.args.get('condition')
    lock.acquire()
    if worker_id in worker_model:
        learner_env = worker_model[worker_id]
        DLogger.logger().debug("model used for " + worker_id)
    else:
        DLogger.logger().debug("Learner model loaded from path {}".format(learner_path))
        learner_model = load_model(learner_path, compile=False)
        learner_env = LearnverEnv(learner_model, 5, 1, 5)
        learner_env.reset()
        worker_model[worker_id] = learner_env
        worker_data[worker_id] = []
        worker_trial[worker_id] = 0
        worker_adv_action[worker_id] = 0
        DLogger.logger().debug("model created for " + worker_id)
    lock.release()

    investment = request.args.get('investment')
    investment = json.loads(investment)
    worker_trial[worker_id] += 1

    adv_state, _, done, learner_info = learner_env.step_adv(worker_adv_action[worker_id], np.array([investment]))

    if (condition == '"adv"'):
        adv_action = np.argmax(adv_model(adv_state), axis=1)
    if (condition == '"rnd"'):
        adv_action = np.random.randint(0, 5, [1])

    worker_adv_action[worker_id] = adv_action

    repay = 3 * investment * (adv_action / (4))
    worker_data[worker_id].append({'investment': investment,
                                   'adv action':adv_action,
                                   'repay': repay,
                                   'condition': condition,
                                   'git': str(get_git())
                                   })

    if worker_trial[worker_id] == 10:
        DLogger.logger().debug("Task ended")
        ensure_dir(output_path + worker_id + '/data/')
        pd.DataFrame(worker_data[worker_id]).to_csv(output_path + worker_id + '/data/output.csv')

    DLogger.logger().debug("investment: " + str(investment) + " adv action: " + str(adv_action) + " repay: " + str(repay))
    return jsonify(repay[0])

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    with LogFile(output_path, 'run.log'):
        ensure_dir(output_path)

    app.run()
