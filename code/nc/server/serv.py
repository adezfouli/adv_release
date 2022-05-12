import h5py
import threading
import json
from flask import Flask, jsonify
from flask import request
from nc.adv_learner import AdvLearner
from util import DLogger
import numpy as np
import pandas as pd
from datetime import datetime
import time

from util.helper import ensure_dir

lock = threading.Lock()
worker_model = {}
worker_data = {}
app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

learner_path = '../models/archive/learner/nc/nc_human_learner_single/cells_10/model-1100.h5'
adv_path = '../models/archive/RL/nc/RL_human_dqn_vec/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/model-600000.h5'
output_path = '../nongit/results/subject_data/'

learner_path = h5py.File(learner_path, 'r')
adv_path = h5py.File(adv_path, 'r')

@app.route('/choice/')
def hello():
    worker_id = request.args.get('worker_id')
    lock.acquire()
    if worker_id in worker_model:
        model = worker_model[worker_id]
        DLogger.logger().debug("model used for " + worker_id)
    else:
        model = AdvLearner(learner_path, adv_path, output_path + worker_id + '/', None)
        model.reset()
        worker_model[worker_id] = model
        worker_data[worker_id] = []
        DLogger.logger().debug("model created for " + worker_id)
    lock.release()

    bias_choice = request.args.get('is_bias_choice')
    bias_choice = json.loads(bias_choice)

    # bias_reward = request.args.get('bias_reward')
    # bias_reward = json.loads(bias_reward)

    # unbias_reward = request.args.get('unbias_reward')
    # unbias_reward = json.loads(unbias_reward)

    # DLogger.logger().debug(bias_reward)
    # DLogger.logger().debug(unbias_reward)
    DLogger.logger().debug(bias_choice)

    if len(bias_choice) == 0:
        reward_vec, adv_action, adv_reward, rnn_action, real_action = model.step(np.array([[0, 0]], np.float32))
        action = np.array([[0, 0]])
    else:
        choice = bias_choice[-1] == 'True'
        if choice:
            action = np.array([[1, 0]])
        else:
            action = np.array([[0, 1]])

        reward_vec, adv_action, adv_reward, rnn_action, real_action = model.step(np.array(action, np.float32))

    worker_data[worker_id].append({'reward_vec': reward_vec.numpy(), 'adv_action':adv_action.numpy(),
                                   'adv_reward': adv_reward, 'rnn_action':rnn_action,
                                   'action': action,
                                   'data-time': datetime.now(),
                                   'time': time.time(),
                                   })

    rew1, rew2 = reward_vec[0, 0].numpy(), reward_vec[0, 1].numpy()
    DLogger.logger().debug("reward: " + str(rew1) + " " + str(rew2))
    return jsonify([str(rew1), str(rew2)])


@app.route('/end_task/')
def end_task():
    DLogger.logger().debug("Task ended")
    worker_id = request.args.get('worker_id')
    bias_choice = request.args.get('is_bias_choice')
    bias_choice = json.loads(bias_choice)
    choice = bias_choice[-1] == 'True'
    if choice:
        action = np.array([1, 0])
    else:
        action = np.array([0, 1])
    worker_data[worker_id].append({'reward_vec': None, 'adv_action': None,
                                   'adv_reward': None, 'rnn_action': None,
                                   'action': action,
                                   'data-time': datetime.now(),
                                   'time': time.time(),
                                   })
    ensure_dir(output_path + worker_id + '/data/')
    pd.DataFrame(worker_data[worker_id]).to_csv(output_path + worker_id + '/data/output.csv')
    return jsonify("success")


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
    ensure_dir(output_path)
    app.run()
