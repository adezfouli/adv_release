import os
from multiprocessing.pool import Pool

from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.models import load_model

from data_reader import DataReader
from onto.learner_env import LearnverEnv

if __name__ == '__main__':
    learner_path = '../models/archive/learner/gonogo/learner_go_nogo_cells_5/model-21100.h5'
    model = load_model(learner_path)
    import numpy as np

    nogo = np.random.randint(0, 350, 10)
    total_rewards = 0
    for s in range(200):
        model = load_model(learner_path)
        le = LearnverEnv(model, 2, 2)
        le.step_adv(0)
        nogo = np.random.randint(0, 350, 10)
        # print(s)
        total_local_reward = 0
        learner_rnn_state = np.zeros((1, 5), dtype=np.float32)
        last_reward = 0
        for i in range(350):
            if i in nogo:
                state = np.array([0, 1], dtype=np.float32)
            else:
                state = np.array([1, 0], dtype=np.float32)
            learner_input = Concatenate(axis=2)([np.zeros((1, 1, 1), dtype=np.float32),
                                                 np.zeros((1, 1, 2), dtype=np.float32),
                                                 state[np.newaxis, np.newaxis]]
                                                )

            learner_rnn_state, pred_pol = model([learner_input, learner_rnn_state])
            learner_rnn_state = learner_rnn_state[0]

            if np.random.random() < pred_pol[0, 0, 0]:
                last_action = 0
                vec_action = np.array([1, 0], np.float32)
            else:
                last_action = 1
                vec_action = np.array([0, 1], np.float32)

            if last_action == 0 and state[0] == 1:
                reward = np.array([1], dtype=np.float32)
                adv_reward = 0

            elif last_action == 1 and state[1] == 1:
                reward = np.array([1], dtype=np.float32)
                adv_reward = 0

            else:
                reward = np.array([0], dtype=np.float32)
                adv_reward = 1

            learner_input = Concatenate(axis=2)([reward[np.newaxis, np.newaxis],
                                                 vec_action[np.newaxis, np.newaxis],
                                                 np.zeros((1, 1, 2), np.float32)]
                                                )

            learner_rnn_state, _ = model([learner_input, learner_rnn_state])
            learner_rnn_state = learner_rnn_state[0]

            total_rewards += adv_reward
            total_local_reward += adv_reward
        print(total_local_reward)
    print(total_rewards / 200)
