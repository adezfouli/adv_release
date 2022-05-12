import numpy as np
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.models import load_model

from nc.learner_env import LearnverEnv
from util.logger import LogFile, DLogger
import tensorflow as tf


class DQNBanditEnv:

    def __init__(self, learner_model_path, adv_model_path, output_path):
        # fix_seeds()
        np.set_printoptions(precision=5)

        self.events = []
        with LogFile(output_path, 'run.log'):
            DLogger.logger().debug("Learner model loaded from path {}".format(learner_model_path))
            learner_model = load_model(learner_model_path)
            self.le = LearnverEnv(learner_model, 2)
            DLogger.logger().debug("Adv model loaded from path {}".format(adv_model_path))
            self.adv_model = load_model(adv_model_path, compile=False)
        self.reset()

    def step(self, action):
        reward = self.le.step_vec(action, self.reward_vec)
        rnn_action = self.le.get_action()
        action_values = self.adv_model.predict(self.le.get_adv_state(action, reward)[np.newaxis,])
        adv_action = tf.squeeze(argmax(action_values))
        self.reward_vec = self.le.adv_action_to_reward(adv_action.numpy())
        return self.reward_vec, adv_action, self.le.adv_reward(action, reward), rnn_action

    def reset(self):
        self.le.reset()
        self.reward_vec = np.array([0, 0], np.float32)


if __name__ == '__main__':
    DQNBanditEnv('../nongit/archive/learner/nc/learner_nc/learner_nc_cells_5/model-8800.h5',
              '../nongit/results/RL_nc_dqn_200/model-73000.h5', '../nongit/results/temp/'
              )
