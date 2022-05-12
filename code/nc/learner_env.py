from util.helper import multinomial_rvs
from util.logger import LogFile, DLogger
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.saving import load_model


"""
This class takes a learner and makes and environment out of it. That is, the learner receives actions of the adversary
uses that action to get reward for the next step, makes an step, and then returns its RNN state and other information
such as number of rewards etc as the state to the adversary.
"""
class LearnverEnv:

    def __init__(self, model, n_actions, n_batches):
        self.n_batches = n_batches
        self.model = model
        n_cells = model.get_layer('GRU').units
        self.n_actions = n_actions
        self.observation_space = type("observation_space", (object,), {"shape": (3 * self.n_actions + n_cells + 1,)})
        self.action_space = type("action_space", (object,), {"n": self.n_actions})
        self.init_state = np.zeros((self.n_batches, n_cells), dtype=np.float32)
        self.reset()

    @staticmethod
    def adv_reward(action, reward):
        return action[:, 0]

    @staticmethod
    def adv_done(curr_trial):
        return curr_trial >= 100

    def reset(self):
        self.norm_factor = 100
        self.curr_trial = np.ones((self.n_batches)) * -1
        self.learner_rnn_state = self.init_state
        self.total_action = np.zeros((self.n_batches, self.n_actions,), dtype=np.float32)
        self.total_reward = np.zeros((self.n_batches, self.n_actions,), dtype=np.float32)
        self.pred_pol = None

    def adv_action_to_reward(self, adv_action):
        learner_reward = np.zeros((self.n_batches, 2))
        learner_reward[adv_action == 0] = np.array([0, 0], np.float32)
        learner_reward[adv_action == 1] = np.array([1, 0], np.float32)
        learner_reward[adv_action == 2] = np.array([0, 1], np.float32)
        learner_reward[adv_action == 3] = np.array([1, 1], np.float32)

        return self.constrained_reward(learner_reward)

    def constrained_reward(self, learner_reward):
        learner_reward = tf.cast(self.total_reward < 25, dtype=tf.float32) * learner_reward
        learner_reward = learner_reward + tf.cast(self.total_reward + 100 - self.curr_trial[:, np.newaxis] < 26, tf.float32)
        learner_reward = tf.clip_by_value(learner_reward, 0, 1)
        return learner_reward

    def step_adv(self, adv_action):
        reward_vec = self.adv_action_to_reward(adv_action)
        action = self.get_action()
        self.step_vec(action, reward_vec)
        return self.get_adv_state(), self.adv_reward(action, reward_vec), self.adv_done(self.curr_trial), None

    def get_action(self):
        if self.pred_pol is None:
            return np.zeros((self.n_batches, 2), np.float32)
        else:
            return multinomial_rvs(1, self.pred_pol.numpy())

    def get_adv_state(self):
        return np.concatenate([
                                        self.learner_rnn_state,
                                        self.pred_pol,
                                        self.total_action / self.norm_factor,
                                        self.total_reward / self.norm_factor,
                                        self.curr_trial[:, np.newaxis] / self.norm_factor,
                                      ], axis=1)

    def step_vec(self, action, reward_vec):
        self.total_reward += reward_vec
        self.total_action += action

        reward = reward_vec.numpy()[np.arange(reward_vec.shape[0]), (action[:, 1]).astype(np.int32)]
        self.step(action, reward)
        self.curr_trial += 1
        return reward

    def step(self, action, reward):

        learner_input = np.concatenate([reward[:, np.newaxis, np.newaxis],
                                             action[:, np.newaxis]], axis=2)

        self.learner_rnn_state, self.pred_pol = self.model([learner_input, self.learner_rnn_state])
        self.pred_pol = self.pred_pol[:, -1]
        self.learner_rnn_state = self.learner_rnn_state[:, -1, :]
        return self.get_action()[0]
