from tensorflow.python.keras.saving import load_model

from util.helper import multinomial_rvs
from util.logger import LogFile, DLogger
import numpy as np
from scipy.stats import binom

"""
This class takes a learner and makes and environment out of it. That is, the learner receives actions of the adversary
uses that action to get reward for the next step, makes an step, and then returns its RNN state and other information
such as number of rewards etc as the state to the adversary.
"""
class LearnverEnv:

    def __init__(self, model, n_actions, n_states, n_batches,
                 LOF=None,
                 LOF_weight = None):
        self.model = model
        self.n_batches = n_batches
        n_cells = model.get_layer('GRU').units
        self.n_actions = n_actions
        self.observation_space = type("observation_space", (object,), {"shape": (3,)})
        self.action_space = type("action_space", (object,), {"n": self.n_actions})
        self.init_state = np.zeros((n_batches, n_cells), dtype=np.float32)
        self.n_states = n_states
        self.null_state = np.zeros((n_batches, 2,), dtype=np.float32)
        self.null_action = np.zeros((n_batches,2,), dtype=np.float32)
        self.null_reward = np.zeros((n_batches, 1,), dtype=np.float32)
        self.LOF = LOF
        self.LOF_weight = LOF_weight
        self.reset()

    @staticmethod
    def adv_reward(state, learner_action):
        return (state * (1 - learner_action)).sum(axis=1)

    @staticmethod
    def adv_done(curr_trial):
        return curr_trial >= 350

    def reset(self):
        self.norm_factor = 100
        self.curr_trial = np.zeros((self.n_batches,))
        self.learner_rnn_state = self.init_state
        self.total_action = np.zeros((self.n_batches, self.n_actions,), dtype=np.float32)
        self.total_reward = np.zeros((self.n_batches, 1,), dtype=np.float32)
        self.total_state = np.zeros((self.n_batches, self.n_states,), dtype=np.float32)
        self.pred_pol = None
        self.last_action = np.zeros((self.n_batches, self.n_actions,), dtype=np.float32)
        self.last_reward = np.zeros((self.n_batches, 1), dtype=np.float32)
        self.reseted = True
        self.Q =[]

    def adv_action_to_state(self, adv_action):
        learner_state = np.zeros((adv_action.shape[0], 2))
        learner_state[np.arange(adv_action.shape[0]), adv_action] = 1
        return self.constrained_state(learner_state)

    def constrained_state(self, learner_state):
        ind = self.total_state[:, 1] >= 35
        if ind.sum() > 0:
            learner_state[ind] = np.array([1, 0], np.float32)

        ind = self.total_state[:, 1] + 350 - self.curr_trial < 36

        if ind.sum() > 0:
            learner_state[ind] = np.array([0, 1], np.float32)

        return learner_state

    def step_action_reward(self, action, reward):
        self.step(self.null_state, action, reward)

    def step_state(self, adv_action):
        state = self.adv_action_to_state(adv_action)
        self.step(state, self.null_action, self.null_reward)
        self.curr_trial += 1
        return state

    def step_adv(self, adv_action):

        state = None
        learner_action = None
        learner_reward = None
        if self.reseted:
            adv_reward = 0
            seudo_rew = 0
            adv_done = False
            adv_state = self.get_adv_state()
        else:
            state = self.step_state(adv_action)
            learner_action, pols = self.get_action()
            learner_reward = np.array([(learner_action * state).sum(1)], np.float32).T
            self.step_action_reward(learner_action, learner_reward)
            adv_reward = self.adv_reward(state, learner_action)
            seudo_rew = adv_reward.copy()
            adv_done = self.adv_done(self.curr_trial)
            adv_state = self.get_adv_state()
            self.Q.insert(0, adv_action.copy())
            if len(self.Q) > 20:
                self.Q.pop()

        # adv_reward = adv_reward * 0
        if len(self.Q) == 20:
            adv_reward += (self.LOF_weight * binom.pmf(np.array(self.Q).sum(axis=0), 20, 0.1))

        self.reseted = False
        return adv_state, \
               adv_reward, \
               adv_done, {'state': state, 'learner_action': learner_action, 'learner_reward': learner_reward,
                          'seudo_rew': seudo_rew
                          }

    def get_action(self):
        return multinomial_rvs(1, self.pred_pol), self.pred_pol

    def get_adv_state(self):
        if self.pred_pol is None:
            pol = np.ones((self.n_batches, 2)) / 2.
        else:
            pol = self.pred_pol

        return np.concatenate([
                                        self.total_state / self.norm_factor,
                                        self.curr_trial[:, np.newaxis] / self.norm_factor,
                                      ], axis=1)

    def step(self, cur_state, last_action, last_reward):
        self.total_reward += last_reward
        self.total_action += last_action
        self.total_state += cur_state

        learner_input = np.concatenate([last_reward,
                                             last_action,
                                             cur_state], axis=1
                                            )

        self.learner_rnn_state, self.pred_pol = self.model.predict([learner_input[:, np.newaxis], self.learner_rnn_state])
        self.pred_pol = self.pred_pol[:, -1]
        self.learner_rnn_state = self.learner_rnn_state[:, -1, :]


if __name__ == '__main__':

    # fix_seeds()
    np.set_printoptions(precision=3)

    output_path = '../nongit/results/sim/'
    model_path = '../nongit/archive/learner/gonogo/learner_go_nogo_cells_5/model-19900.h5'

    with LogFile(output_path, 'run.log'):
        model = load_model(model_path)
        le = LearnverEnv(model, 2, 2)

        learner_action = np.zeros((2,), dtype=np.float32)
        reward = np.zeros((1,), dtype=np.float32)
        cur_adv_action = 0
        adv_done = False
        i = 0
        le.step_adv(0)
        while not adv_done:
            adv_state, adv_reward, adv_done, _ = le.step_adv(1)
            DLogger.logger().debug("obs: {} reward: {} done: {}".format(adv_state, adv_reward, adv_done))
