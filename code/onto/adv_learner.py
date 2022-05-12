import numpy as np
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.models import load_model

from onto.learner_env import LearnverEnv
from util.logger import LogFile, DLogger
import tensorflow as tf

"""
This class includes an adversary and an a learning in the form of a RNN, and also 
a real agent which selects the actions. In each step, the model receives the action 
of the learner, return the reward vector provided by the adversary and also action of 
the RNN agent and also action of the real agent.
"""
class AdvLearner:

    def __init__(self, learner_model_path, adv_model_path, output_path, real_model):
        np.set_printoptions(precision=5)
        self.real_model = real_model
        self.events = []
        with LogFile(output_path, 'run.log'):
            if real_model is None:
                DLogger.logger().debug("Real model is not provided -- using learner for selecting actions.")
            DLogger.logger().debug("Learner model loaded from path {}".format(learner_model_path))
            learner_model = load_model(learner_model_path, compile=False)
            self.le = LearnverEnv(learner_model, 2, 2, 200)
            DLogger.logger().debug("Adv model loaded from path {}".format(adv_model_path))
            self.adv_model = load_model(adv_model_path, compile=False)
        self.reset()

    def step(self):
        logits = self.adv_model.predict(self.le.get_adv_state())
        if isinstance(logits, list) and len(logits) > 1:  #if the return includes both policies and estiamted values
            logits = logits[0]

        adv_action = argmax(logits, axis=1)
        state = self.le.step_state(adv_action.numpy())
        rnn_action, _ = self.le.get_action()
        reward = np.array([(state * rnn_action).sum(axis=1)], np.float32).T
        self.le.step_action_reward(rnn_action, reward)
        adv_reward = self.le.adv_reward(state, rnn_action)
        if self.real_model is not None:
            real_action = self.real_model.step(rnn_action, reward)
        else:
            real_action = None

        return reward, adv_action, adv_reward, rnn_action, real_action, state

    def reset(self):
        self.le.reset()
        if self.real_model:
            self.real_model.reset()
