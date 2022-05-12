# from https://github.com/keon/deep-q-learning/blob/master/ddqn.py


import random
import numpy as np
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.ops.init_ops_v2 import VarianceScaling

from arrq import ArrQ
from util import DLogger
from util.logger import LogFile


class DQNAgent:
    def __init__(self, state_size, action_size,
                 buffer_size,
                 epsilon,
                 lr
                 ):
        q_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        self.memory = {'state': ArrQ(q_size, [state_size]),
                       'action': ArrQ(q_size, [action_size]),
                       'reward': ArrQ(q_size, [1]),
                       'next_state': ArrQ(q_size, [state_size]),
                       'done': ArrQ(q_size, [1])
                       }
        self.gamma = 1  # discount rate
        self.epsilon = epsilon
        self.learning_rate = lr
        # self.epsilon_min = 0.1
        # self.epsilon_decay = 0.99
        # self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.actions = np.eye(self.action_size)
        self.update_target_model()
        self.next_max = 0

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """
    def _huber_loss(self, y_true, y_pred, clip_delta=0.1):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error) * tf.cast(cond, tf.float32)

        return K.mean(squared_loss)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_size,), activation='relu', kernel_initializer=VarianceScaling(scale=2)))
        model.add(Dense(128, activation='relu', kernel_initializer=VarianceScaling(scale=2)))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer=VarianceScaling(scale=2)))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(np.array(reward))
        self.memory['next_state'].append(next_state)
        self.memory['done'].append(np.array(done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state), axis=1)[0]

    def replay(self, batch_size):
        batch_indx = random.sample(range(0, len(self.memory['state'])), batch_size)
        state = self.memory['state'][batch_indx]
        action = self.memory['action'][batch_indx]
        reward = self.memory['reward'][batch_indx]
        next_state = self.memory['next_state'][batch_indx]
        done = self.memory['done'][batch_indx]

        next_values = np.max(self.target_model.predict(next_state), axis=1)
        next_values = next_values * (1 - done)[:, 0]
        target = reward[:, 0] + self.gamma * next_values

        target_value = self.model.predict(state)
        target_value[np.arange(0, target_value.shape[0]), np.argmax(action, axis=1)] = target
        self.next_max = target_value.mean()

        self.model.fit(state, target_value, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, env, batch_size=50, total_episodes=200000, output_path=None):

        state_size = self.state_size
        cur_iter = 0

        for e in range(total_episodes):
            if e % 1000 == 0:
                save_path = output_path + 'model-' + str(e) + '.h5'
                DLogger.logger().debug('Trained model saved to: ' + save_path)
                self.model.save(save_path)

            env.reset()
            state, _, _, _ = env.step_adv(0)
            state = np.reshape(state, [1, state_size])
            total_r = 0
            while True:
                action = self.act(state)
                next_state, reward, done, _ = env.step_adv(action)
                next_state = np.reshape(next_state, [1, state_size])
                self.remember(state, self.actions[action][np.newaxis,], [[reward]], next_state, [[done]])
                state = next_state
                total_r += reward
                if done:
                    DLogger.logger().debug("episode: {}/{}, score: {}, e: {:.2}, reward: {}, max: {}"
                          .format(e, total_episodes, 0, self.epsilon, total_r, self.next_max))
                    # self.epsilon = max(1 - e / 50000, 0.02)
                    break

                if cur_iter % 10000 == 0:
                    self.update_target_model()
                    DLogger.logger().debug("target network updated")

                if cur_iter % 4 == 0 and len(self.memory['state']) > batch_size:
                    self.replay(batch_size)

                cur_iter += 1
