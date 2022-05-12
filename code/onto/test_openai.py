import gym

from baselines import deepq, logger
from onto.learner_env import LearnverEnv
from util import DLogger
import numpy as np

from util.helper import get_git
from util.logger import LogFile
from tensorflow_core.python.keras.saving.save import load_model


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


def main():
    learner_model_path = '../models/archive/learner/gonogo/onto/gonogo_learner_single/model-19100.h5'
    output_path = '../nongit/temp/dez004/RL_gonogo_dqn/RL_nc_dqn_buf_/'
    logger.configure(dir=output_path)

    logger.debug('config: ')
    np.set_printoptions(precision=3)

    logger.log("version control: " + str(get_git()))
    logger.log("Learner model loaded from path {}".format(learner_model_path))
    model = load_model(learner_model_path)
    logger.log("Learner model:")
    model.summary(print_fn=logger.log)
    le = LearnverEnv(model, 2, 2)
    le.reset()
    deepq.learn(
        Env(le_env=le),
        network='mlp',
        lr=1e-3,
        total_timesteps=10000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        checkpoint_path=output_path + '/model/',
    )

if __name__ == '__main__':
    main()