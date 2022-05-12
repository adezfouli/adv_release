import tensorflow as tf
from util.helper import multinomial_rvs
from util.logger import LogFile, DLogger
import numpy as np
import pandas as pd

"""
This class takes an joint adversary/learning and simulates them in a gonogo task.
"""
def sim_rmrtt(learner_env, adv_model, output_path, stochastic=False):
    with LogFile(output_path, 'run.log'):
        adv_reward_list = []
        psudo_rew = []
        reward_vecs = []
        rnn_actions = []
        rnn_actions_cont = []
        adv_actions = []
        adv_rewards = []
        adv_states = []
        learner_states = []
        seudo_rew = []

        learner_env.reset()
        adv_state, _, _, learner_info = learner_env.step_adv(0)
        #
        # cc = []
        # for a in range(learner_env.n_batches):
        #     cc.append(np.random.choice(np.arange(0, 350), 35, replace=False))
        #
        # cc = np.vstack(cc)
        #
        for t in range(10):

            # adv_action = np.argmax(adv_model(adv_state), axis=1)
            # if (np.random.rand() < 0.05):
            #     adv_action = np.random.randint(0, 2, adv_action.shape[0])
            # r_size = int(adv_action.shape[0] * 0.05)
            # adv_action[np.random.choice(np.arange(0, adv_action.shape[0]), r_size)] = np.random.randint(0, 2, r_size)

            if stochastic:
                adv_action = np.argmax(multinomial_rvs(1, tf.nn.softmax(adv_model(adv_state)).numpy()), axis=1)
            else:
                adv_action = np.argmax(adv_model(adv_state), axis=1)

            #
            # adv_action = np.sum(t == cc, axis=1)
            #
            cur_learner_action = learner_info['learner_action'][np.newaxis]
            cur_learner_action_cont = learner_info['learner_action_cont'][np.newaxis]

            # ************************
            cur_learner_action_cont[cur_learner_action_cont == 0] = 0.001
            adv_action = np.ceil((4 * cur_learner_action_cont - 20) / (6 * cur_learner_action_cont) * 10000)[0]
            adv_action[adv_action <= 0] = 0
            # ************************

            adv_next_state, adv_reward, done, learner_info = learner_env.step_adv(adv_action)

            reward_vecs.append(learner_info['learner_reward'][np.newaxis])
            rnn_actions.append(cur_learner_action)
            seudo_rew.append(learner_info['seudo_rew'][np.newaxis])
            adv_actions.append(adv_action[np.newaxis])
            adv_rewards.append(adv_reward[np.newaxis])
            adv_states.append(adv_state[np.newaxis])
            learner_states.append(learner_info['state'][np.newaxis])
            rnn_actions_cont.append(cur_learner_action_cont)
            adv_state = adv_next_state

        reward_vecs = np.concatenate(reward_vecs, axis=0)
        rnn_actions = np.concatenate(rnn_actions, axis=0)
        rnn_actions_cont = np.concatenate(rnn_actions_cont, axis=0)
        adv_actions = np.concatenate(adv_actions, axis=0)
        adv_rewards = np.concatenate(adv_rewards, axis=0)
        adv_states = np.concatenate(adv_states, axis=0)
        learner_states = np.concatenate(learner_states, axis=0)
        seudo_rew = np.concatenate(seudo_rew, axis=0)

        for j in range(reward_vecs.shape[1]):
            events = {
             'learner reward': np.squeeze(reward_vecs[:, j]),
             'learner action': [str(x) for x in rnn_actions[:, j]],
             'learner action cont': [str(x) for x in rnn_actions_cont[:, j]],
             'adv action': adv_actions[:, j],
             'adv reward': adv_rewards[:, j],
             'psudo reward': seudo_rew[:, j, 0],
             'real model action': None,
             'adv state': [str(x) for x in adv_states[:, j]],
             'learner state': [str(x) for x in learner_states[:, j]]
             }
            pd.DataFrame(events).to_csv(output_path + "events_" + str(j) + ".csv")
            adv_reward_list.append(adv_rewards[:, j].sum())
            psudo_rew.append(seudo_rew[:, j].sum())
        pd.DataFrame(adv_reward_list).to_csv(output_path + "adv_reward_" + ".csv")
        pd.DataFrame(psudo_rew).to_csv(output_path + "psudo_reward_" + ".csv")
