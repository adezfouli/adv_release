import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from util import DLogger
from util.helper import fix_seeds
import tensorflow.keras.backend as K



class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


"""
This class is used to train an A2CAgent as an adversary.
"""
class A2CAgent:
    def __init__(self, num_actions, state_size, layers, units, eps, model=None, lr=0.0001):
        # hyperparameters for loss terms
        self.params = {'value': 0.5, 'init_entropy': eps, 'gamma': 1, 'entropy': eps}

        if model is None:
            DLogger.logger().debug("pol grad with layers: " + str(layers))
            inputs = tf.keras.Input(shape=(state_size,), name='state')
            x = tf.convert_to_tensor(inputs, dtype=tf.float32)

            prev_out = x
            for l in range(layers):
                prev_out = kl.Dense(units, activation='relu', name= "value_" + str(l))(prev_out)
            value_layer = kl.Dense(1, name='value')
            value = value_layer(prev_out)

            # logits are unnormalized log probabilities
            prev_out = x
            for l in range(layers):
                prev_out = kl.Dense(units, activation='relu', name= "pol_" + str(l))(prev_out)
            logits_layer = kl.Dense(num_actions, name='policy_logits')
            logits = logits_layer(prev_out)

            self.model = tf.keras.Model(inputs=x, outputs=[logits, value], name='A2CRL')
        else:
            self.model = model

        self.model.summary(print_fn=DLogger.logger().debug)
        self.model.compile(
            optimizer=ko.RMSprop(lr=lr),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
        self.dist = ProbabilityDistribution()

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.model(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        return ep_reward

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(tf.keras.backend.softmax(logits), logits, from_logits=True)
        entropy_loss = K.mean(entropy_loss)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss

    def train(self, env, batch_sz=200, updates=200000, output_path=None):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        env.reset()
        next_obs, _, _, _ = env.step_adv(0)
        for update in range(updates):
            # if update % 10 == 0:
            #     self.params['entropy'] = 100 * 10 ** -int(update / 10)
            #     self.model.compile(
            #         optimizer=ko.RMSprop(lr=0.0007),
            #         # define separate losses for policy logits and value estimate
            #         loss=[self._logits_loss, self._value_loss]
            #     )
            #     DLogger.logger().debug("changing epsilon to " + str(self.params['entropy']) + " and compiling model")

            if output_path is not None:
                if update % 5000 == 0:
                    save_path = output_path + 'model-' + str(update) +'.h5'
                    DLogger.logger().debug('Trained model saved to: ' + save_path)
                    self.model.save(save_path)

            cur_rews = 0
            step = 0
            # for step in range(batch_sz):
            while True:
                observations[step] = next_obs
                # actions[step], values[step] = self.action_value(next_obs[None, :])
                values[step] = 0.1
                actions[step] = 1
                next_obs, rewards[step], dones[step], _ = env.step_adv(0)

                cur_rews += rewards[step]
                if dones[step]:
                    env.reset()
                    next_obs, _, _, _ = env.step_adv(0)
                    break
                step += 1

            step += 1
            # _, next_value = self.model.action_value(next_obs[None, :])
            next_value = [0.0]
            returns, advs = self._returns_advantages(rewards[:step], dones[:step], values[:step], next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:step, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations[:step], [acts_and_advs, returns])

            DLogger.logger().debug("iter: {} loss: {} reward: {} entropy: {}".
                                   format(update, losses, cur_rews, self.params['entropy']))
            self.params['entropy'] = self.params['init_entropy'] * (10 ** (-(update / 10000.)))
            env.model


        if output_path is not None:
            save_path = output_path + 'model-final.h5'
            DLogger.logger().debug('Trained model saved to: ' + save_path)
            self.model.save(save_path)


    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages


if __name__ == '__main__':
    import gym

    fix_seeds()

    env = gym.make('CartPole-v0')
    env.seed(1)

    obs = env.reset()

    agent = A2CAgent(env.action_space.n ,env.observation_space.shape[0])

    rewards_history = agent.train(env)
    agent.model.to_json()
    print("Finished training, testing...")
    print("%d out of 200" % agent.test(env)) # 200 out of 200
