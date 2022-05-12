import numpy as np

from util.helper import one_hot


class QRL_ENV():

    def __init__(self, alpha, beta, persv):
        self.alpha = alpha
        self.beta = beta
        self.persv = persv

        self.Q = None
        self.reset()

    def reset(self):
        self.Q = np.array([0.0, 0.0])

    def step(self, action, reward):
        if action[0] == 1:
            a = 0
        elif action[1] == 1:
            a = 1
        else:
            a = None

        if a is not None:
            self.Q[a] = (1 - self.alpha) * self.Q[a] + self.alpha * reward

        pol0 = np.exp(self.beta * self.Q[0] + (a == 0) * self.persv) / \
               (np.exp(self.beta * self.Q[0] + (a == 0) * self.persv) +
                np.exp((a == 1) * self.persv + self.beta * self.Q[1]))

        if np.random.uniform(0, 1) < pol0:
            a = 0
        else:
            a = 1

        return one_hot(np.array([a]), 2)[0]

class QRL_nc_ENV():

    def __init__(self, alpha, beta, epsilon, persv):
        self.alpha = alpha
        self.beta = beta
        self.persv = persv
        self.epsilon = epsilon

        self.Q = None
        self.a_chosen = None
        self.trial = None
        self.reset()

    def reset(self):
        self.Q = np.array([0.0, 0.0])
        self.a_chosen = [False, False]
        self.trial = 0

    def step(self, action, reward):
        if action.shape[0] != 1 or reward.shape[0] != 1:
            raise Exception('vector sim not supported')
        action = action[0]
        reward = reward[0]
        if action[0] == 1:
            a = 0
        elif action[1] == 1:
            a = 1
        else:
            a = None

        if not a is None:
            if not self.a_chosen[a]:
                self.Q[a] = reward
            else:
                self.Q[a] = (1 - self.alpha) * self.Q[a] + self.alpha * reward
            self.a_chosen[a] = True

        if self.trial > 1:
            pol0 = np.exp(self.beta * self.Q[0] + (a == 0) * self.persv) / \
                   (np.exp(self.beta * self.Q[0] + (a == 0) * self.persv) + np.exp((a == 1) * self.persv + self.beta * self.Q[1]))
            pol0 = (1 - 2*self.epsilon) * pol0 + self.epsilon
        else:
            pol0 = 0.5

        if np.random.uniform(0, 1) < pol0:
            a = 0
        else:
            a = 1

        self.trial += 1

        return one_hot(np.array([a]), 2)[0]


if __name__ == '__main__':
    ql = QRL_ENV(0.0, 4, 0)
    a = ql.reset()

    k = 0
    for i in range(200):
        r = 0
        if a == 1:
            if np.random.uniform(0, 1) < 0.5:
                r = 1

        if a == 0:
            if np.random.uniform(0, 1) < 0.1:
                r = 1

        a = ql.step(r)
        k += a

        print(a)

    print(k / 200)