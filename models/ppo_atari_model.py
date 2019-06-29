import chainer
import chainer.links as L
import chainer.functions as F

import numpy as np

from researchutils.arrays import one_hot

if __name__ == '__main__':
    from ppo_model import PPOModel
else:
    from .ppo_model import PPOModel


class PPOAtariModel(PPOModel):
    def __init__(self, action_num):
        super(PPOAtariModel, self).__init__()
        with self.init_scope():
            self._conv1 = L.Convolution2D(
                in_channels=4, out_channels=16, ksize=8, stride=4)
            self._conv2 = L.Convolution2D(
                in_channels=16, out_channels=32, ksize=4, stride=2)
            self._linear1 = L.Linear(in_size=None, out_size=256)
            self._linear2 = L.Linear(in_size=256, out_size=action_num)
            self._linear3 = L.Linear(in_size=256, out_size=1)
        self._action_num = action_num

    def __call__(self, s):
        pi, _ = self._pi_and_value(s)
        actions = self._choose_action(pi)
        return actions

    def value(self, s):
        _, v = self._pi_and_value(s)
        return v

    def compute_log_likelihood(self, s, a):
        pi, _ = self._pi_and_value(s)
        log_pi = F.log(pi)
        one_hot_action = self._to_one_hot_action(
            a, shape=(s.shape[0], self._action_num))
        device = s.get_device()
        one_hot_action = chainer.Variable(device.send(one_hot_action))
        return F.sum(log_pi * one_hot_action, axis=1)

    def compute_entropy(self, s):
        pi, _ = self._pi_and_value(s)
        log_pi = F.log(pi)
        entropy = pi * log_pi
        return F.sum(entropy, axis=1)

    def _pi_and_value(self, s):
        h = self._conv1(s)
        h = F.relu(h)
        h = self._conv2(h)
        h = F.relu(h)
        h = self._linear1(h)
        h = F.relu(h)

        pi = self._linear2(h)
        pi = F.softmax(pi, axis=1)

        value = self._linear3(h)

        return pi, value

    def _to_one_hot_action(self, a, shape):
        return one_hot(a, shape=shape)

    def _choose_action(self, pi):
        xp = chainer.backend.get_array_module(pi)
        action = [[xp.random.choice(a=self._action_num, p=p)] for p in pi]
        print('action: ', action)
        return xp.asarray(action)


if __name__ == "__main__":
    policy = PPOAtariPolicy(action_num=4)

    pi = np.asarray([[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0]])
    action = policy._choose_action(pi=pi)
    assert len(action) == len(pi)
    assert action.shape == (2, 1)
