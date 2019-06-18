from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class PPOPolicy(Chain):
    def __init__(self, action_num):
        super(PPOPolicy, self).__init__()
        with self.init_scope():
            self.linear1 = L.Linear(in_size=None, out_size=64)
            self.linear2 = L.Linear(in_size=64, out_size=64)
            self.linear3 = L.Linear(in_size=64, out_size=action_num * 2)

    def __call__(self, s):
        mu, ln_var = self._mean_and_variance(s)
        action = F.gaussian(mu, ln_var)
        return action

    def compute_log_likelihood(self, s, a):
        mu, ln_var = self._mean_and_variance(s)
        log_likelihood = -F.gaussian_nll(a, mu, ln_var, reduce='no')
        log_likelihood = F.sum(log_likelihood, axis=1)
        return log_likelihood

    def compute_entropy(self, s):
        # differential entropy
        # k/2*(1 + log(2*pi)) + 1/2 *ln_var
        mu, ln_var = self._mean_and_variance(s)
        return 0.5 * mu.shape[1] * (1 + np.log(2.0 * np.pi)) + 0.5 * F.sum(ln_var, axis=1)

    def _mean_and_variance(self, x):
        h = self.linear1(x)
        h = F.tanh(h)
        h = self.linear2(h)
        h = F.tanh(h)
        h = self.linear3(h)

        return F.split_axis(h, 2, axis=-1)
