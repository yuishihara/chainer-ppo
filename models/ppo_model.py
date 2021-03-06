from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class PPOModel(Chain):
    def __init__(self):
        super(PPOModel, self).__init__()
        pass

    def __call__(self, s):
        raise NotImplementedError()

    def value(self, s):
        raise NotImplementedError() 

    def compute_log_likelihood(self, s, a):
        raise NotImplementedError()

    def compute_entropy(self, s):
        raise NotImplementedError()
