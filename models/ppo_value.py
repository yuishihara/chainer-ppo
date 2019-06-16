from chainer import Chain
import chainer.links as L
import chainer.functions as F


class PPOValue(Chain):
    def __init__(self):
        super(PPOValue, self).__init__()
        with self.init_scope():
            self.linear1 = L.Linear(in_size=None, out_size=64)
            self.linear2 = L.Linear(in_size=64, out_size=64)
            self.linear3 = L.Linear(in_size=64, out_size=1)

    def __call__(self, s):
        h = self.linear1(s)
        h = F.tanh(h)
        h = self.linear2(h)
        h = F.tanh(h)
        value = self.linear3(h)
        value = F.squeeze(value)
        return value
