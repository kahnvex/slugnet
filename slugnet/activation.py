import numpy as np


class Activation(object):
    pass


class Sigmoid(Activation):
    def call(self, x):
        self.last_out = 1. / (1. + np.exp(-x))

        return self.last_out

    def derivative(self, x=None):
        z = self.call(x) if x else self.last_out

        return z * (1 - z)


class Noop(Activation):
    def call(self, x):
        return x

    def derivative(self, x=1.):
        return x


class ReLU(Activation):

    def __init__(self):
        super(ReLU, self).__init__()

    def call(self, x):
        self.last_forward = x

        return np.maximum(0.0, x)

    def derivative(self, x=None):
        last_forward = x if x else self.last_forward
        res = np.zeros(last_forward.shape, dtype='float32')
        res[last_forward > 0] = 1.

        return res


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()

    def call(self, x):
        assert np.ndim(x) == 2
        self.last_forward = x
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return s

    def derivative(self, x=None):
        last_forward = x if x else self.last_forward
        return np.ones(last_forward.shape, dtype='float32')
