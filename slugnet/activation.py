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
