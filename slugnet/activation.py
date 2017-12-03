import numpy as np


class Activation(object):
    pass


class Sigmoid(Activation):
    def call(self, x):
        return 1. / (1 + np.exp(-x))

    def derivative(self, x):
        return self.call(x) * (1 - self.call(x))


class Noop(Activation):
    def call(self, x):
        return x

    def derivative(self, x):
        return x
