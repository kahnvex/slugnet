import numpy as np

from slugnet.activation import Noop


class Layer(object):
    pass


class Dense(Layer):
    def __init__(self, ind, outd, activation=Noop()):
        self.shape = ind, outd
        self.w = np.random.normal(0, 0.1, self.shape)
        self.activation = activation

    def call(self, X):
        return self.activation.call(np.dot(X, self.w))

    def backprop(self, inp, d_nxt, hidden_output=None):
        """
        Computes the derivative for the next layer
        """
        if hidden_output is not None:
            d_nxt *= self.activation.derivative(hidden_output)

        self.w += np.dot(inp.T, d_nxt)
        d_prev = np.dot(d_nxt, self.w.T)

        return d_prev
