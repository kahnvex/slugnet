import numpy as np


class L2Regularization(object):
    """The Common L2 regularization penalty."""
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def call(self, w):
        return np.sum(self.alpha * np.square(w))

    def grad(self, w, *args, **kwargs):
        return self.alpha * w
