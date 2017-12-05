import numpy as np


class Optimizer(object):
    def __init__(self, lr=0.001, clip=-1, decay=0., lr_min=0., lr_max=np.inf):
        self.lr = lr
        self.clip = clip
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.iterations = 0

    def update(self, params, grads):
        """Update parameters.
        Parameters
        ----------
        params : list
            A list of parameters in model.
        grads : list
            A list of gradients in model.
        """
        self.iterations += 1
        self.lr *= (1. / 1 + self.decay * self.iterations)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)

    def __str__(self):
        return self.__class__.__name__


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    """

    def update(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * npdl_clip(g, self.clip)

        super(SGD, self).update(params, grads)


def npdl_clip(grad, boundary):
    if boundary > 0:
        return np.clip(grad, -boundary, boundary)
    else:
        return grad
