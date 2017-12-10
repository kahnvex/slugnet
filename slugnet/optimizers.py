import numpy as np

from slugnet.initializations import _zero


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


class RMSProp(Optimizer):
    """RMSProp updates
    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.
    Parameters
    ----------
    rho : float
        Gradient moving average decay factor.
    epsilon : float
        Small value added for numerical stability.
    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}
    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    def __init__(self, rho=0.9, epsilon=1e-6, *args, **kwargs):
        super(RMSProp, self).__init__(*args, **kwargs)

        self.rho = rho
        self.epsilon = epsilon

        self.cache = None
        self.iterations = 0

    def update(self, params, grads):
        # init cache
        if self.cache is None:
            self.cache = [_zero(p.shape) for p in params]

        # update parameters
        for i, (c, p, g) in enumerate(zip(self.cache, params, grads)):
            c = self.rho * c + (1 - self.rho) * np.power(g, 2)
            p -= (self.lr * g / np.sqrt(c + self.epsilon))
            self.cache[i] = c
