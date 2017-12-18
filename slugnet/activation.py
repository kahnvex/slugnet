import numpy as np


class Activation(object):
    pass


class Noop(Activation):
    def call(self, x):
        return x

    def derivative(self, x=1.):
        return x


class ReLU(Activation):
    """
    The common rectified linean unit, or ReLU activation funtion.

    A rectified linear unit implements the nonlinear function
    :math:`\phi(z) = \\text{max}\{0, z\}`.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt

        z = np.arange(-2, 2, .1)
        zero = np.zeros(len(z))
        y = np.max([zero, z], axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(z, y)
        ax.set_ylim([-1.0, 2.0])
        ax.set_xlim([-2.0, 2.0])
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('phi(z)')
        ax.set_title('Rectified linear unit')

        plt.show()
    """
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


class Tanh(Activation):
    """
    The hyperbolic tangent activation function.

    A hyperbolic tangent activation function implements the
    nonlinearity given by :math:`\phi(z) = \\text{tanh}(z)`, which is
    equivalent to :math:`\\sfrac{\\text{sinh}(z)}{\\text{cosh}(z)}`.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt

        z = np.arange(-2, 2, .01)
        phi_z = np.tanh(z)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(z, phi_z)
        ax.set_ylim([-1.0, 1.0])
        ax.set_xlim([-2.0, 2.0])
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('phi(z)')
        ax.set_title('Hyperbolic Tangent')

        plt.show()
    """
    def call(self, x):
        self.last_forward = np.tanh(x)

        return self.last_forward

    def derivative(self, x=None):
        h = self.call(x) if x else self.last_forward

        return 1 - h**2


class Sigmoid(Activation):
    """
    Represent a probability distribution over two classes.

    The sigmoid function is given by :math:`\phi(z) = \\frac{1}{1 + e^{-z}}`.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt

        z = np.arange(-4, 4, .01)
        phi_z = 1 / (1 + np.exp(-z))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(z, phi_z)
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([-4.0, 4.0])
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('phi(z)')
        ax.set_title('Sigmoid')

        plt.show()
    """
    def call(self, x):
        self.last_out = 1. / (1. + np.exp(-x))

        return self.last_out

    def derivative(self, x=None):
        z = self.call(x) if x else self.last_out

        return z * (1 - z)


class Softmax(Activation):
    r"""
    Represent a probability distribution over :math:`n` classes.

    The softmax activation function is given by

    .. math::

        \phi(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \,
        \forall \, i \in \{1, \dots, K\}

    where :math:`K` is the number of classes. We can see that softmax is
    a generalization of the sigmoid function to :math:`n` classes. Below,
    we derive the sigmoid function using softmax with two classes.

    .. math::

        \phi(z_1) &= \frac{e^{z_1}}{\sum_{i=1}^2 e^{z_i}} \\
                  &= \frac{1}{e^{z_1 - z_1} + e^{z_2 - z_1}} \\
                  &= \frac{1}{1 + e^{-z_1}}, \, \text{substituting} \, z_2 = 0

    We substitute :math:`z_2 = 0` because we only need one variable to
    represent the probability distribution over two classes. This leaves
    us with the definition of the sigmoid function.
    """
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
