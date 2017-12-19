import numpy as np

from slugnet.activation import Noop
from slugnet.initializations import _zero, GlorotUniform


class Layer(object):
    pass


class Dense(Layer):
    r"""
    A common densely connected neural network layer.

    The :code:`Dense` layer implements the feed forward operation

    .. math::
        :nowrap:

        \[
            \bm{a} = \phi(\bm{W}^T \bm{x} + \bm{b})
        \]

    where :math:`\bm{a}` is activated output, :math:`\phi`
    is the activation function, :math:`\bm{W}` are weights,
    :math:`\bm{b}` is our bias.

    On feed backward, or backpropogation, the :code:`Dense` layer
    calculates two values as follows

    .. math::
        :nowrap:

        \begin{flalign}
            \frac{\partial \ell}{\partial \bm{z}^{(i)}} &=
                g'(z^{(i)}) \circ
                \Big[ \bm{W}^{(i + 1)^T}
                \frac{\partial \ell}{\partial z^{(i + 1)}}\Big] \\
            \frac{\partial \ell}{\partial \bm{W}^{(i)}} &=
                \frac{\partial \ell}{\partial \bm{z}^{(i)}} \bm{x}^T
        \end{flalign}

    When looking at the source, there is a notable absence of
    :math:`\bm{W}^{(i + 1)^T}`
    and :math:`\frac{\partial \ell}{\partial z^{(i + 1)}}`.
    This is because their dot product is calculated in the previous layer.
    The model propogates that gradient to this layer.
    """

    def __init__(self, ind, outd, activation=Noop(), init=GlorotUniform()):
        self.shape = ind, outd
        self.w = init(self.shape)
        self.b = _zero((outd, ))
        self.dw = None
        self.db = None
        self.activation = activation

    def call(self, X):
        self.last_X = X

        return self.activation.call(np.dot(X, self.w) + self.b)

    def backprop(self, nxt_grad):
        """
        Computes the derivative for the next layer
        and computes update for this layers weights
        """
        act_grad = nxt_grad * self.activation.derivative()
        self.dw = np.dot(self.last_X.T, act_grad)
        self.db = np.mean(act_grad, axis=0)

        return np.dot(act_grad, self.w.T)

    def get_params(self):
        return self.w, self.b

    def get_grads(self):
        return self.dw, self.db
