import numpy as np

from slugnet.activation import Noop


class Layer(object):
    pass


class Dense(Layer):
    """
    A common densely connected neural network layer.

    The :code:`Dense` layer implements the feed forward operation

    .. math::
        :nowrap:

        \[
            \mathbf{z} = g(\mathbf{W}_i^T \mathbf{x} + \mathbf{b})
        \]

    where :math:`z` is output, :math:`g` is the activation
    function, :math:`W` are weights, :math:`b` is our
    bias, and :math:`i` is the index of the current layer.

    On feed backward, or backpropogation, the :code:`Dense` layer
    calculates two values as follows

    .. math::
        :nowrap:

        \\begin{flalign}
            \\frac{\partial \ell}{\partial \\bm{z}^{(i)}} &=
                g'(z^{(i)}) .*\
                \Big[ \\bm{W}^{(i + 1)^T} \\frac{\partial \ell}{\partial z^{(i + 1)}}\Big] \\\\
            \\frac{\partial \ell}{\partial \\bm{W}^{(i)}} &= \\frac{\partial \ell}{\partial \\bm{z}^{(i)}} \\bm{x}^T
        \\end{flalign}

    When looking at the source, there is a notable absence of
    :math:`\\bm{W}^T` and :math:`\\frac{\partial \ell}{\partial z^{(i + 1)}}`.
    This is because their dot product is calculated in the previous layer.
    The model propogates that gradient to this layer.
    """
    def __init__(self, ind, outd, activation=Noop()):
        self.shape = ind, outd
        self.w = np.random.normal(0, 0.1, self.shape)
        self.activation = activation

    def call(self, X):
        return self.activation.call(np.dot(X, self.w))

    def backprop(self, inp, d_nxt, hidden_output=None):
        """
        Computes the derivative for the next layer
        and computes update for this layers weights
        """
        if hidden_output is not None:
            d_nxt *= self.activation.derivative(hidden_output)

        self.w -= np.dot(inp.T, d_nxt)
        d_prev = np.dot(d_nxt, self.w.T)

        return d_prev
