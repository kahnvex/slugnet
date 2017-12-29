import numpy as np

from slugnet.activation import Noop, ReLU
from slugnet.initializations import _zero, GlorotUniform


class Layer(object):
    def get_params(self):
        return []

    def get_grads(self):
        return []


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
            \frac{\partial \ell}{\partial \bm{a}^{(i)}} &=
                \Big[ \bm{W}^{(i + 1)^T}
                \frac{\partial \ell}{\partial \bm{a}^{(i + 1)}}\Big]
                \circ \phi'(\bm{a}^{(i)}) \\
            \frac{\partial \ell}{\partial \bm{W}^{(i)}} &=
                \frac{\partial \ell}{\partial \bm{a}^{(i)}} \bm{x}^T
        \end{flalign}

    When looking at the source, there is a notable absence of
    :math:`\bm{W}^{(i + 1)^T}`
    and :math:`\frac{\partial \ell}{\partial \bm{a}^{(i + 1)}}`.
    This is because their dot product is calculated in the previous layer.
    The model propogates that gradient to this layer.

    :param ind: The input dimension at this layer.
    :type ind: int

    :param outd: The output dimension at this layer.
    :type outd: int

    :param activation: The activation function to be used at the layer.
    :type activation: slugnet.activation.Activation

    :param init: The initialization function to be used
    :type init: slugnet.initializations.Initializer
    """

    def __init__(self, ind, outd, activation=Noop(), init=GlorotUniform()):
        self.shape = ind, outd
        self.w = init(self.shape)
        self.b = _zero((outd, ))
        self.dw = None
        self.db = None
        self.activation = activation

    def call(self, X, *args, **kwargs):
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


class Dropout(Layer):
    r"""
    Dropout is a method of regularization that trains subnetworks by turning
    off non-output nodes with some probability :math:`p`.

    This approximates bagging, which involves training an ensemble of models
    to overcome weaknesses in any given model [1]_.

    We can formalize dropout by representing the subnetworks created by dropout
    with a mask vector :math:`\bm{\mu}`. Now, we note each subnetwork defines a
    new probability distribution of :math:`y` as
    :math:`\mathds{P}(y | \bm{x}, \bm{\mu})` [1]_. If we define
    :math:`\mathds{P}(\bm{\mu})` as the probability distribution of mask vectors
    :math:`\bm{\mu}`, we can write the mean of all subnetworks as

    .. math::
        :nowrap:

        \[
            \sum_{\bm{\mu}} \mathds{P}(\bm{\mu}) \mathds{P}(y | \bm{x}, \bm{\mu}).
        \]

    The problem with evaluating this term is the exponential number of mask
    vectors. In practice, we approximate this probability distribution by
    including all nodes during inference, and multiplying each output by
    :math:`1 - p`, the probability that any node is included in the network during
    training. This rule is called the weight scaling inference rule [1]_.

    :param p: The probability of a non-ouput node being removed from the network.
    :type p: float

    .. [1] Goodfellow, Bengio, Courville (2016), Deep Learning, http://www.deeplearningbook.org
    """
    def __init__(self, p=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def call(self, X, train=True, *args, **kwargs):
        if 0. > self.p or self.p > 1.:
            return X

        if not train:
            return X * (1 - self.p)

        binomial = np.random.binomial(1, 1 - self.p, X.shape)
        self.last_mask = binomial / (1 - self.p)

        return X * self.last_mask

    def backprop(self, pre_grad, *args, **kwargs):
        if 0. < self.p < 1.:
            return pre_grad * self.last_mask

        return pre_grad


class Convolution(Layer):
    """
    A layer that implements the convolution operation.

    In the general case, a discrete convolution operation implements
    the function:

    .. math::
        :nowrap:

        \[s(i) = \sum_{a=-\infty}^\infty x(a) w(i - a)\]

    where :math:`x` is the input and :math:`w`
    is the kernel, or in some cases the weighting function.
    """
    def __init__(self, ind, nb_filter, filter_size, stride=1,
            init=GlorotUniform(), activation=ReLU()):

        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.ind = ind
        self.stride = stride

        self.w, self.b = None, None
        self.dw, self.db = None, None

        self.outd = None
        self.last_output = None
        self.last_input = None

        self.activation = activation
        self.w = init((self.nb_filter, self))
