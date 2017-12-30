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
    r"""
    A layer that implements the convolution operation.

    In the general case, a discrete convolution operation implements
    the function:

    .. math::
        :nowrap:

        \[s(i) = \sum_{a=-\infty}^\infty x(a) w(i - a)\]

    where :math:`x` is the input and :math:`w`
    is the kernel, or in some cases the weighting function.

    In the case of convolutional neural networks, the input
    is typically two dimensional image :math:`I`, and it
    follows that we have a two dimensional kernel :math:`K`.
    Now we can write out convolution function with both axes:

    .. math::
        :nowrap:

        \[S(i, j) = \sum_m \sum_n I(m, n) K(i - m, j - n).\]

    Note that we can write the infinite sum over the domains of
    :math:`m` and :math:`n` as discrete sums because we assume
    that the kernel :math:`K` is zero everywhere but the set of
    points in which we store data [1]_.

    The motivation for using the convolution operation in a
    neural network is best described using an example of an
    image. In a densely connected neural network, each node
    at layer :math:`i` is connected to every node at layer
    :math:`i + 1`. This does not lend itself to image processing,
    where location of a shape relative to another shape is
    important. For instance, finding a right angle involves
    detecting two edges that are perpendicular, *and* whose
    lines cross one another. If we make the kernel smaller
    than the input image, we can process parts of the image
    at a time, thereby ensuring locality of the input signals.
    To process the entire image, we slide the kernel over the
    input, along both axes. At each step, an output is produced
    which will be used as input for the next layer.
    This configuration allows us to learn the parameters of the
    kernel :math:`K` the same way we'd learn ordinary parameters
    in a densely connected neural network.

    .. tikz::

        \def\input {
            0/2.4/a,
            1.2/2.4/b,
            2.4/2.4/c,
            0/1.2/d,
            1.2/1.2/e,
            2.4/1.2/f,
            0/0/g,
            1.2/0/h,
            2.4/0/i
        }

        \def\kernel {
            0/1.2/w,
            1.2/1.2/x,
            0/0/y,
            1.2/0/z
        }

        \def\output {
            0/-4.6/aw + bx + dy + ez,
            3.4/-4.6/bw + cx + ey + fz,
            0/-8/dw + ex + gy + hz,
            3.4/-8/ew + fx + hy + iz
        }

        \draw (0.5,3.8) node {Input};
        \foreach \x/\y/\l in \input
            \draw (\x,\y) -- (\x,\y + 1) -- (\x + 1,\y + 1) -- (\x + 1,\y) -- (\x,\y)
            node[anchor=south west]{$\l$};

        \draw (5.5,2.6) node {Kernel};
        \foreach \x/\y/\l in \kernel
            \draw (\x + 5,\y) -- (\x + 5, \y + 1) -- (\x + 6, \y + 1) -- (\x + 6, \y) -- (\x + 5, \y)
            node[anchor=south west]{$\l$};

        \draw (0.7,-1.3) node {Output};
        \foreach \x/\y/\l in \output
            \draw (\x,\y) -- (\x,\y + 3) -- (\x + 3,\y + 3) -- (\x + 3, \y) -- (\x,\y)
            node[xshift=1.5cm, yshift=1.5cm]{\footnotesize $\l$};

        \draw (1.1,3.5) -- (3.5, 3.5) -- (3.5, 1.1) -- (1.1, 1.1) -- (1.1, 3.5);
        \draw (4.9,2.3) -- (7.3, 2.3) -- (7.3, -0.1) -- (4.9, -0.1) -- (4.9, 2.3);

        \draw [-|>] (3.5, 2.3) -- (4.0, 2.3) -- (4.0, -1.5);
        \draw [-|>] (6, -0.1) -- (6, -1.5);

    .. rst-class:: caption

        **Figure 1:** An example of a two dimension convolution operation. The
        input is an image in :math:`\mathds{R}^{3 \times 3}`, and the kernel is
        in :math:`\mathds{R}^{2 \times 2}`. As the kernel is slid over the input
        with a stride width of one, an output in
        :math:`\mathds{R}^{2 \times 2}` is produced. In the example, the arrows
        and boxes demonstrate how the upper-right portion of the input image
        are compbined with the kernel parameters to produce the upper right
        unit of output.

        --Modified from source: Goodfellow, Bengio, Courville (Deep Learning,
        2016, Figure 9.1).

    The stride width determines how far the kernel moves at each step. Of
    course, to learn anything interesting, we require multiple kernels at
    each layer. These are all configurable hyperparameters that can be set
    upon network instantiation. When the network is operating in feedforward
    mode, the output at each layer is a three dimensional tensor, rather than
    a matrix. This is due to the fact that each kernel produces its own
    two dimensional output, and there are multiple kernels at every layer.

    :param nb_kernel: The number of kernels to use.
    :type nb_kernel: int

    :param kernel_size: The size of the kernel as a tuple, heigh by width
    :type kernel_size: (int, int)

    :param stride: The stide width to use
    :type stride: int

    :param init: The initializer to use
    :type init: slugnet.initializations.Initializer

    :param activation: The activation function to be used at the layer.
    :type activation: slugnet.activation.Activation


    .. [1] Goodfellow, Bengio, Courville (2016), Deep Learning, Chapter 9,
         http://www.deeplearningbook.org
    """
    def __init__(self, ind, nb_kernel, kernel_size, stride=1,
            init=GlorotUniform(), activation=ReLU()):

        self.nb_kernel = nb_kernel
        self.kernel_size = kernel_size
        self.ind = ind
        self.stride = stride

        self.w, self.b = None, None
        self.dw, self.db = None, None

        self.outd = None
        self.last_output = None
        self.last_input = None

        self.activation = activation
        self.w = init((self.nb_kernel, self))
