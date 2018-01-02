import numpy as np

from slugnet.activation import Noop, ReLU
from slugnet.initializations import _zero, GlorotUniform


class Layer(object):
    first_layer = False

    def set_first_layer(self):
        self.first_layer = True

    def get_params(self):
        return []

    def get_grads(self):
        return []


class Dense(Layer):
    r"""
    A common densely connected neural network layer.

    :param outshape: The output shape at this layer.
    :type outshape: int

    :param inshape: The input shape at this layer.
    :type inshape: int

    :param activation: The activation function to be used at the layer.
    :type activation: slugnet.activation.Activation

    :param init: The initialization function to be used
    :type init: slugnet.initializations.Initializer
    """

    def __init__(self, outshape, inshape=None, activation=Noop(),
                 init=GlorotUniform()):
        self.outshape = None, outshape
        self.activation = activation
        self.inshape = inshape
        self.init = init

    def connect(self, prev_layer=None):
        if prev_layer:
            if len(prev_layer.outshape) != 2:
                raise ValueError('Previous layers outshape is incompatible')

            self.inshape = prev_layer.outshape[-1]
        elif not self.inshape:
            raise ValueError('inshape must be given to first layer of network')

        self.shape = self.inshape, self.outshape[-1]
        self.w = self.init(self.shape)
        self.b = _zero((self.outshape[-1], ))
        self.dw = None
        self.db = None

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
    A layer that removes units from a network with probability :code:`p`.

    :param p: The probability of a non-ouput node being removed from the network.
    :type p: float
    """
    def __init__(self, p=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def connect(self, prev_layer=None):
        self.outshape = prev_layer.outshape

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
    """
    def __init__(self, nb_kernel, kernel_size, stride=1,
            inshape=None, init=GlorotUniform(), activation=None):

        if activation is None:
            activation = ReLU()
        self.inshape = inshape
        self.nb_kernel = nb_kernel
        self.kernel_size = kernel_size
        self.stride = stride

        self.w, self.b = None, None
        self.dw, self.db = None, None

        self.outshape = None
        self.last_output = None
        self.last_input = None

        self.init = init
        self.activation = activation

    def connect(self, prev_layer=None):
        if prev_layer:
            self.inshape = prev_layer.outshape
        elif not self.inshape:
            raise ValueError('inshape must be given to first layer of network')

        prev_nb_kernel = self.inshape[1]

        kernel_h, kernel_w = self.kernel_size
        self.w = self.init((self.nb_kernel, prev_nb_kernel, kernel_h, kernel_w))
        self.b = _zero((self.nb_kernel,))

        prev_kh, prev_kw = self.inshape[2], self.inshape[3]
        height = (prev_kh - kernel_h) // self.stride + 1
        width = (prev_kw - kernel_w) // self.stride + 1
        self.outshape = (self.inshape[0], self.nb_kernel, height, width)

    def call(self, X, *args, **kwargs):
        self.last_input = X
        batch_size, depth, height, width = X.shape
        kernel_h, kernel_w = self.kernel_size
        out_h, out_w = self.outshape[2:]

        outputs = _zero((batch_size, self.nb_kernel, out_h, out_w))

        for x in np.arange(batch_size):
            for y in np.arange(self.nb_kernel):
                for h in np.arange(out_h):
                    for w in np.arange(out_w):
                        h1, w1 = h * self.stride, w * self.stride
                        h2, w2 = h1 + kernel_h, w1 + kernel_w
                        patch = X[x, :, h1: h2, w1: w2]
                        conv_product = patch * self.w[y]
                        outputs[x, y, h, w] = np.sum(conv_product) + self.b[y]

        self.last_output = self.activation.call(outputs)

        return self.last_output

    def backprop(self, grad, *args, **kwargs):
        batch_size, depth, input_h, input_w = self.last_input.shape
        out_h, out_w = self.outshape[2:]
        kernel_h, kernel_w = self.kernel_size

        # gradients
        self.dw = _zero(self.w.shape)
        self.db = _zero(self.b.shape)
        delta = grad * self.activation.derivative()

        # dw
        for r in np.arange(self.nb_kernel):
            for t in np.arange(depth):
                for h in np.arange(kernel_h):
                    for w in np.arange(kernel_w):
                        input_window = self.last_input[:, t,
                                       h:input_h - kernel_h + h + 1:self.stride,
                                       w:input_w - kernel_w + w + 1:self.stride]
                        delta_window = delta[:, r]
                        self.dw[r, t, h, w] = np.sum(input_window * delta_window) / batch_size

        # db
        for r in np.arange(self.nb_kernel):
            self.db[r] = np.sum(delta[:, r]) / batch_size

        # dX
        if not self.first_layer:
            layer_grads = _zero(self.last_input.shape)
            for b in np.arange(batch_size):
                for r in np.arange(self.nb_kernel):
                    for t in np.arange(depth):
                        for h in np.arange(out_h):
                            for w in np.arange(out_w):
                                h1, w1 = h * self.stride, w * self.stride
                                h2, w2 = h1 + kernel_h, w1 + kernel_w
                                layer_grads[b, t, h1:h2, w1:w2] += self.w[r, t] * delta[b, r, h, w]
            return layer_grads


class MeanPooling(Layer):
    r"""
    Pool outputs using the arithmetic mean.
    """

    def __init__(self, pool_size, inshape=None):
        self.pool_size = pool_size

        self.outshape = None
        self.inshape = inshape

    def connect(self, prev_layer=None):
        if prev_layer:
            self.inshape = prev_layer.outshape
        elif not self.inshape:
            raise ValueError('inshape must be given to first layer of network')

        old_h, old_w = self.inshape[-2:]
        pool_h, pool_w = self.pool_size
        new_h, new_w = old_h // pool_h, old_w // pool_w

        self.outshape = self.inshape[:-2] + (new_h, new_w)

    def call(self, X, *args, **kwargs):
        self.inshape = X.shape
        pool_h, pool_w = self.pool_size
        new_h, new_w = self.outshape[-2:]

        # forward
        outputs = _zero(self.inshape[:-2] + self.outshape[-2:])

        if np.ndim(X) == 4:
            nb_batch, nb_axis, _, _ = X.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            outputs[a, b, h, w] = np.mean(X[a, b, h:h + pool_h, w:w + pool_w])

        elif np.ndim(X) == 3:
            nb_batch, _, _ = X.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        outputs[a, h, w] = np.mean(X[a, h:h + pool_h, w:w + pool_w])

        else:
            raise ValueError()

        return outputs

    def backprop(self, pre_grad, *args, **kwargs):
        new_h, new_w = self.outshape[-2:]
        pool_h, pool_w = self.pool_size
        length = np.prod(self.pool_size)

        layer_grads = _zero(self.inshape)

        if np.ndim(pre_grad) == 4:
            nb_batch, nb_axis, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            h1, w1 = h * pool_h, w * pool_w
                            h2, w2 = h1 + pool_h, w1 + pool_w
                            layer_grads[a, b, h1:h2, w1:w2] = pre_grad[a, b, h, w] / length

        elif np.ndim(pre_grad) == 3:
            nb_batch, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        h_shift, w_shift = h * pool_h, w * pool_w
                        layer_grads[a, h_shift: h_shift + pool_h, w_shift: w_shift + pool_w] = \
                            pre_grad[a, h, w] / length

        else:
            raise ValueError()

        return layer_grads


class Flatten(Layer):
    def __init__(self, outdim=2, inshape=None):
        self.outdim = outdim
        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)

        self.last_input_shape = None
        self.outshape = None
        self.inshape = inshape

    def connect(self, prev_layer=None):
        if prev_layer:
            self.inshape = prev_layer.outshape
        elif not self.inshape:
            raise ValueError('inshape must be given to first layer of network')

        to_flatten = np.prod(self.inshape[self.outdim - 1:])
        flattened_shape = self.inshape[:self.outdim - 1] + (to_flatten,)
        self.outshape = flattened_shape

    def call(self, X, *args, **kwargs):
        self.last_input_shape = X.shape

        # to_flatten = np.prod(self.last_input_shape[self.outdim-1:])
        # flattened_shape = input.shape[:self.outdim-1] + (to_flatten, )
        flattened_shape = X.shape[:self.outdim - 1] + (-1,)
        return np.reshape(X, flattened_shape)

    def backprop(self, pre_grad, *args, **kwargs):
        return np.reshape(pre_grad, self.last_input_shape)
