import numpy as np

from slugnet.optimizers import SGD
from slugnet.loss import BinaryCrossEntropy


class Model(object):
    """
    Models implement functionality for fitting neural networks and
    making predictions.
    """
    def __init__(self, lr=0.1, n_epoch=400000, n_batches=1, layers=[], l1=0.0,
                 l2=0.0, optimizer=SGD(), loss=BinaryCrossEntropy()):
        self.layers = layers
        self.lr = lr
        self.n_epoch = n_epoch
        self.l1 = l1
        self.l2 = l2
        self.n_batches = n_batches
        self.optimizer = optimizer
        self.loss = loss

    def add_layer(self, layer):
        self.layers.append(layer)

    def _forward(self, X):
        for layer in self.layers:
            X = layer.call(X)

        return X

    def _backward(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backprop(grad)

    def fit(self, X, y):
        """
        Train the model given samples :code:`X` and labels or values :code`y`.
        """
        for epoch in range(self.n_epoch):
            X_mb = np.array_split(X, self.n_batches)
            y_mb = np.array_split(y, self.n_batches)

            for Xi, yi in zip(X_mb, y_mb):

                yhi = self._forward(Xi)
                loss = self.loss.forward(yhi, yi)
                grad = self.loss.backward(yhi, yi)
                self._backward(grad)

                if epoch % 5000:
                    print('loss at %s: %s' % (epoch, loss))

                params = []
                grads = []

                for layer in self.layers:
                    params += layer.get_params()
                    grads += layer.get_grads()

                self.optimizer.update(params, grads)

    def transform(self, X):
        """
        Predict the labels or values of some input matrix :code:`X`.
        """
        return self._forward(X)[0]
