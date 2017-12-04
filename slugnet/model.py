import numpy as np


class Model(object):
    def __init__(self, lr=0.1, n_epoch=50000, n_batches=1, layers=[], l1=0.0,
                 l2=0.0):
        self.layers = layers
        self.lr = lr
        self.n_epoch = n_epoch
        self.l1 = l1
        self.l2 = l2
        self.n_batches = n_batches

    def add_layer(self, layer):
        self.layers.append(layer)

    def _forward(self, X):
        out = X
        ins = []

        for layer in self.layers:
            ins.append(out)
            out = layer.call(out)

        return out, ins

    def _backward(self, X, ins, out, y):
        error = y - out
        dz = error * self.lr

        for i, layer in list(enumerate(self.layers))[::-1]:
            if i == len(self.layers) - 1:
                dz = layer.backprop(ins[i], dz)

            else:
                dz = layer.backprop(ins[i], dz, ins[i + 1])

    def fit(self, X, y):

        for epoch in range(self.n_epoch):
            X_mb = np.array_split(X, self.n_batches)
            y_mb = np.array_split(y, self.n_batches)

            for Xi, yi in zip(X_mb, y_mb):

                out, ins = self._forward(Xi)
                grads = self._backward(Xi, ins, out, y)

    def transform(self, X):
        return self._forward(X)[0]
