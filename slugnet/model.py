import numpy as np

from slugnet.optimizers import SGD
from slugnet.loss import BinaryCrossEntropy
from sklearn.model_selection import train_test_split


class Model(object):
    """
    Models implement functionality for fitting neural networks and
    making predictions.
    """
    def __init__(self, lr=0.1, n_epoch=400000, batch_size=32, layers=[], l1=0.0,
                 l2=0.0, optimizer=SGD(), loss=BinaryCrossEntropy(),
                 validation_split=0.2, metrics=['loss']):
        self.layers = layers
        self.lr = lr
        self.n_epoch = n_epoch
        self.l1 = l1
        self.l2 = l2
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.metrics = metrics

    def add_layer(self, layer):
        self.layers.append(layer)

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.call(X)

        return X

    def backpropogation(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backprop(grad)

    def log_metrics(self, yh, y, epoch, title='training'):
        if 'loss' in self.metrics:
            loss = self.loss.forward(yh, y)
            print('%s loss at epoch %s: %s' % (title, epoch, loss))

        if 'accuracy' in self.metrics:
            acc = self.accuracy(yh, y)
            print('%s accuracy at epoch %s: %s' % (title, epoch, acc))

    def accuracy(self, yh, y):
        y_predicts = np.argmax(yh, axis=1)
        y_targets = np.argmax(y, axis=1)
        acc = y_predicts == y_targets

        return np.mean(acc)

    def fit(self, X, y):
        """
        Train the model given samples :code:`X` and labels or values :code`y`.
        """

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=self.validation_split)
        n_samples = X_train.shape[0]

        for epoch in range(self.n_epoch):

            for batch in range(n_samples // self.batch_size):
                batch_start = self.batch_size * batch
                batch_end = batch_start + self.batch_size
                X_mb = X_train[batch_start:batch_end]
                y_mb = Y_train[batch_start:batch_end]

                yhi = self.feedforward(X_mb)
                grad = self.loss.backward(yhi, y_mb)
                self.backpropogation(grad)

                params = []
                grads = []

                for layer in self.layers:
                    params += layer.get_params()
                    grads += layer.get_grads()

                self.optimizer.update(params, grads)

            val_yh = self.feedforward(X_test)
            self.log_metrics(yhi, y_mb, epoch, title='training')
            self.log_metrics(val_yh, Y_test, epoch, title='validation')

    def transform(self, X):
        """
        Predict the labels or values of some input matrix :code:`X`.
        """
        return self.feedforward(X)[0]
