import numpy as np

from tabulate import tabulate
from tqdm import tqdm, trange

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
                 validation_split=0.2, metrics=['loss'], progress=True):
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
        self.progress = progress

    def add_layer(self, layer):
        self.layers.append(layer)

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.call(X)

        return X

    def backpropogation(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backprop(grad)

    def get_metrics(self, yh, y):
        metrics = {}

        if 'loss' in self.metrics:
            metrics['loss'] = self.loss.forward(yh, y)

        if 'accuracy' in self.metrics:
            metrics['accuracy'] = self.accuracy(yh, y)

        return metrics

    def init_predictions(self):
        self.metrics_dict = {
            'yh': np.empty(dtype=np.float64, shape=(0, 10)),
            'y': np.empty(dtype=np.int, shape=(0, 10))
        }

    def log_metrics(self, metrics, epoch):
        header = ['run', 'epoch'] + self.metrics
        train, val = ['train', epoch], ['validation', epoch]

        for metric_name in self.metrics:
            train.append(metrics['train'][metric_name])
            val.append(metrics['val'][metric_name])

        tqdm.write(tabulate([train, val], header, tablefmt="grid"))

    def stash_predictions(self, yh, y):
        yh_concat = [self.metrics_dict['yh'], yh]
        y_concat = [self.metrics_dict['y'], y]

        self.metrics_dict['yh'] = np.concatenate(yh_concat)
        self.metrics_dict['y'] = np.concatenate(y_concat)

    def get_predictions(self):
        return self.metrics_dict['yh'], self.metrics_dict['y']

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

        epoch_iter = trange(self.n_epoch, total=self.n_epoch,
                            disable=not self.progress)

        for epoch in epoch_iter:
            epoch_iter.set_description('Epoch %s' % epoch)
            self.init_predictions()

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
                self.stash_predictions(yhi, y_mb)

            val_yh = self.feedforward(X_test)
            train_yh, train_y = self.get_predictions()
            metrics = {}
            metrics['train'] = self.get_metrics(train_yh, train_y)
            metrics['val'] = self.get_metrics(val_yh, Y_test)
            self.log_metrics(metrics, epoch)

    def transform(self, X):
        """
        Predict the labels or values of some input matrix :code:`X`.
        """
        return self.feedforward(X)[0]
