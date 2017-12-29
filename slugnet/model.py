import numpy as np

from tabulate import tabulate
from tqdm import tqdm, trange

from slugnet.optimizers import SGD
from slugnet.loss import BinaryCrossEntropy
from sklearn.model_selection import train_test_split


class Model(object):
    """
    A model implement functionality for fitting a neural network and
    making predictions.

    Parameters
    ----------

    :param lr: The learning rate to be used during training.
    :type lr: float

    :param n_epoch: The number of training epochs to use.
    :type n_epoch: int

    :param batch_size: The size of each batch for training.
    :type batch_size: int

    :param layers: Initial layers to add the the network, more can
        be added layer using the :code:`model.add_layer` method.
    :type layers: list[slugnet.layers.Layer]

    :param optimizer: The opimization method to use during training.
    :type optimizer: slugnet.optimizers.Optimizer

    :param loss: The loss function to use during training and validation.
    :type loss: slugnet.loss.Objective

    :param validation_split: The percent of data to use for validation,
        default is zero.
    :type validation_split: float

    :param metrics: The metrics to print during training, options are
        :code:`loss` and :code:`accuracy`.
    :type metrics: list[str]

    :param progress: Display progress-bar while training.
    :type progress: bool

    :param log_interval: The epoch interval on which to print progress.
    :type log_interval: int
    """
    def __init__(self, lr=0.1, n_epoch=400000, batch_size=32, layers=None,
                 optimizer=SGD(), loss=BinaryCrossEntropy(),
                 validation_split=0.2, metrics=['loss'], progress=True,
                 log_interval=1):
        self.layers = layers if layers else []
        self.lr = lr
        self.n_epoch = n_epoch
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.metrics = metrics
        self.progress = progress
        self.log_interval = log_interval

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_n_output(self):
        out_layer = self.layers[-1]
        ind, outd = out_layer.shape

        return outd

    def feedforward(self, X, train=True):
        for layer in self.layers:
            X = layer.call(X, train=train)

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
        outd = self.get_n_output()
        self.metrics_dict = {
            'yh': np.empty(dtype=np.float64, shape=(0, outd)),
            'y': np.empty(dtype=np.int, shape=(0, outd))
        }

    def log_metrics(self, metrics, epoch):
        validation = 'val' in metrics
        header = ['run', 'epoch'] + self.metrics
        train, val = ['train', epoch], ['validation', epoch]

        for metric_name in self.metrics:
            train.append(metrics['train'][metric_name])
            if validation:
                val.append(metrics['val'][metric_name])

        if validation:
            tqdm.write(tabulate([train, val], header, tablefmt='grid'))
        else:
            tqdm.write(tabulate([train], header, tablefmt='grid'))

    def stash_predictions(self, yh, y):
        yh_concat = [self.metrics_dict['yh'], yh]
        y_concat = [self.metrics_dict['y'], y]

        self.metrics_dict['yh'] = np.concatenate(yh_concat)
        self.metrics_dict['y'] = np.concatenate(y_concat)

    def get_predictions(self):
        return self.metrics_dict['yh'], self.metrics_dict['y']

    def accuracy(self, yh, y):
        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            yh = np.rint(yh)
            acc = yh == y
        else:
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

            metrics = {}
            train_yh, train_y = self.get_predictions()
            metrics['train'] = self.get_metrics(train_yh, train_y)

            if self.validation_split > 0:
                val_yh = self.feedforward(X_test)
                metrics['val'] = self.get_metrics(val_yh, Y_test)

            if epoch % self.log_interval == 0:
                self.log_metrics(metrics, epoch)

        return metrics


    def transform(self, X):
        """
        Predict the labels or values of some input matrix :code:`X`.
        """
        return self.feedforward(X, train=False)[0]
