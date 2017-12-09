import numpy as np


class Objective(object):
    """An objective function (or loss function, or optimization score
    function) is one of the two parameters required to compile a model.

    """
    def forward(self, outputs, targets):
        """ Forward function.
        """
        raise NotImplementedError()

    def backward(self, outputs, targets):
        """Backward function.

        Parameters
        ----------
        outputs, targets : numpy.array
            The arrays to compute the derivatives of them.

        Returns
        -------
        numpy.array
            An array of derivative.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class BinaryCrossEntropy(Objective):
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, yh, y):
        yh = np.clip(y, self.epsilon, 1 - self.epsilon)
        loss = -np.sum(y * np.log(yh) + (1 - y) * np.log(1 - yh), axis=1)
        return np.mean(loss)

    def backward(self, outputs, targets):
        outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        divisor = np.maximum(outputs * (1 - outputs), self.epsilon)

        return (outputs - targets) / divisor


class SoftmaxCategoricalCrossEntropy(Objective):
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return np.mean(-np.sum(targets * np.log(outputs), axis=1))

    def backward(self, outputs, targets):
        outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return outputs - targets
