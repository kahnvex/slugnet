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
    """Computes the binary cross-entropy between predictions and targets.

    Returns
    -------
    numpy array
        An expression for the element-wise binary cross-entropy.
    Notes
    -----
    This is the loss function of choice for binary classification problems
    and sigmoid output units.

    """
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, yh, y):
        """Forward pass.

        .. math::
            :nowrap:

            \\begin{flalign}
                &L_i = y_i \\log(\hat{y}_i) +
                    (1 - y_i) \\log(1 - \hat{y}_i) \\\\
                &L = -\\frac{1}{N} \sum_{i=1}^N L_i
            \\end{flalign}

        Parameters
        ----------
        outputs : numpy.array
            Predictions in (0, 1), such as sigmoidal output of a neural network.
        targets : numpy.array
            Targets in [0, 1], such as ground truth labels.
        """
        yh = np.clip(y, self.epsilon, 1 - self.epsilon)
        loss = -np.sum(y * np.log(yh) + (1 - y) * np.log(1 - yh), axis=1)
        return np.mean(loss)

    def backward(self, outputs, targets):
        """Backward pass.
        Parameters

        outputs : numpy.array
            Predictions in (0, 1), such as sigmoidal output of a neural network.
        targets : numpy.array
            Targets in [0, 1], such as ground truth labels.
        """
        outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        divisor = np.maximum(outputs * (1 - outputs), self.epsilon)
        return (outputs - targets) / divisor
