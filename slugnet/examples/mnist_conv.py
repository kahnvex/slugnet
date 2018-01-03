import numpy as np

from slugnet.activation import ReLU, Softmax
from slugnet.layers import Convolution, Dense, MeanPooling, Flatten
from slugnet.loss import SoftmaxCategoricalCrossEntropy as SCCE
from slugnet.model import Model
from slugnet.optimizers import SGD
from slugnet.data.mnist import get_mnist


X, y = get_mnist()
X = X.reshape((-1, 1, 28, 28)) / 255.0
np.random.seed(100)
X = np.random.permutation(X)[:1000]
np.random.seed(100)
y = np.random.permutation(y)[:1000]

model = Model(lr=0.001, n_epoch=100, batch_size=3, loss=SCCE(),
              metrics=['loss', 'accuracy'], optimizer=SGD())

model.add_layer(Convolution(1, (3, 3), inshape=(None, 1, 28, 28)))
model.add_layer(MeanPooling((2, 2)))
model.add_layer(Convolution(2, (4, 4)))
model.add_layer(MeanPooling((2, 2)))
model.add_layer(Flatten())
model.add_layer(Dense(10, activation=Softmax()))

model.fit(X, y)
