from slugnet.activation import ReLU, Softmax
from slugnet.layers import Dense, Dropout
from slugnet.loss import SoftmaxCategoricalCrossEntropy as SCCE
from slugnet.model import Model
from slugnet.optimizers import RMSProp
from slugnet.data.mnist import get_mnist


X, y = get_mnist()
model = Model(lr=0.01, n_epoch=3, loss=SCCE(),
              metrics=['loss', 'accuracy'],
              optimizer=RMSProp())

model.add_layer(Dense(200, inshape=784, activation=ReLU()))
model.add_layer(Dropout(0.5))
model.add_layer(Dense(10, activation=Softmax()))

model.fit(X, y)
