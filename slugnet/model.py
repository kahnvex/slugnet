class Model(object):
    def __init__(self, lr=0.1, n_epoch=500000, layers=[]):
        self.layers = layers
        self.lr = lr
        self.n_epoch = n_epoch

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, y):

        for epoch in range(self.n_epoch):
            out = X
            ins = []

            for layer in self.layers:
                ins.append(out)
                out = layer.call(out)

            error = y - out
            dz = error * self.lr

            if epoch % 5000 == 0:
                print('error sum %s' % sum(error))

            for i, layer in list(enumerate(self.layers))[::-1]:
                if i == len(self.layers) - 1:
                    dz = layer.backprop(ins[i], dz)

                else:
                    dz = layer.backprop(ins[i], dz, ins[i + 1])

    def transform(self, X):
        for layer in self.layers:
            X = layer.call(X)

        return X
