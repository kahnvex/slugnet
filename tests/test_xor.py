import numpy as np
import unittest

from slugnet.model import Model
from slugnet.layers import Dense
from slugnet.activation import Sigmoid


class TestSlugnetOnXOR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(progress=False, validation_split=0, batch_size=4,
                          metrics=['loss', 'accuracy'])

        cls.model.add_layer(Dense(2, 3, Sigmoid()))
        cls.model.add_layer(Dense(3, 1, Sigmoid()))

        cls.X_train = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        cls.Y_train = np.array([
            [0],
            [1],
            [1],
            [0]
        ])
        cls.model.fit(cls.X_train, cls.Y_train)

    def test_xor_01(self):
        output = self.model.transform([0, 1])
        self.assertGreater(output, 0.9)

    def test_xor_10(self):
        output = self.model.transform([1, 0])
        self.assertGreater(output, 0.9)

    def test_xor_00(self):
        output = self.model.transform([0, 0])
        self.assertLess(output, 0.1)

    def test_xor_11(self):
        output = self.model.transform([1, 1])
        self.assertLess(output, 0.1)
