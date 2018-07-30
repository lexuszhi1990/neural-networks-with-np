# -*- coding: utf-8 -*-

import unittest

from src.activation import *

class TestActivation(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[-0.5, 0.5],
                           [-1.0, 1.0]]).astype(np.float32)

        self.alpha = 0.2
        self.leaky_relu_res = np.array([[-0.1, 0.5],
                                        [-0.2, 1.0]]).astype(np.float32)
        self.leaky_relu_d_res = np.array([[self.alpha, 1],
                                        [self.alpha, 1]]).astype(np.float32)

    def test_leaky_relu(self):
        res = leaky_relu(self.x, alpha=self.alpha)
        self.assertEqual(0, np.sum(res - self.leaky_relu_res))

    def test_d_leaky_relu(self):
        res = d_leaky_relu(self.x, alpha=self.alpha)
        # https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertAlmostEqual
        self.assertAlmostEqual(0, np.sum(res - self.leaky_relu_d_res))

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([
        TestActivation("test_leaky_relu"),
        TestActivation("test_d_leaky_relu"),
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)
