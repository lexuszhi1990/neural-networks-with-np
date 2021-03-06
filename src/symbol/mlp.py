import numpy as np

from src.activation import *
from src.layer import *
from src.loss import softmax_loss


class mlp(object):

    def __init__(self, num_cls=10, input_shape=(1, 28, 28), reg=0.0,
                 weight_scale=5e-3):
        self.name = 'mlp'
        self.reg = reg
        self.params = {}
        self.init_params(input_shape, num_cls)


    def init_params(self, input_shape, num_cls):
        # (1, 28, 28) -> (256, 1)
        self.params['l1_weight'] = np.random.randn(np.prod(input_shape), 512).astype(np.float32) * np.sqrt(2.0/np.prod(input_shape))
        self.params['l1_bias'] = np.zeros(512).astype(np.float32)
        # (1024, 1) -> (2048, 1)
        self.params['l2_weight'] = np.random.randn(512, 1024).astype(np.float32) * np.sqrt(2.0/512)
        self.params['l2_bias'] = np.zeros(1024).astype(np.float32)
        # (1024, 1) -> (10, 1)
        self.params['l3_weight'] = np.random.randn(1024, num_cls).astype(np.float32) * np.sqrt(2.0/1024)
        self.params['l3_bias'] = np.zeros(num_cls).astype(np.float32)

    def forward(self, x):
        x, self.layer1_params = Linear(x, self.params['l1_weight'], self.params['l1_bias'])
        x, self.layer1_avt_params = leaky_relu_forward(x)
        x, self.layer2_params = Linear(x, self.params['l2_weight'], self.params['l2_bias'])
        x, self.layer2_avt_params = leaky_relu_forward(x)
        x, self.layer3_params = Linear(x, self.params['l3_weight'], self.params['l3_bias'])

        self.output = x

        return self.output


    def compute_loss(self, labels_batch):
        train_loss, self.d_output = softmax_loss(self.output, labels_batch)
        weights_reg = sum([np.sum(self.params['l%d_weight'%i] ** 2) for i in range(1, 4)])
        reg_loss = 0.5 * self.reg * weights_reg

        return train_loss, reg_loss


    def backward(self):
        grads = {}

        d_layer3, grads['l3_weight'], grads['l3_bias'] = Linear_backward(self.d_output, self.layer3_params)
        d_layer3_avt = leaky_relu_backward(d_layer3, self.layer2_avt_params)
        d_layer2, grads['l2_weight'], grads['l2_bias'] = Linear_backward(d_layer3_avt, self.layer2_params)
        d_layer2_avt = leaky_relu_backward(d_layer2, self.layer1_avt_params)
        # d_layer2_avt = sigmoid_backword(d_layer2, self.layer1_avt_params)
        d_layer1, grads['l1_weight'], grads['l1_bias'] = Linear_backward(d_layer2_avt, self.layer1_params)

        grads['l2_weight'] += self.reg * self.params['l2_weight']
        grads['l1_weight'] += self.reg * self.params['l1_weight']

        return grads

