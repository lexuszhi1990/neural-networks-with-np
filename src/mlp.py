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
        self.params['l1_weight'] = np.random.randn(np.prod(input_shape), 1024).astype(np.float32) * 0.005
        self.params['l1_bias'] = np.zeros(1024).astype(np.float32)
        # (1024, 1) -> (10, 1)
        self.params['l2_weight'] = np.random.randn(1024, num_cls).astype(np.float32) * 0.005
        self.params['l2_bias'] = np.zeros(num_cls).astype(np.float32)

    def forward(self, x):
        self.layer1_fw_out, self.layer1_params = Linear(x, self.params['l1_weight'], self.params['l1_bias'])
        # self.layer1_avt_out, self.layer1_avt_params = relu_forward(self.layer1_fw_out)
        self.layer1_avt_out, self.layer1_avt_params = sigmoid_forward(self.layer1_fw_out)
        self.layer2_fw_out, self.layer2_params = Linear(self.layer1_avt_out, self.params['l2_weight'], self.params['l2_bias'])

        self.output = self.layer2_fw_out

        return self.output


    def compute_loss(self, labels_batch):
        training_loss, self.d_output = softmax_loss(self.output, labels_batch)
        reg_loss = 0.5 * self.reg * (np.sum(self.params['l1_weight'] ** 2) + np.sum(self.params['l2_weight'] ** 2))

        return training_loss, reg_loss


    def backward(self):
        grads = {}

        d_layer2, grads['l2_weight'], grads['l2_bias'] = Linear_backward(self.d_output, self.layer2_params)
        # d_layer2_avt = relu_backward(d_layer2, self.layer1_avt_params)
        d_layer2_avt = sigmoid_backword(d_layer2, self.layer1_avt_params)
        d_layer1, grads['l1_weight'], grads['l1_bias'] = Linear_backward(d_layer2_avt, self.layer1_params)

        grads['l2_weight'] += self.reg * self.params['l2_weight']
        grads['l1_weight'] += self.reg * self.params['l1_weight']

        return grads

