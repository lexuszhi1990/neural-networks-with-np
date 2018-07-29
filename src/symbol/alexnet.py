import numpy as np

from src.activation import *
from src.layer import *
from src.loss import softmax_loss


class alexnet(object):

    def __init__(self, num_cls=10, input_dim=1, reg=1e-3,
                 base_channel=32):
        self.name = 'alexnet'
        self.reg = reg
        self.params = {}
        self.init_params(input_dim, num_cls, base_channel)


    def init_params(self, input_dim, num_cls, base_channel):
        # (3, 28, 28) -> (32, 13, 13)
        self.params['l1_weight'] = np.random.randn(base_channel, input_dim, 4, 4).astype(np.float32) * np.sqrt(2.0/(base_channel*4*4))
        self.params['l1_bias'] = np.zeros(base_channel).astype(np.float32)

        # (32, 13, 13) -> (64, 7, 7)
        self.params['l2_weight'] = np.random.randn(base_channel*2, base_channel, 3, 3).astype(np.float32) * np.sqrt(2.0/(base_channel*2*3*3))
        self.params['l2_bias'] = np.zeros(base_channel*2).astype(np.float32)

        # (64, 7, 7) -> (128, 4, 4)
        self.params['l3_weight'] = np.random.randn(base_channel*4, base_channel*2, 3, 3).astype(np.float32) * np.sqrt(2.0/(base_channel*4*3*3))
        self.params['l3_bias'] = np.zeros(base_channel*4).astype(np.float32)

        # (128, 4, 4) -> (1024,)
        self.params['l4_weight'] = np.random.randn(4*4*(base_channel*4), 1024).astype(np.float32) * np.sqrt(2.0/(4*4*(base_channel*4)))
        self.params['l4_bias'] = np.zeros(1024).astype(np.float32)

        # (1024, 1) -> (10, 1)
        self.params['l5_weight'] = np.random.randn(1024, num_cls).astype(np.float32) * np.sqrt(2.0/1024)
        self.params['l5_bias'] = np.zeros(num_cls).astype(np.float32)


    def forward(self, x):

        x, self.layer1_params = conv2d(x, self.params['l1_weight'], self.params['l1_bias'], 2, 0)
        x, self.layer1_avt_params = relu_forward(x)
        x, self.layer2_params = conv2d(x, self.params['l2_weight'], self.params['l2_bias'], 2, 1)
        x, self.layer2_avt_params = relu_forward(x)
        x, self.layer3_params = conv2d(x, self.params['l3_weight'], self.params['l3_bias'], 2, 1)
        x, self.layer3_avt_params = relu_forward(x)
        x, self.layer4_params = Linear(x, self.params['l4_weight'], self.params['l4_bias'])
        x, self.layer5_params = Linear(x, self.params['l5_weight'], self.params['l5_bias'])

        self.output = x

        return self.output


    def compute_loss(self, labels_batch):
        train_loss, self.d_output = softmax_loss(self.output, labels_batch)
        weights = sum([np.sum(self.params['l%d_weight'%i] ** 2) for i in range(1, 6)])
        reg_loss = 0.5 * self.reg * weights

        return train_loss, reg_loss


    def backward(self):
        grads = {}

        d_layer5, grads['l5_weight'], grads['l5_bias'] = Linear_backward(self.d_output, self.layer5_params)
        d_layer4, grads['l4_weight'], grads['l4_bias'] = Linear_backward(d_layer5, self.layer4_params)

        d_layer4_avt = relu_backward(d_layer4, self.layer3_avt_params)
        d_layer3, grads['l3_weight'], grads['l3_bias'] = conv2d_backward(d_layer4_avt, self.layer3_params)

        d_layer3_avt = relu_backward(d_layer3, self.layer2_avt_params)
        d_layer2, grads['l2_weight'], grads['l2_bias'] = conv2d_backward(d_layer3_avt, self.layer2_params)

        d_layer2_avt = relu_backward(d_layer2, self.layer1_avt_params)
        d_layer1, grads['l1_weight'], grads['l1_bias'] = conv2d_backward(d_layer2_avt, self.layer1_params)

        grads['l5_weight'] += self.reg * self.params['l5_weight']
        grads['l4_weight'] += self.reg * self.params['l4_weight']
        grads['l3_weight'] += self.reg * self.params['l3_weight']
        grads['l2_weight'] += self.reg * self.params['l2_weight']
        grads['l1_weight'] += self.reg * self.params['l1_weight']

        return grads

