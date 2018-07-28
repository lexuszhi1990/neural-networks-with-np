import numpy as np

from src.activation import *
from src.layer import *
from src.loss import softmax_loss


class alexnet(object):

    def __init__(self, num_cls=10, input_dim=1, reg=0.0,
                 base_channel=32, weight_scale=5e-3):
        self.reg = reg
        self.params = {}
        self.init_params(input_dim, base_channel, num_cls, weight_scale)


    def init_params(self, input_dim, base_channel, num_cls, weight_scale):

        # (3, 28, 28) -> (32, 13, 13)
        self.params['l1_weight'] = np.random.randn(base_channel, input_dim, 4, 4).astype(np.float32) * np.sqrt(2.0/(base_channel*4*4))
        self.params['l1_bias'] = np.zeros(base_channel).astype(np.float32)
        # (32, 13, 13) -> (64, 7, 7)
        self.params['l2_weight'] = np.random.randn(base_channel*2, base_channel, 3, 3).astype(np.float32) * np.sqrt(2.0/(base_channel*2*3*3))
        self.params['l2_bias'] = np.zeros(base_channel*2).astype(np.float32)
        # (64, 7, 7) -> (10,)
        self.params['l3_weight'] = np.random.randn(7*7*(base_channel*2), 256).astype(np.float32) * np.sqrt(2.0/(7*7*(base_channel*2)))
        self.params['l3_bias'] = np.zeros(256).astype(np.float32)
        self.params['l4_weight'] = np.random.randn(256, num_cls).astype(np.float32) * np.sqrt(2.0/(256))
        self.params['l4_bias'] = np.zeros(num_cls).astype(np.float32)

    def forward(self, x):

        self.layer1_fw_out, self.layer1_params = conv2d(x, self.params['l1_weight'], self.params['l1_bias'], 2, 0)
        self.layer1_avt_out, self.layer1_avt_params = relu_forward(self.layer1_fw_out)
        self.layer2_fw_out, self.layer2_params = conv2d(self.layer1_avt_out, self.params['l2_weight'], self.params['l2_bias'], 2, 1)
        self.layer2_avt_out, self.layer2_avt_params = relu_forward(self.layer2_fw_out)
        self.layer3_fw_out, self.layer3_params = Linear(self.layer2_avt_out, self.params['l3_weight'], self.params['l3_bias'])
        self.layer3_avt_out, self.layer3_avt_params = relu_forward(self.layer3_fw_out)
        self.layer4_fw_out, self.layer4_params = Linear(self.layer3_avt_out, self.params['l4_weight'], self.params['l4_bias'])

        return self.layer4_fw_out


    def compute_loss(self, labels_batch):
        loss, self.d_out = softmax_loss(self.layer4_fw_out, labels_batch)
        print("pure loss is %.4f" % loss)
        loss += 0.5 * self.reg * (np.sum(self.params['l1_weight'] ** 2) + np.sum(self.params['l2_weight'] ** 2) + np.sum(self.params['l3_weight'] ** 2))
        print("total loss is %.4f" % loss)

        return loss


    def backward(self):
        grads = {}

        d_layer4, grads['l4_weight'], grads['l4_bias'] = Linear_backward(self.d_out, self.layer4_params)
        d_layer2_avt = relu_backward(d_layer4, self.layer3_avt_params)
        d_layer3, grads['l3_weight'], grads['l3_bias'] = Linear_backward(d_layer2_avt, self.layer3_params)
        d_layer2_avt = relu_backward(d_layer3, self.layer2_avt_params)
        d_layer2, grads['l2_weight'], grads['l2_bias'] = conv2d_backward(d_layer2_avt, self.layer2_params)
        d_layer2_avt = relu_backward(d_layer2, self.layer1_avt_params)
        d_layer1, grads['l1_weight'], grads['l1_bias'] = conv2d_backward(d_layer2_avt, self.layer1_params)

        grads['l4_weight'] += self.reg * self.params['l4_weight']
        grads['l3_weight'] += self.reg * self.params['l3_weight']
        grads['l2_weight'] += self.reg * self.params['l2_weight']
        grads['l1_weight'] += self.reg * self.params['l1_weight']

        print("l1_weight: %.4f" % np.sum(grads['l1_weight']))
        # print(grads)

        aa = self.params['l1_weight']
        for key, value in self.params.items():
            self.params[key] = value - 5e-3 * grads[key]
        bb = self.params['l1_weight']
        print(np.sum(aa-bb))
