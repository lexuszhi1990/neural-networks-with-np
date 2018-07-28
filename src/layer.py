# -*- coding: utf-8 -*-

import numpy as np

# def conv2d(x, w, b, conv_param):
def conv2d(inputs, weight, bias, stride=1, padding=0):
    """
    pytorch-like convolution2D (https://pytorch.org/docs/0.4.0/nn.html#torch.nn.Conv2d):

    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

    Input:
    - inputs: (N, C, H, W)
    - weight: (F, C, HH, WW)
    - bias: Biases, of shape (F,)
    - stride: controls the stride for the cross-correlation, a single number or a tuple
    - padding: The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * padding - HH) / stride
        W' = 1 + (W + 2 * padding - WW) / stride
    """

    batch_size, in_channels, in_height, in_weight = inputs.shape
    out_channels, in_channels_, kernel_height, kernel_width = weight.shape
    assert in_channels_ == in_channels

    out_height = int((in_height + 2 * padding - kernel_height) / stride + 1)
    out_width = int((in_weight + 2 * padding - kernel_width) / stride + 1)
    out = np.zeros((batch_size, out_channels, out_height, out_width))
    # 右下角padding
    padded_inputs = np.pad(inputs, ((0,), (0,), (padding,), (padding,)), mode='constant', constant_values=0)

    for i in range(out_height):
        for j in range(out_width):
            input_patch = padded_inputs[:, :, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
            for k in range(out_channels):
                out[:, k , i, j] = np.sum(input_patch * weight[k, :, :, :], axis=(1,2,3))

    out = out + (bias)[None, :, None, None]
    return out, (inputs, padded_inputs, weight, bias, stride, padding)


def conv2d_backward(d_out, params):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - params: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    inputs, padded_inputs, weight, bias, stride, padding = params
    batch_size, feat_channels, out_height, out_width = d_out.shape
    _, _, kernel_height, kernel_width = weight.shape

    d_padded_inputs = np.zeros_like(padded_inputs)
    d_inputs = np.zeros_like(inputs)
    d_weight = np.zeros_like(weight)
    d_bias = np.zeros_like(bias)

    for i in range(out_height):
        for j in range(out_width):
            for k in range(feat_channels):
                input_patch = padded_inputs[:, :, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                d_weight[k, :, :, :] += np.sum(input_patch * (d_out[:, k, i, j]).reshape(batch_size, 1, 1, 1), axis=0)
            for k in range(batch_size):
                d_padded_inputs[k, :, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width] += \
                        np.sum(weight[:, :, :, :] * d_out[k, :, i, j].reshape(feat_channels, 1, 1, 1), axis=0)

    d_bias = np.sum(d_out, axis = (0,2,3))
    d_inputs = d_padded_inputs[:,:,padding:-padding,padding:-padding]

    return d_inputs, d_weight, d_bias


def Linear(inputs, weight, bias):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    batch_size = inputs.shape[0]
    col_num = np.prod(inputs.shape[1:])
    reshaped_inputs = inputs.reshape(batch_size, col_num)
    out = np.dot(reshaped_inputs, weight) + bias

    return out, (inputs, reshaped_inputs, weight, bias)


def Linear_backward(d_out, params):
    """
    computes the backward pass for a fc layer.

    Inputs:
    - d_out: Upstream derivative, of shape (batch_size, num_cls)
    - params: Tuple of:

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    inputs, reshaped_inputs, weight, bias = params

    d_weight = np.dot(reshaped_inputs.T, d_out)
    d_inputs = np.dot(d_out, weight.T).reshape(inputs.shape)
    d_bias = np.sum(d_out, axis=0)

    return d_inputs, d_weight, d_bias


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x , axis = 0)
        x_hat = (x - sample_mean) / (np.sqrt(sample_var  + eps))
        out = gamma * x_hat + beta
        cache = (gamma, x, sample_mean, sample_var, eps, x_hat)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

    elif mode == 'test':
        scale = gamma / (np.sqrt(running_var  + eps))
        out = x * scale + (beta - running_mean * scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


# def batchnorm_backward(dout, cache):

#   dxhat = dout * gamma
#   dvar = np.sum(dxhat, axis=0) * (x - batch_mean) * (-0.5 * np.power(batch_var + eps, -1.5))
#   dxm1 = dxhat / (batch_var + eps)
#   dxm2 = np.ones((N,D)) * dvar / N * (2 * (x - batch_mean))
#   dmu = -1 * np.sum(dxm1 + dxm2, axis=0)
#   dx1 = dxm1 + dxm2
#   dx2 = np.ones((N,D)) * dmu / N

#   dx = dx1 + dx2
#   dgamma = np.sum(np.multiply(dout, batch_norm_x), axis=0)
#   dbeta = np.sum(dout, axis=0)

#   #############################################################################
#   #                             END OF YOUR CODE                              #
#   #############################################################################

#   return dx, dgamma, dbeta

