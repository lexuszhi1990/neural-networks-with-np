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


def relu_forward(inputs):
    return np.maximum(0, inputs), inputs


def relu_backward(d_out, params):
    inputs = params
    outputs = np.maximum(0, inputs)
    outputs[d_out > 0] = 1
    d_out *= outputs

    return d_out


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
    batch_size, in_channels, in_height, in_weight = inputs.shape
    col_num = in_channels * in_height * in_weight
    reshaped_inputs = inputs.reshape(batch_size, col_num)
    out = np.dot(reshaped_inputs, weight) + bias

    return out, (inputs, weight, bias)


def Linear_backward(d_out, params):
    """
    computes the backward pass for a fc layer.

    Inputs:
    - d_out: Upstream derivative, of shape (N, M)
    - params: Tuple of:

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    inputs, weight, bias = params
    batch_size = inputs.shape[0]
    reshaped_inputs = inputs.reshape(batch_size, np.prod(inputs.shape[1:]))

    d_inputs = np.dot(d_out, weight.T).reshape(inputs.shape)
    d_weight = np.dot(reshaped_inputs.T, d_out)
    d_bias = np.sum(d_out, axis=0)

    return d_inputs, d_weight, d_bias
