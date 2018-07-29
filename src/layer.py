# -*- coding: utf-8 -*-

import numpy as np

def conv2d(inputs, weight, bias, stride=1, padding=0):

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

    batch_size = inputs.shape[0]
    col_num = np.prod(inputs.shape[1:])
    reshaped_inputs = inputs.reshape(batch_size, col_num)
    out = np.dot(reshaped_inputs, weight) + bias

    return out, (inputs, reshaped_inputs, weight, bias)


def Linear_backward(d_out, params):

    inputs, reshaped_inputs, weight, bias = params

    d_weight = np.dot(reshaped_inputs.T, d_out)
    d_inputs = np.dot(d_out, weight.T).reshape(inputs.shape)
    d_bias = np.sum(d_out, axis=0)

    return d_inputs, d_weight, d_bias

