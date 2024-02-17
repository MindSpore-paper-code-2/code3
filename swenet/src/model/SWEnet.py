# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""SWEnet."""


import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore.ops import operations as P
from mindspore import Parameter
import mindspore.numpy as msnp
from src.model.se import SELayer


def init_weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


class SeResidualBlock(nn.Cell):

    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 dilation=1,

                 ):
        super(SeResidualBlock, self).__init__()
        # self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = conv1x1_layer(in_channel, channel, stride=1, dilation=dilation)
        self.bn1 = bn_layer(channel)
        self.conv2 = conv3x3_layer(channel, channel, stride=stride, dilation=dilation)
        self.bn2 = bn_layer(channel)

        self.conv3 = conv1x1_layer(channel, out_channel, stride=1, dilation=dilation)
        self.bn3 = bn_last_layer(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 0 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([conv5x5_layer(in_channel, out_channel, stride, 0,
                                                                      "same", dilation), bn_layer(out_channel)])
        self.add = P.Add()
        self.se = SELayer(out_channel)

    def construct(self, x):
        """se_block"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class SWE(nn.Cell):

    def __init__(self, kernel_size, window='box'):
        super(SWE, self).__init__()
        self.kernel_size = kernel_size
        self.radius = kernel_size//2
        self.window = window
        self.zeros = mindspore.ops.Zeros()
        self.ones = mindspore.ops.Ones()
        self.L = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                           requires_grad=False)
        self.R = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                           requires_grad=False)
        self.U = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                           requires_grad=False)
        self.D = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                           requires_grad=False)
        self.NW = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                            requires_grad=False)
        self.NE = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                            requires_grad=False)
        self.SW = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                            requires_grad=False)
        self.SE = Parameter(self.ones((1, 1, self.kernel_size, self.kernel_size), mindspore.float32),
                            requires_grad=False)

        self.window_num = Parameter(Tensor(np.array([[[[self.kernel_size * self.radius]],
                                                      [[self.kernel_size * self.radius]],
                                                      [[self.kernel_size * self.radius]],
                                                      [[self.kernel_size * self.radius]], [[self.radius ** 2]],
                                                      [[self.radius ** 2]], [[self.radius ** 2]],
                                                      [[self.radius ** 2]]]]), mindspore.float32))

        self.div = mindspore.ops.Div()
        self.Mul = mindspore.ops.Mul()

    def construct(self, im):
        b, c, h, w = im.shape
        d = self.zeros((b, 8, h, w), mindspore.float32)
        res = im.copy()

        if self.window == 'box':
            self.L[:, :, :, self.radius:] = 0.0
            self.R[:, :, :, 0: self.radius] = 0.0
            self.U[:, :, self.radius:, :] = 0.0
            self.D[:, :, 0: self.radius, :] = 0.0
            self.NW[:, :, self.radius:, :] = 0.0
            self.NE[:, :, self.radius:, :] = 0.0
            self.SW[:, :, self.radius:, :] = 0.0
            self.SE[:, :, self.radius:, :] = 0.0
            self.NW[:, :, :, self.radius:] = 0.0
            self.NE[:, :, :, 0: self.radius] = 0.0
            self.SW[:, :, :, self.radius:] = 0.0
            self.SE[:, :, :, 0: self.radius] = 0.0

        for ch in range(c):
            im_ch = im[:, ch, ::].copy().view(b, 1, h, w)
            d[:, 0, ::] = self.Mul(im_ch, self.L / (self.radius * self.kernel_size)).squeeze(axis=1)
            d[:, 1, ::] = self.Mul(im_ch, self.R / (self.radius * self.kernel_size)).squeeze(axis=1)
            d[:, 2, ::] = self.Mul(im_ch, self.U / (self.radius * self.kernel_size)).squeeze(axis=1)
            d[:, 3, ::] = self.Mul(im_ch, self.D / (self.radius * self.kernel_size)).squeeze(axis=1)
            d[:, 4, ::] = self.Mul(im_ch, self.NW / (self.radius * self.radius)).squeeze(axis=1)
            d[:, 5, ::] = self.Mul(im_ch, self.NE / (self.radius * self.radius)).squeeze(axis=1)
            d[:, 6, ::] = self.Mul(im_ch, self.SW / (self.radius * self.radius)).squeeze(axis=1)
            d[:, 7, ::] = self.Mul(im_ch, self.SE / (self.radius * self.radius)).squeeze(axis=1)
            d_cha = mindspore.ops.Abs()(d.sum(axis=(2, 3)) - im_ch[:, :, h // 2, w // 2])
            mask_min = mindspore.ops.ArgMinWithValue(axis=1, keep_dims=True)(d_cha)[0]
            mask_index = msnp.arange(b).reshape((b, 1))
            mask = msnp.hstack((mask_index, mask_min))
            d = d*self.window_num
            dm = mindspore.ops.GatherNd()(d, mask)
            res[:, ch, ::] = dm
        return res


def conv5x5_layer(in_channel, out_channel, stride=1, padding=2, pad_mode='pad', dilation=1):
    weight_shape = (out_channel, in_channel, 5, 5)
    weight = init_weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=5, stride=stride, padding=padding, pad_mode=pad_mode, dilation=dilation,
                     weight_init=weight)


def conv3x3_layer(in_channel, out_channel, stride=1, padding=2, pad_mode='pad', dilation=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = init_weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode, dilation=dilation,
                     weight_init=weight)


def conv1x1_layer(in_channel, out_channel, stride=1, padding=0, dilation=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = init_weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=padding, pad_mode='pad', dilation=dilation,
                     weight_init=weight)


def bn_last_layer(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, gamma_init=0,
                          beta_init=0, moving_mean_init=0, moving_var_init=1)


def bn_layer(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, gamma_init=1,
                          beta_init=0, moving_mean_init=0, moving_var_init=1)


def fully_c_layer(input_c, output_c):
    w_shape = (output_c, input_c)
    w = init_weight_variable(w_shape)
    return nn.Dense(input_c, output_c, has_bias=True, weight_init=w, bias_init=0)


class ResNet(nn.Cell):

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv5_5 = conv5x5_layer(3, 10, stride=1, padding=2, pad_mode='pad')
        self.bn1_5 = bn_layer(10)
        self.relu = P.ReLU()

        self.conv3_3 = conv3x3_layer(3, 10, stride=1, padding=1, pad_mode='pad')
        self.bn1_3 = bn_layer(10)

        self.conv1_1 = conv1x1_layer(3, 10, stride=1, padding=0)
        self.bn1_1 = bn_layer(10)

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0], dilation=2)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1], dilation=2)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2], dilation=2)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3], dilation=2)

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.fc = fully_c_layer(out_channels[3] + out_channels[2] + out_channels[1] + out_channels[0], 32)
        self.end_point = fully_c_layer(32, num_classes)
        self.swe = SWE(kernel_size=16)
        self.op = ops.Concat(1)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, dilation):

        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride, dilation=dilation)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1, dilation=dilation)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x1, x2):
        """construct network"""

        x1_1 = self.conv1_1(x1)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)

        x1_3 = self.conv3_3(x1)
        x1_3 = self.bn1_3(x1_3)
        x1_3 = self.relu(x1_3)

        x1_5 = self.conv5_5(x1)
        x1_5 = self.bn1_5(x1_5)
        x1_5 = self.relu(x1_5)

        x2_1 = self.conv1_1(x2)
        x2_1 = self.bn1_1(x2_1)
        x2_1 = self.relu(x2_1)

        x2_3 = self.conv3_3(x2)
        x2_3 = self.bn1_3(x2_3)
        x2_3 = self.relu(x2_3)

        x2_5 = self.conv5_5(x2)
        x2_5 = self.bn1_5(x2_5)
        x2_5 = self.relu(x2_5)

        x1 = self.op((x1_1, x1_3, x1_5))
        x2 = self.op((x2_1, x2_3, x2_5))

        ops_resize = ops.ResizeBilinear((16, 16))
        x1 = ops_resize(x1)
        x2 = ops_resize(x2)

        d = ops.mul(x1, x1) - ops.mul(x2, x2)
        d = ops.mul(d, d)
        d_1 = self.swe(d)
        d += d_1

        x_layer1 = self.layer1(d)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer1 = self.mean(x_layer1, (2, 3))
        x_layer2 = self.mean(x_layer2, (2, 3))
        x_layer3 = self.mean(x_layer3, (2, 3))
        x_layer4 = self.mean(x_layer4, (2, 3))

        x_layer1 = self.flatten(x_layer1)
        x_layer2 = self.flatten(x_layer2)
        x_layer3 = self.flatten(x_layer3)
        x_layer4 = self.flatten(x_layer4)
        x_cat = self.op((x_layer1, x_layer2, x_layer3, x_layer4))
        out = self.fc(x_cat)
        out = self.end_point(out)

        return out


def SWEnet(class_num=10):

    return ResNet(SeResidualBlock,
                  [2, 2, 2, 2],
                  [30, 16, 32, 64],
                  [16, 32, 64, 128],
                  [1, 1, 1, 1],
                  class_num)
