# Copyright 2023 Huawei Technologies Co., Ltd
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
#
# This file has been derived from the https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch
# repository and modified.
# ============================================================================
"""misc functions for tnt"""
import collections.abc
from itertools import repeat

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops
from scipy.stats import truncnorm


def trunc_array(shape, sigma=0.02):
    """output truncnormal array in shape"""
    return truncnorm.rvs(-2, 2, loc=0, scale=sigma, size=shape, random_state=None)


def _ntuple(n):
    """get _ntuple"""

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Identity(nn.Cell):
    """Identity"""

    def construct(self, *inputs, **kwargs):
        return inputs[0]


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).

    Args:
        drop_prob(float): Probability of dropout
        ndim(int): Number of dimensions in input tensor

    Returns:
        Tensor: Output tensor after dropout
    """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath1D(DropPath):
    """DropPath1D"""

    def __init__(self, drop_prob):
        super(DropPath1D, self).__init__(drop_prob=drop_prob, ndim=1)
