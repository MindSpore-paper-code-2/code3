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
"""
Unfolding layer (ravel dense patches to linear order).
"""
import math
from typing import Union, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as P


def compute_l(kernel_size, stride, padding, spacial_size):
    r"""
    Compute L according to
    L = \prod_{d}^{}\left\lfloor \frac{spatial\_size[d] +
    2 \times padding[d] - dilation[d] \times (kernel\_size[d] - 1) - 1}
    {stride[d]} + 1 \right\rfloor
    """
    l_dim1 = spacial_size[0] + 2 * padding[0] - kernel_size[0]
    l_dim1 = l_dim1 / stride[0] + 1
    l_dim1 = math.floor(l_dim1)

    l_dim2 = spacial_size[1] + 2 * padding[1] - kernel_size[1]
    l_dim2 = l_dim2 / stride[1] + 1
    l_dim2 = math.floor(l_dim2)
    return l_dim1 * l_dim2


class UnfoldCustom(nn.Cell):
    """Extract sliding local blocks from a batched input tensor."""
    def __init__(
            self,
            kernel_size: Union[Tuple[int, int], int],
            stride: Union[Tuple[int, int], int],
            padding: Union[Tuple[int, int], int],
            in_channels: int = 3,
            image_size: Union[Tuple[int, int], int] = (224, 224),
    ):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert len(kernel_size) == 2, 'Expected len of kernel_size equals 2'
        assert len(padding) == 2, 'Expected len of padding equals 2'
        assert len(stride) == 2, 'Expected len of stride equals 2'
        assert len(image_size) == 2, 'Expected len of image_size equals 2'

        self.kernel_size = kernel_size

        self.padding = P.Pad((
            (0, 0),
            (0, 0),
            (padding[0], padding[0]),
            (padding[1], padding[1])
        ))

        self.unfold = nn.Unfold(
            ksizes=[1, kernel_size[0], kernel_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            rates=[1, 1, 1, 1],
            padding='valid'
        )
        self.reshape = P.Reshape()
        self.concat = P.Concat(1)
        self.in_channels = in_channels
        self.L = compute_l(kernel_size, stride, padding, image_size)
        self.out_c = self.in_channels * kernel_size[0] * kernel_size[1]
        self.out = ms.Parameter(
            ms.Tensor(ms.numpy.zeros((1, self.out_c, self.L))),
            requires_grad=False
        )
        self.tile = ms.ops.Tile()

    def construct(self, inputs):
        inputs = self.padding(inputs)
        B, C, _, _ = inputs.shape
        out = self.tile(self.out, (B, 1, 1))
        unfolded = self.unfold(inputs)
        unfolded = self.reshape(
            unfolded,
            (B, C * self.kernel_size[0] * self.kernel_size[1], -1)
        )
        for i in range(self.in_channels):
            s = i * (self.out_c // self.in_channels)
            e = s + self.out_c // self.in_channels
            out[:, s:e, :] = unfolded[:, i::self.in_channels]
        return out
