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
"""Unfolding layer (ravel dense patches to linear order)"""

import mindspore.nn as nn
import mindspore.ops.operations as P


class UnfoldKernelEqPatch(nn.Cell):
    """
    UnfoldKernelEqPatch with better performance

    Args:
        kernel_size(tuple): kernel size (along each side)
        strides(tuple): Stride (along each side)

    Returns:
        Tensor, output tensor
    """

    def __init__(self, kernel_size, strides):
        super(UnfoldKernelEqPatch, self).__init__()
        assert kernel_size == strides
        self.kernel_size = kernel_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, *inputs, **kwargs):
        inputs = inputs[0]
        b, c, h, w = inputs.shape
        inputs = self.reshape(inputs,
                              (b, c, h // self.kernel_size[0], self.kernel_size[0], w))
        inputs = self.transpose(inputs, (0, 2, 1, 3, 4))
        inputs = self.reshape(inputs, (-1, c, self.kernel_size[0], w // self.kernel_size[1], self.kernel_size[1]))
        inputs = self.transpose(inputs, (0, 3, 1, 2, 4))
        inputs = self.reshape(inputs, (-1, c, self.kernel_size[0], self.kernel_size[1]))
        # inputs = self.reshape(
        #     inputs,
        #     (B, C,
        #      H // self.kernel_size[0], self.kernel_size[0],
        #      W // self.kernel_size[1], self.kernel_size[1])
        # )
        # inputs = self.transpose(inputs, )

        return inputs
