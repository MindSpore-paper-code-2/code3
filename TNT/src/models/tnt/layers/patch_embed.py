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
"""Patch embedding layer (image to visual features)"""

import math

import mindspore.nn as nn
import mindspore.ops.operations as P

from .misc import to_2tuple
from .unfold_kernel import UnfoldKernelEqPatch


class PatchEmbed(nn.Cell):
    """
    Image to Visual Word Embedding

    Args:
        img_size(int): Image size (side, px)
        patch_size(int): Output patch size (side, px)
        in_chans(int): Number of input channels
        outer_dim(int): Number of output features (not used)
        inner_dim(int): Number of internal features
        inner_stride(int): Stride of patches (px)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 outer_dim=768, inner_dim=24, inner_stride=4):
        super().__init__()
        _ = outer_dim
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)

        self.unfold = UnfoldKernelEqPatch(kernel_size=patch_size, strides=patch_size)
        # unfold_shape = [1, *patch_size, 1]
        # self.unfold = nn.Unfold(unfold_shape, unfold_shape, unfold_shape)
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=inner_dim, kernel_size=7, stride=inner_stride,
                              pad_mode='pad', padding=3, has_bias=True)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        b, _ = x.shape[0], x.shape[1]
        x = self.unfold(x)  # B, Ck2, N
        x = self.proj(x)  # B*N, C, 8, 8
        x = self.reshape(x, (b * self.num_patches, self.inner_dim, -1,))  # B*N, 8*8, C
        x = self.transpose(x, (0, 2, 1))
        return x
