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
"""Attention/projection layer"""

import mindspore.nn as nn
import mindspore.ops.operations as P


class Attention(nn.Cell):
    """
    Attention layer

    Args:
        dim(int): Number of output features
        hidden_dim(int): Number of hidden features
        num_heads(int): Number of output heads
        qkv_bias(bool): Enable bias weights in Qk / v dense layers
        qk_scale(float): Qk scale (multiplier)
        attn_drop(float): Attention dropout rate
        proj_drop(float): Projection dropout rate
    """

    def __init__(self, dim, hidden_dim,
                 num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Dense(in_channels=dim, out_channels=hidden_dim * 2, has_bias=qkv_bias)
        # self.q = nn.Dense(in_channels=dim, out_channels=hidden_dim, has_bias=qkv_bias)
        # self.k = nn.Dense(in_channels=dim, out_channels=hidden_dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = P.BatchMatMul()

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, *inputs, **kwargs):
        """Attention construct"""
        x = inputs[0]
        b, n, _ = x.shape
        qk = self.reshape(self.qk(x),
                          (b, n, 2, self.num_heads, self.head_dim))
        qk = self.transpose(qk, (2, 0, 3, 1, 4))
        q, k = qk[0], qk[1]

        v = self.reshape(self.v(x),
                         (b, n, self.num_heads, -1))
        v = self.transpose(v, (0, 2, 1, 3))

        attn = self.matmul(q, self.transpose(k, (0, 1, 3, 2))
                           ) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.transpose(self.matmul(attn, v), (0, 2, 1, 3))
        x = self.reshape(x, (b, n, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
