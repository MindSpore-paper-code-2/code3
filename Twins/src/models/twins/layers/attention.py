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
"""Attention block"""

import mindspore.nn as nn
import mindspore.ops.operations as P


class Attention(nn.Cell):
    """Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 name='', h=0, w=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.kv = nn.Dense(in_channels=dim, out_channels=dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = P.BatchMatMul()

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, has_bias=True)
            self.norm = nn.LayerNorm([dim])
            self.norm.beta.name = '{}.{}'.format(name, self.norm.beta.name)
            self.norm.gamma.name = '{}.{}'.format(name, self.norm.gamma.name)
        self.h, self.w = h, w

    def construct(self, *inputs, **kwargs):
        """Attention construct"""
        x = inputs[0]
        b, n, c = x.shape

        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads
                              ).transpose((0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = x.transpose((0, 2, 1)).reshape(b, c, self.h, self.w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose((0, 2, 1))
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).transpose((2, 0, 3, 1, 4))
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).transpose((2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = self.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.matmul(attn, v).transpose((0, 2, 1, 3)).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupAttention(nn.Cell):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1,
                 name='', h=0, w=0):
        _ = name  # no parameter renaming
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)
        self.ws = ws
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = P.BatchMatMul()
        self.h, self.w = h, w

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        b, n, c = x.shape
        h_group, w_group = self.h // self.ws, self.w // self.ws

        total_groups = h_group * w_group

        x = x.reshape(b, h_group, self.ws, w_group, self.ws, c)
        x = x.transpose((0, 1, 3, 2, 4, 5))

        qkv = self.qkv(x).reshape(b, total_groups, -1, 3, self.num_heads, c // self.num_heads
                                  ).transpose((3, 0, 1, 4, 2, 5))
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = self.matmul(q, k.transpose((0, 1, 2, 4, 3))) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = self.softmax(attn)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = self.matmul(attn, v)
        attn = attn.transpose((0, 1, 3, 2, 4)).reshape(b, h_group, w_group, self.ws, self.ws, c)
        x = attn.transpose((0, 1, 3, 2, 4, 5)).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
