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
# This file has been derived from the
# https://github.com/yitu-opensource/T2T-ViT
# repository and modified.
# ============================================================================
"""Attention layer of TransformerBlock."""
import mindspore.nn as nn
from mindspore import ops


class Attention(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads=8,
            in_dim=None,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            use_skip_connection=False,
            scale_qk=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_skip_connection = use_skip_connection
        self.scale_qk = scale_qk

        out_dim = in_dim or dim
        self.qkv = nn.Dense(dim, out_dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(out_dim, out_dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        last_dim = self.in_dim or C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, last_dim)
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.scale_qk:
            attn = ops.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        else:
            attn = ops.matmul(q * self.scale, k.transpose(0, 1, 3, 2))
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_C = self.in_dim or C
        x = ops.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, x_C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.use_skip_connection:
            # skip connection
            # because the original x has different size with current x,
            # use v to do skip connection
            x = v.squeeze(1) + x

        return x
