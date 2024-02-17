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
"""
Take Performer as T2T Transformer
"""
import math
import mindspore as ms
import mindspore.nn as nn

from mindspore import ops
from mindspore.common.initializer import initializer, Orthogonal

from .mlp import Mlp


class TokenPerformer(nn.Cell):
    def __init__(
            self,
            dim,
            in_dim,
            head_cnt=1,
            kernel_ratio=0.5,
            dp1=0.1,
            dp2=0.1,
            act_layer=nn.GELU(),
    ):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Dense(dim, 3 * self.emb)
        self.dp = nn.Dropout(keep_prob=1. - dp1)
        self.proj = nn.Dense(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm((dim,))
        self.norm2 = nn.LayerNorm((self.emb,))
        self.split = ops.Split(-1, 3)
        self.einsum1 = ops.Einsum('bti,mi->btm')
        self.einsum2 = ops.Einsum('bti,bi->bt')
        self.einsum3 = ops.Einsum('bin,bim->bnm')
        self.einsum4 = ops.Einsum('bti,bni->btn')
        self.epsilon = 1e-8  # for stable in division

        self.mlp = Mlp(self.emb, self.emb, self.emb, act_layer, dp2)

        self.m = int(self.emb * kernel_ratio)
        self.m_t = ms.Tensor(self.emb * kernel_ratio)
        self.w = ms.Parameter(initializer(Orthogonal(), (self.m, self.emb))
                              * math.sqrt(self.m),
                              requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from
        # https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ops.tile((x * x).sum(-1, keepdims=True), (1, 1, self.m)) / 2
        wtx = self.einsum1((x, self.w))

        return (wtx - xd).exp() / ops.sqrt(self.m_t)

    def single_attn(self, x):
        k, q, v = self.split(self.kqv(x))
        # (B, T, m), (B, T, m)
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        # (B, T, m) * (B, m) -> (B, T, 1)
        D = self.einsum2((qp, kp.sum(1))).expand_dims(2)
        kptv = self.einsum3((v, kp))  # (B, emb, m)
        # (B, T, emb)/Diag
        y = self.einsum4((qp, kptv)) / (ops.tile(D, (1, 1, self.emb))
                                        + self.epsilon)
        # skip connection
        # same as token_transformer in T2T layer, use v as skip connection
        y = v + self.dp(self.proj(y))

        return y

    def construct(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
