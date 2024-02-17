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
Take the standard Transformer as T2T Transformer
"""
import mindspore.nn as nn

from .drop_path_timm import DropPath
from .mlp import Mlp
from .attention import Attention


class TokenTransformer(nn.Cell):
    def __init__(
            self,
            dim,
            in_dim,
            num_heads,
            mlp_ratio=1.,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU(),
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim,
            in_dim=in_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_skip_connection=True,
            scale_qk=True,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )
        self.norm2 = norm_layer((in_dim,))
        self.mlp = Mlp(
            in_features=in_dim,
            hidden_features=int(in_dim*mlp_ratio),
            out_features=in_dim,
            act_layer=act_layer,
            drop=drop
        )

    def construct(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
