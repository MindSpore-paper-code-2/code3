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
import mindspore as ms
import mindspore.nn as nn
from mindspore.common import initializer as init

from .layers.t2t_module import T2T_module
from .layers.transformer_block import Block
from .utils import get_sinusoid_encoding


class T2T_ViT(nn.Cell):
    def __init__(
            self,
            img_size=224,
            tokens_type='performer',
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            token_dim=64,
            approximate_gelu=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.tokens_to_token = T2T_module(
            img_size=img_size,
            tokens_type=tokens_type,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim,
            act_layer=nn.GELU(approximate_gelu)
        )
        num_patches = self.tokens_to_token.num_patches

        self.pos_embed = ms.Parameter(
            get_sinusoid_encoding(
                n_position=num_patches + 1, d_hid=embed_dim
            ),
            requires_grad=False
        )
        self.pos_drop = nn.Dropout(keep_prob=1 - drop_rate)

        # stochastic depth decay rule
        linspace = ms.ops.LinSpace()
        dpr = [
            x for x in linspace(
                ms.Tensor(0, ms.float32),
                ms.Tensor(drop_path_rate, ms.float32),
                depth
            )
        ]
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.GELU(approximate_gelu),
            )
            for i in range(depth)])
        self.norm = norm_layer((embed_dim,))

        # Classifier head
        self.head = (
            nn.Dense(embed_dim, num_classes)
            if num_classes > 0 else nn.Identity()
        )

        self.cls_token = ms.Parameter(
            init.initializer(
                init.TruncatedNormal(0.02),
                (1, 1, embed_dim),
                ms.float32
            )
        )
        self.tile = ms.ops.Tile()
        self.print = ms.ops.Print()

    def _init_weights(self):
        for _, m in self.parameters_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(
                    init.initializer(
                        init.TruncatedNormal(sigma=0.02),
                        m.weight.shape,
                        m.weight.dtype
                    )
                )
                if isinstance(m, nn.Dense) and m.bias is not None:
                    m.bias.set_data(
                        init.initializer(
                            init.Constant(0),
                            m.bias.shape,
                            m.bias.dtype
                        )
                    )
            elif isinstance(m, nn.LayerNorm):
                m.gamma.set_data(
                    init.initializer(
                        init.Constant(1),
                        m.gamma.shape,
                        m.gamma.dtype
                    )
                )
                m.beta.set_data(
                    init.initializer(
                        init.Constant(0),
                        m.beta.shape,
                        m.beta.dtype
                    )
                )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Dense(self.embed_dim, num_classes)
            if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.tile(self.cls_token, (B, 1, 1))
        x = ms.ops.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def t2t_vit_7(**kwargs):
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7,
                    num_heads=4, mlp_ratio=2., **kwargs)
    return model


def t2t_vit_10(**kwargs):
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10,
                    num_heads=4, mlp_ratio=2., **kwargs)
    return model


def t2t_vit_12(**kwargs):
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12,
                    num_heads=4, mlp_ratio=2., **kwargs)
    return model


def t2t_vit_14(**kwargs):
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14,
                    num_heads=6, mlp_ratio=3., **kwargs)
    return model


def t2t_vit_19(**kwargs):
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19,
                    num_heads=7, mlp_ratio=3., **kwargs)
    return model


def t2t_vit_24(**kwargs):
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24,
                    num_heads=8, mlp_ratio=3., **kwargs)
    return model


# adopt transformers for tokens to token
def t2t_vit_t_14(**kwargs):
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14,
                    num_heads=6, mlp_ratio=3., **kwargs)
    return model


# adopt transformers for tokens to token
def t2t_vit_t_19(**kwargs):
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19,
                    num_heads=7, mlp_ratio=3., **kwargs)
    return model


# adopt transformers for tokens to token
def t2t_vit_t_24(**kwargs):
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24,
                    num_heads=8, mlp_ratio=3., **kwargs)
    return model
