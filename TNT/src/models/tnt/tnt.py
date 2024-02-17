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
"""Transformer in Transformer (TNT)"""
from dataclasses import dataclass

import numpy as np
import mindspore.common.initializer as weight_init
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype

from .layers.misc import DropPath1D, trunc_array
from .layers.patch_embed import PatchEmbed
from .layers.attention import Attention


def make_divisible(v, divisor=8, min_value=None):
    """
    Round number to the multiple of divisor
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Mlp(nn.Cell):
    """
    Multi-layer perceptron

    Args:
        in_features(int): Number of input features
        hidden_features(int): Number of hidden features
        out_features(int): Number of output features
        act_layer(class): Activation layer (base class)
        drop(float): Dropout rate
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop) # if drop > 0. else Identity()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Cell):
    """SE Block"""

    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.SequentialCell([
            nn.LayerNorm(normalized_shape=dim, epsilon=1e-5),
            nn.Dense(in_channels=dim, out_channels=hidden_dim),
            nn.ReLU(),
            nn.Dense(in_channels=hidden_dim, out_channels=dim),
            nn.Tanh()
        ])

        self.reduce_mean = P.ReduceMean()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        a = self.reduce_mean(True, x, 1)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Block(nn.Cell):
    """
    TNT base block

    Args:
        outer_dim(int): Number of output features
        inner_dim(int): Number of internal features
        outer_num_heads(int): Number of output heads
        inner_num_heads(int): Number of internal heads
        num_words(int): Number of 'visual words' (feature groups)
        mlp_ratio(float): Rate of MLP per hidden features
        qkv_bias(bool): Use Qk / v bias
        qk_scale(float): Qk scale
        drop(float): Dropout rate
        attn_drop(float): Dropout rate of attention layer
        drop_path(float): Path dropout rate
        act_layer(class): Activation layer (class)
        norm_layer(class): Normalization layer
        se(int): SE parameter
    """

    def __init__(self, outer_dim, inner_dim, outer_num_heads,
                 inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer((inner_dim,), epsilon=1e-5)
            self.inner_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer((inner_dim,), epsilon=1e-5)
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer((num_words * inner_dim,), epsilon=1e-5)
            self.proj = nn.Dense(in_channels=num_words * inner_dim, out_channels=outer_dim, has_bias=False)
            self.proj_norm2 = norm_layer((outer_dim,), epsilon=1e-5)
        # Outer
        self.outer_norm1 = norm_layer((outer_dim,), epsilon=1e-5)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath1D(drop_path)
        self.outer_norm2 = norm_layer((outer_dim,), epsilon=1e-5)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = 0
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)
        self.zeros = Tensor(np.zeros([1, 1, 1]), dtype=mstype.float32)

        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, *inputs, **kwargs):
        """TNT Block construct"""

        inner_tokens, outer_tokens = inputs[0], inputs[1]
        if self.has_inner:
            in1 = self.inner_norm1(inner_tokens)
            attn1 = self.inner_attn(in1)
            inner_tokens = inner_tokens + self.drop_path(attn1)  # B*N, k*k, c
            in2 = self.inner_norm2(inner_tokens)
            mlp = self.inner_mlp(in2)
            inner_tokens = inner_tokens + self.drop_path(mlp)  # B*N, k*k, c
            b, n, _ = P.Shape()(outer_tokens)
            # zeros = P.Tile()(self.zeros, (B, 1, C))
            proj = self.proj_norm2(self.proj(self.proj_norm1(
                self.reshape(inner_tokens, (b, n - 1, -1,))
            )))
            proj = self.cast(proj, mstype.float32)
            # proj = P.Concat(1)((zeros, proj))
            # outer_tokens = outer_tokens + proj  # B, N, C
            outer_tokens[:, 1:] = outer_tokens[:, 1:] + proj
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(
                self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(
                tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(
                self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(
                self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class TNT(nn.Cell):
    """
    TNT (Transformer in Transformer) for computer vision

    Args:
        img_size(int): Image size (side, px)
        patch_size(int): Patch size (side, px)
        in_chans(int): Number of input channels
        num_classes(int): Number of output classes
        outer_dim(int): Number of output features
        inner_dim(int): Number of internal features
        depth(int): Number of TNT base blocks
        outer_num_heads(int): Number of output heads
        inner_num_heads(int): Number of internal heads
        mlp_ratio(float): Rate of MLP per hidden features
        qkv_bias(bool): Use Qk / v bias
        qk_scale(float): Qk scale
        drop_rate(float): Dropout rate
        attn_drop_rate(float): Dropout rate for attention layer
        drop_path_rate(float): Dropout rate for DropPath layer
        norm_layer(class): Normalization layer
        inner_stride(int): Number of strides for internal patches
        se(int): SE parameter
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, outer_dim=768, inner_dim=48,
                 depth=12, outer_num_heads=12, inner_num_heads=4,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 # drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 # norm_layer=LayerNormFixOrder, inner_stride=4, se=0,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, inner_stride=4, se=0,
                 **kwargs):
        super().__init__()
        _ = kwargs
        self.num_classes = num_classes
        self.outer_dim = outer_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim,
            inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words

        self.proj_norm1 = norm_layer((num_words * inner_dim,), epsilon=1e-5)
        self.proj = nn.Dense(in_channels=num_words * inner_dim, out_channels=outer_dim, has_bias=True)
        self.proj_norm2 = norm_layer((outer_dim,), epsilon=1e-5)

        self.cls_token = Parameter(Tensor(trunc_array([1, 1, outer_dim]), dtype=mstype.float32), name="cls_token",
                                   requires_grad=True)
        self.outer_pos = Parameter(Tensor(trunc_array([1, num_patches + 1, outer_dim]), dtype=mstype.float32),
                                   name="outer_pos")
        self.inner_pos = Parameter(Tensor(trunc_array([1, num_words, inner_dim]), dtype=mstype.float32))
        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                    inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.CellList(blocks)
        # self.norm = norm_layer(outer_dim, eps=1e-5)
        self.norm = norm_layer((outer_dim,))

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(outer_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        mask = np.zeros([1, num_patches + 1, 1])
        mask[:, 0] = 1
        self.mask = Tensor(mask, dtype=mstype.float32)
        self.head = nn.Dense(in_channels=outer_dim, out_channels=num_classes, has_bias=True)

        self.reshape = P.Reshape()
        self.concat = P.Concat(1)
        self.tile = P.Tile()
        self.cast = P.Cast()

        self.init_weights()
        print("================================success================================")

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):
        """TNT forward_features"""
        b = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C

        outer_tokens = self.proj_norm2(
            self.proj(self.proj_norm1(
                self.reshape(inner_tokens, (b, self.num_patches, -1,))
            ))
        )
        outer_tokens = self.cast(outer_tokens, mstype.float32)
        outer_tokens = self.concat((
            self.tile(self.cls_token, (b, 1, 1)), outer_tokens
        ))

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)  # [batch_size, num_patch+1, outer_dim)
        return outer_tokens[:, 0]

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.forward_features(x)
        x = self.head(x)
        return x


def tnt_s_patch16_224(args):
    """tnt_s_patch16_224"""

    patch_size = 16
    inner_stride = 4
    outer_dim = 384
    inner_dim = 24
    outer_num_heads = 6
    inner_num_heads = 4
    depth = 12
    drop_path_rate = args.drop_path_rate
    drop_out = args.drop_out
    num_classes = args.num_classes
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=depth,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, drop_path_rate=drop_path_rate, drop_out=drop_out, num_classes=num_classes)
    return model


def tnt_b_patch16_224(args):
    """tnt_b_patch16_224"""

    patch_size = 16
    inner_stride = 4
    outer_dim = 640
    inner_dim = 40
    outer_num_heads = 10
    inner_num_heads = 4
    depth = 12
    drop_path_rate = args.drop_path_rate
    drop_out = args.drop_out
    num_classes = args.num_classes
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=depth,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, drop_path_rate=drop_path_rate, drop_out=drop_out, num_classes=num_classes)
    return model


@dataclass
class NetworkParams:
    num_classes: int
    drop_path_rate: float
    drop_out: float


def get_model_by_name(arch, num_classes, drop_path_rate, drop_out,
                      **kwargs) -> TNT:
    """get network by name and initialize it"""
    _ = kwargs
    models = {
        'tnt_s_patch16_224': tnt_s_patch16_224,
        'tnt_b_patch16_224': tnt_b_patch16_224
    }
    args = NetworkParams(num_classes, drop_path_rate, drop_out)
    return models[arch](args)
