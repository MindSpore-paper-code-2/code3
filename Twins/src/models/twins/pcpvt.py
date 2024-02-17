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
# This file has been derived from the https://github.com/Meituan-AutoML/Twins
# repository and modified.
# ============================================================================
"""PCPVT model implementation"""

import math
from functools import partial

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Parameter
import mindspore.ops.functional as F
import mindspore.common.initializer as weight_init

from .layers import Attention, DropPath, PatchEmbed, PosCNN


class Identity(nn.Cell):
    """
    Copy of nn.Identity layer (for ONNX export)
    """

    def construct(self, *inputs, **kwargs):
        return inputs[0]


class Mlp(nn.Cell):
    """2-layer perceptron module"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(1.0 - drop)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Cell):
    """Base PCPVT block class"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 name='', h=0, w=0):
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.norm1.beta.name = '{}1.{}'.format(name, self.norm1.beta)
        self.norm1.gamma.name = '{}1.{}'.format(name, self.norm1.gamma)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
            name=name + '_attn', h=h, w=w
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer([dim])
        self.norm2.beta.name = '{}2.{}'.format(name, self.norm2.beta)
        self.norm2.gamma.name = '{}2.{}'.format(name, self.norm2.gamma)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.h, self.w = h, w

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PyramidVisionTransformer(nn.Cell):
    """PVT base model architecture from https://github.com/whai362/PVT.git"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = []
        self.pos_embeds = []
        self.pos_drops = []
        self.blocks = []

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i], name=i))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i],
                               name=i))
            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) - 1 else self.patch_embeds[
                -1].num_patches
            self.pos_embeds.append(Parameter(
                weight_init.initializer(weight_init.Zero(),
                                        (1, patch_num, embed_dims[i]),
                                        ms.dtype.float32)
            ))
            self.pos_drops.append(nn.Dropout(1.0 - drop_rate))
        self.patch_embeds = nn.CellList(self.patch_embeds)

        dpr = np.linspace(0, drop_path_rate, sum(depths)
                          )  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            block = [block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k],
                name='b{}.{}'.format(k, i), h=self.patch_embeds[k].h, w=self.patch_embeds[k].w
            ) for i in range(depths[k])]
            self.blocks.extend(block)
            cur += depths[k]

        self.norm = norm_layer([embed_dims[-1]])
        self.cls_token = Parameter(
            weight_init.initializer(weight_init.Zero(),
                                    (1, 1, embed_dims[-1]),
                                    ms.dtype.float32)
        )

        # classification head
        self.head = nn.Dense(embed_dims[-1], num_classes) if num_classes > 0 else Identity()

        # init weights
        for pos_emb in self.pos_embeds:
            # trunc_normal_(pos_emb, std=.02)
            pos_emb.set_data(weight_init.initializer(
                weight_init.TruncatedNormal(sigma=0.02),
                pos_emb.shape,
                pos_emb.dtype
            ))

    def reset_drop_path(self, drop_path_rate):
        dpr = np.linspace(0, drop_path_rate, sum(self.depths))
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        _ = global_pool
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        """Base feature processing method"""
        b = x.shape[0]
        for i in range(len(self.depths)):
            x = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = F.concat((cls_tokens, x), axis=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x)
            if i < len(self.depths) - 1:
                x = x.reshape(b, self.patch_embeds[i].h, self.patch_embeds[i].w, -1
                              ).transpose((0, 3, 1, 2)).contiguous()

        x = self.norm(x)

        return x[:, 0]

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.forward_features(x)
        x = self.head(x)

        return x


class CPVTV2(PyramidVisionTransformer):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), block_cls=Block):
        super(CPVTV2, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios,
                                     qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, depths,
                                     sr_ratios, block_cls)
        del self.pos_embeds
        del self.cls_token
        self.pos_block = nn.CellList([
            PosCNN(embed_dims[k], embed_dims[k], name=k,
                   h=self.patch_embeds[k].h, w=self.patch_embeds[k].w)
            for k, embed_dim in enumerate(embed_dims)
        ])
        self.merge_blocks = nn.CellList()
        total = 0
        self.inds = []
        for k, d in enumerate(self.depths):
            self.merge_blocks.append(nn.SequentialCell([
                self.blocks[total],
                self.pos_block[k]
            ] + self.blocks[total + 1:total + d]))
            self.inds.append([total, -1 - k] + list(range(total + 1, total + d)))
            total += d
        # self.apply(self._init_weights)

    def _init_weights(self):
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
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.groups
                cell.gamma.set_data(weight_init.initializer(weight_init.Normal(0.0, math.sqrt(2.0 / fan_out)),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                if isinstance(cell, nn.Conv2d) and cell.bias is not None:
                    cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.beta.shape,
                                                               cell.beta.dtype))

    def forward_features(self, x):
        b = x.shape[0]

        for i in range(len(self.depths)):
            # x, (H, W) = self.patch_embeds[i](x)
            x = self.patch_embeds[i](x)
            h, w = self.patch_embeds[i].w, self.patch_embeds[i].w
            x = self.pos_drops[i](x)
            x = self.merge_blocks[i](x)
            if i < len(self.depths) - 1:
                x = x.reshape(b, h, w, -1).transpose((0, 3, 1, 2))

        x = self.norm(x)

        return x.mean(axis=1)  # GAP here


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def pcpvt_small_v0(pretrained=False, **kwargs) -> CPVTV2:
    _ = pretrained  # no explicit checkpoint loading
    model = CPVTV2(
        patch_size=4, embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1),
        **kwargs)
    model.default_cfg = _cfg()
    return model


def pcpvt_base_v0(pretrained=False, **kwargs) -> CPVTV2:
    _ = pretrained  # no explicit checkpoint loading
    model = CPVTV2(
        patch_size=4, embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=(3, 4, 18, 3), sr_ratios=(8, 4, 2, 1),
        **kwargs)
    model.default_cfg = _cfg()
    return model


def pcpvt_large_v0(pretrained=False, **kwargs) -> CPVTV2:
    _ = pretrained  # no explicit checkpoint loading
    model = CPVTV2(
        patch_size=4, embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=(3, 8, 27, 3), sr_ratios=(8, 4, 2, 1),
        **kwargs)
    model.default_cfg = _cfg()
    return model
