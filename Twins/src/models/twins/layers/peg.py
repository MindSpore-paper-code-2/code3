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
"""Conditional position encoding (PEG) block"""

from mindspore import nn


class PosCNN(nn.Cell):
    """Position embedding module from https://arxiv.org/abs/2102.10882"""

    def __init__(self, in_chans, embed_dim=768, s=1,
                 name='', h=0, w=0):
        super(PosCNN, self).__init__()
        self.proj = nn.SequentialCell([nn.Conv2d(
            in_chans, embed_dim, 3, s,
            pad_mode='pad', padding=1, has_bias=True, group=embed_dim
        )])
        for elem in self.proj:
            elem.weight.name = '{}.{}'.format(name, elem.weight.name)
            elem.bias.name = '{}.{}'.format(name, elem.bias.name)
        self.s = s
        self.h, self.w = h, w

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        b, _, c = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose((0, 2, 1)).view(b, c, self.h, self.w)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.reshape((x.shape[0], x.shape[1], -1)).transpose((0, 2, 1))
        return x
