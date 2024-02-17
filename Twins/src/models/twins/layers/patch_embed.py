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
"""Patch embedding layer"""

from mindspore import nn

from .misc import to_2tuple


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, name=''):
        super().__init__()

        def set_name(src):
            return 'p{}.{}'.format(name, src)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.h, self.w = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.h * self.w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                              has_bias=True)
        self.proj.weight.name = set_name(self.proj.weight.name)
        self.proj.bias.name = set_name(self.proj.bias.name)
        self.norm = nn.LayerNorm([embed_dim])
        self.norm.beta.name = set_name(self.norm.beta.name)
        self.norm.gamma.name = set_name(self.norm.gamma.name)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        # b, c, h, w = x.shape

        x = self.proj(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.transpose((0, 2, 1))
        x = self.norm(x)

        return x
