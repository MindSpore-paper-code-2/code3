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
import mindspore.nn as nn
import numpy as np

from .token_transformer import TokenTransformer
from .token_performer import TokenPerformer
from .unfold import UnfoldCustom


class T2T_module(nn.Cell):
    """
    Tokens-to-Token encoding module
    """
    def __init__(
            self,
            img_size=224,
            tokens_type='performer',
            in_chans=3,
            embed_dim=768,
            token_dim=64,
            act_layer=nn.GELU(),
    ):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = UnfoldCustom(
                kernel_size=(7, 7), stride=(4, 4), padding=(2, 2),
                in_channels=in_chans, image_size=img_size
            )
            img_size_soft_split1 = int(np.sqrt(self.soft_split0.L))
            self.soft_split1 = UnfoldCustom(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                in_channels=token_dim,
                image_size=(img_size_soft_split1, img_size_soft_split1)
            )
            img_size_soft_split2 = int(np.sqrt(self.soft_split1.L))
            self.soft_split2 = UnfoldCustom(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                in_channels=token_dim,
                image_size=(img_size_soft_split2, img_size_soft_split2)
            )

            self.attention1 = TokenTransformer(
                dim=in_chans * 7 * 7,
                in_dim=token_dim,
                num_heads=1,
                mlp_ratio=1.0,
                act_layer=act_layer,
            )
            self.attention2 = TokenTransformer(
                dim=token_dim * 3 * 3,
                in_dim=token_dim,
                num_heads=1,
                mlp_ratio=1.0,
                act_layer=act_layer,
            )
            self.project = nn.Dense(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = UnfoldCustom(
                kernel_size=(7, 7), stride=(4, 4), padding=(2, 2),
                in_channels=in_chans, image_size=img_size
            )
            img_size_soft_split1 = int(np.sqrt(self.soft_split0.L))
            self.soft_split1 = UnfoldCustom(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                in_channels=token_dim,
                image_size=(img_size_soft_split1, img_size_soft_split1)
            )
            img_size_soft_split2 = int(np.sqrt(self.soft_split1.L))
            self.soft_split2 = UnfoldCustom(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                in_channels=token_dim,
                image_size=(img_size_soft_split2, img_size_soft_split2)
            )

            self.attention1 = TokenPerformer(
                dim=in_chans * 7 * 7,
                in_dim=token_dim,
                kernel_ratio=0.5,
                act_layer=act_layer,
            )
            self.attention2 = TokenPerformer(
                dim=token_dim * 3 * 3,
                in_dim=token_dim,
                kernel_ratio=0.5,
                act_layer=act_layer,
            )
            self.project = nn.Dense(token_dim * 3 * 3, embed_dim)

        # there are 3 soft split, stride are 4, 2, 2 separately
        self.num_patches = ((img_size // (4 * 2 * 2))
                            * (img_size // (4 * 2 * 2)))

    def construct(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose((0, 2, 1))  # B, L1, OutC

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)  # B, L1, token_dim
        B, new_HW, C = x.shape
        x = x.transpose((0, 2, 1)).reshape(
            B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW))
        )  # B, token_dim, sqrt(L1), sqrt(L1)
        # iteration1: soft split
        x = self.soft_split1(x).transpose((0, 2, 1))  # B, L1, OutC

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)  # B, L2, token_dim
        B, new_HW, C = x.shape
        x = x.transpose((0, 2, 1)).reshape(
            B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW))
        )
        # iteration2: soft split
        x = self.soft_split2(x).transpose((0, 2, 1))

        # final tokens
        x = self.project(x)

        return x
