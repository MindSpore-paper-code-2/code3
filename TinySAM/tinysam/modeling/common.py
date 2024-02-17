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
from typing import Type

import mindspore
import mindspore.nn as nn


class MLPBlock(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Cell] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Dense(embedding_dim, mlp_dim)
        self.lin2 = nn.Dense(mlp_dim, embedding_dim)
        self.act = act()

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Cell):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mindspore.Parameter(mindspore.ops.ones(num_channels))
        self.bias = mindspore.Parameter(mindspore.ops.zeros(num_channels))
        self.eps = eps

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        u = x.mean(1, keep_dims=True)
        s = (x - u).pow(2).mean(1, keep_dims=True)
        x = (x - u) / mindspore.ops.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
