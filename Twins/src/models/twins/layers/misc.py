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
"""Tensor/array manipulation routines"""

import collections.abc
from itertools import repeat

from scipy.stats import truncnorm


def trunc_array(shape, sigma=0.02):
    """output truncnormal array in shape"""
    return truncnorm.rvs(-2, 2, loc=0, scale=sigma, size=shape, random_state=None)


def _ntuple(n):
    "get _ntuple"

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
