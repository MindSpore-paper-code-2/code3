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

# Modified from: https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0

from .compose import Compose
from .formatting import (Collect, ImageToTensor, ToDataContainer, ToTensor,
                         Transpose, to_tensor)
from .loading import LoadAnnotations, LoadImageFromFile
from .test_time_aug import MultiScaleFlipAug
from .transforms import (CLAHE, AdjustGamma, Normalize, Pad,
                         PhotoMetricDistortion, RandomCrop, RandomCutOut,
                         RandomMosaic, RandomRotate, Rerange,
                         Resize, RGB2Gray, SegRescale)
