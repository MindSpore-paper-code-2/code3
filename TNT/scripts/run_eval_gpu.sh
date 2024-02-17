#!/bin/bash
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

if [ $# -lt 2 ]
then
    echo "Usage: bash ./scripts/run_eval_gpu.sh DEVICE_ID CONFIG_PATH PRETRAINED_PATH"
    echo "Call CUDA_VISIBLE_DEVICES=\"k1,k2,...\" bash ... to set available GPUs"
exit 1
fi
DEVICE_ID=$1
CONFIG_PATH=$2
PRETRAINED_PATH=$3

python eval.py --tnt_config "$CONFIG_PATH" \
  --device_id $DEVICE_ID \
  --pretrained "$PRETRAINED_PATH"

