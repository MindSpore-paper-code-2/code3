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
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash scripts/run_infer_onnx.sh [ONNX_PATH] [DATA_PATH] [OUTPUT_PATH] [DEVICE_TARGET(optional)]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

ONNX_PATH=$(get_real_path "$1")
DATA_PATH=$(get_real_path "$2")
OUTPUT_PATH=$(get_real_path "$3")

if [ ! -f "$ONNX_PATH" ]
then
    echo "error: CHECKPOINT_PATH=$ONNX_PATH is not a file"
exit 1
fi

if [ $# -eq 4 ]; then
  DEVICE_TARGET="$4"
else
  DEVICE_TARGET=CPU
fi

echo "onnx_path: $ONNX_PATH"
echo "dataset_path: $DATA_PATH"
echo "output_path: $OUTPUT_PATH"
echo "device_target: $DEVICE_TARGET"

python ./infer_onnx.py "$DATA_PATH" --onnx_path "$ONNX_PATH" --device_target "$DEVICE_TARGET" --output "$OUTPUT_PATH" &> infer_onnx.log &