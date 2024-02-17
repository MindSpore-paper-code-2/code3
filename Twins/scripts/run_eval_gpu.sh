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
    echo "Usage: bash ./scripts/run_distribute_train_gpu.sh N_GPU CONFIG_PATH DEVICE_IDS EXTRA_ARGS"
    echo "Call CUDA_VISIBLE_DEVICES=\"k1,k2,...\" bash ... to set available GPUs"
exit 1
fi
N_GPU=$1
CONFIG_PATH=$2
DEVICE_IDS=$3
if [ $# -eq 4 ]
then
    EXTRA_ARGS=$4
else
    EXTRA_ARGS=""
fi
export RANK_SIZE=$N_GPU
export DEVICE_NUM=$N_GPU
#export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"


rm -rf ./train_parallel
mkdir ./train_parallel
cp -r ../src ./train_parallel
cp ../train.py ./train_parallel
DIR_CKPT=./train_parallel/ckpt/
DIR_BEST_CKPT=./train_parallel/ckpt_best/
DIR_SUMMARY=./train_parallel/summary/
echo "start training"
cd ./train_parallel || exit
env > env.log
mpirun -n $N_GPU --allow-run-as-root --output-filename log_output \
  python train.py --device-id $DEVICE_IDS --device-target GPU \
  --config $CONFIG_PATH \
  --dir-ckpt $DIR_CKPT \
  --dir-best-ckpt $DIR_BEST_CKPT \
  --dir-summary $DIR_SUMMARY $EXTRA_ARGS \
    > log 2>&1 &
cd ..
