#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# != 2 ]; then
  echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -f $PATH1 ]; then
  echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
  exit 1
fi

if [ ! -d $PATH2 ]; then
  echo "error: TRAIN_DATA_DIR=$PATH2 is not a directory"
  exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ../*.py ./train_parallel$i
  cp *.sh ./train_parallel$i
  cp -r ../src ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env >env.log

  nohup python train.py \
        --batch_size 16 \
        --lr 1e-4 \
        --scale 2 \
        --task_id 0 \
        --dir_data $PATH2 \
        --epochs 1000 \
        --test_every 100 \
        --patch_size 48 > train.log 2>&1 &
  cd ..
done
