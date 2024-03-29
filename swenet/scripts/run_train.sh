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

if [ $# != 1 ]
then
    echo "Usage: sh run_train.sh [DATA_PATH]"
exit 1
fi


if [ ! -d $1 ]
then
    echo "error: DATA_PATH=$1 is not a directory"
exit 1
fi


#ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8


EXECUTE_PATH=$(pwd)
config_path="${EXECUTE_PATH}/config.yaml"
echo "config path is : ${config_path}"

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))


cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cp -r ./model_utils ./train_parallel$i
    cp -r ./*.yaml ./train_parallel$i
    cp ./train.py ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./train_parallel$i ||exit
    env > env.log
    taskset -c $cmdopt python train.py --dataroot='../'$1  --is_distributed True > log 2>&1 &
    cd ..
done