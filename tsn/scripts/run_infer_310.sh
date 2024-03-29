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

if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DATA_PATH] [MODALITY] [TEST_LIST] [SCORE_NAME] [DEVICE_TARGET] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

model=$(get_real_path $1)

if [ "$2" == "y" ] || [ "$2" == "n" ];then
    need_preprocess=$2
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

data_path=$(get_real_path $3)

if [ "$4" == "Flow" ] || [ "$4" == "RGB" ] || [ "$4" == "RGBDiff" ];then
    modality=$4
else
  echo "device_target must be in  ['Flow', 'RGB', 'RGBDiff']"
  exit 1
fi

test_list=$5
score_name=$6
if [ "$7" == "GPU" ] || [ "$7" == "CPU" ] || [ "$7" == "Ascend" ];then
    device_target=$7
else
  echo "device_target must be in  ['GPU', 'CPU', 'Ascend']"
  exit 1
fi


device_id=0
if [ $# == 8 ]; then    
    device_id=$8
fi

echo "mindir name: "$model
echo "need_preprocess: "$need_preprocess
echo "data_path: "$data_path
echo "modality: "$modality
echo "test_list: "$test_list
echo "device_target: "$device_target
echo "device_id: "$device_id

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --modality=$modality --test_list=$test_list --dataset_path=$data_path --result_path=./preprocess_Result/
}

function compile_app()
{
    cd ../ascend310_infer/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log    
}

function infer()
{
    echo ""
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/out/main --mindir_path=$model --input0_path=../ascend310_infer/preprocess_Result/00_data --device_id=$device_id  &> infer.log
}

function cal_acc()
{
    python ../postprocess.py --test_list=$test_list --modality=$modality --result_dir=./result_Files --label_dir=../ascend310_infer/preprocess_Result/label_ids.npy --save_scores=$score_name  &> acc.log
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess dataset failed"
        exit 1
    fi
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi