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
"""
##############export checkpoint file into air, mindir models#################
python export.py
"""

import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.model.SWEnet import SWEnet
from model_utils.config import config
from model_utils.device_adapter import get_device_id


def main():

    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())

    network = SWEnet(2)
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)

    input_arr1 = Tensor(np.ones([1, 3, config.imageSize, config.imageSize]), ms.float32)
    input_arr2 = Tensor(np.ones([1, 3, config.imageSize, config.imageSize]), ms.float32)

    export(network, input_arr1, input_arr2, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    main()
