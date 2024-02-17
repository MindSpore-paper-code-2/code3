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
"""Convert model weights from PyTorch to MindSpore"""
import json

import torch

from mindspore import context
from mindspore.common import set_seed
import mindspore as ms

from src.args import parse_args
from src.tools.cell import cast_amp
from src.tools.misc import get_model


def main():
    args = parse_args()
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.pynative_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)

    # get model and cast amp_level
    ms_net = get_model(args.model, json.loads(args.params))
    cast_amp(ms_net, args)

    def prepare_pt_key(cur_name: str):  # -> pt
        pt_name = cur_name
        if cur_name.startswith('merge_blocks.'):
            _, i, j, suffix = cur_name.split('.', 3)
            k = int(j)
            if k > 1:
                k -= 1
            pt_name = 'blocks.{}.{}.{}'.format(i, k, suffix)
        pt_name = (pt_name
                   .replace('model.', '')
                   .replace('.beta', '.bias')
                   .replace('.gamma', '.weight'))
        return pt_name

    ms_name_map = {
        prepare_pt_key(name): 'model.' + name
        for name, param in ms_net.parameters_and_names()
    }  # pt -> ms

    pt_weights = torch.load(args.src)
    ms_weights = []
    for name_pt, param_pt in pt_weights.items():
        name_ms = ms_name_map[name_pt]
        ms_weights.append({
            'name': name_ms,
            'data': ms.Tensor(param_pt.data.cpu().numpy())
        })

    ms_net.phase = 'eval'
    ms.save_checkpoint(ms_weights, args.dst)
    print('Checkpoint was saved at', args.dst)


if __name__ == '__main__':
    main()
