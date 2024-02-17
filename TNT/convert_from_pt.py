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
import torch

from mindspore import context
from mindspore.common import set_seed
import mindspore as ms

from src.args import args
from src.tools.cell import cast_amp
from src.tools.get_misc import get_model


def main():
    set_seed(args.seed)
    # mode = {
    #     0: context.GRAPH_MODE,
    #     1: context.PYNATIVE_MODE
    # }
    # context.set_context(mode=mode[args.pynative_mode], device_target=args.device_target)
    mode = context.PYNATIVE_MODE
    context.set_context(mode=mode, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)

    # get model and cast amp_level
    ms_net = get_model(args)
    cast_amp(ms_net, args)

    ms_name_map = {
        param.name
        .replace('model.', '')
        .replace('.beta', '.bias')
        .replace('.gamma', '.weight'): param.name
        for name, param in ms_net.parameters_and_names()
    }  # pt -> ms

    pt_weights = torch.load(args.tnt_pt_pretrained)
    for name_pt, param_pt in pt_weights.items():
        if name_pt not in ms_name_map.keys():
            print('Unmatched parameter (not in MS model):', name_pt)
            continue
        val_pt = param_pt.data.cpu().numpy()
        name_ms = ms_name_map[name_pt]
        param_ms = ms_net.parameters_dict()[name_ms]
        param_ms.set_data(ms.Tensor(val_pt))

    # names_ms = [param.name
    #             .replace('model.', '')
    #             .replace('.beta', '.bias')
    #             .replace('.gamma', '.weight')
    #             for name, param in ms_net.parameters_and_names()
    #             ]
    # names_pt = [name for name, param in pt_net.state_dict().items()
    #             if name != 'outer_tokens']
    # print('MS\n', sorted(names_ms),
    #       '\n\nPT\n', sorted(names_pt))
    # exit()
    # with open('ms.txt', 'w') as fp:
    #     fp.write(str(ms_net))
    # with open('pt.txt', 'w') as fp:
    #     fp.write(str(pt_net))

    # ms_net.phase = 'eval'
    # random_input = np.random.random((1, 3, 224, 224))
    # ms_out = (
    #     ms_net.construct(ms.Tensor(random_input)
    #                      .astype(ms.float32))
    #     .asnumpy()
    # )
    # pt_out = (
    #     pt_net(torch.tensor(random_input, dtype=torch.float32
    #                         ).to('cpu:0')
    #            ).detach().numpy()
    # )
    # print('Norm of difference between PT and MS predictions:',
    #       np.linalg.norm((pt_out - ms_out).ravel()))
    ms.save_checkpoint(ms_net, args.tnt_ms_export)
    print('Checkpoint was saved at', args.tnt_ms_export)


if __name__ == '__main__':
    main()
