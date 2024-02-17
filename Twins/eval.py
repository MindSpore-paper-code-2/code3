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
"""Evaluation script. Need training config."""
import os
import json
from functools import reduce

import mindspore as ms
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import TimeMonitor

from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.misc import pretrained, get_train_one_step
from src.data.imagenet import create_dataset_imagenet
from src.tools.optimizers import get_optimizer
from src.tools.misc import get_model


def eval_ckpt(args):
    """Evaluate MindSpore Twins model by checkpoint"""
    print('=== Use checkpoint ===')
    net = get_model(args.model, json.loads(args.params))
    cast_amp(net, args)
    criterion = get_criterion(args)

    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained(args, net)

    print(
        'Number of parameters (before deploy):',
        sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in net.trainable_params()
        )
    )
    # switch_net_to_deploy(net)
    print(
        'Number of parameters (after deploy):',
        sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in net.trainable_params()
        )
    )
    cast_amp(net, args)
    net.set_train(False)

    data = create_dataset_imagenet(
        str(args.ds_val), args, training=False
    )
    batch_num = data.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    eval_network = nn.WithEvalCell(
        net, criterion, args.amp_level in ['O2', 'O3', 'auto']
    )
    eval_indexes = [0, 1, 2]
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net_with_loss, metrics=eval_metrics,
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    print('=> begin eval')
    results = model.eval(data, callbacks=[TimeMonitor()])
    return results


def eval_mindir(args):
    """Evaluate MindSpore Twins model by MindIR exported model"""
    print('=== Use MINDIR model ===')
    data = create_dataset_imagenet(
        str(args.dataset_path), args, training=False
    )
    iterator = data.create_dict_iterator(num_epochs=1)

    graph = ms.load(str(args.pretrained))
    net = nn.GraphCell(graph)
    metrics = {
        'Top1-Acc': nn.Top1CategoricalAccuracy(),
        'Top5-Acc': nn.Top5CategoricalAccuracy(),
    }
    print('=> begin eval')
    for batch in iterator:
        y_pred = net(batch['image'])
        for metric in metrics.values():
            metric.update(y_pred, batch['label'])

    return {name: metric.eval() for name, metric in metrics.items()}


def main():
    """Entry point."""
    from src.args import parse_args
    args = parse_args()

    set_seed(0)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == 'Ascend':
        context.set_context(enable_auto_mixed_precision=True)

    os.environ["RANK_SIZE"] = '0'

    # get model
    if args.pretrained.endswith('.ckpt'):
        results = eval_ckpt(args)
    elif args.pretrained.endswith('.mindir'):
        results = eval_mindir(args)
    else:
        raise ValueError('Incorrect format checkpoint')

    print(f'=> eval results:{results}')
    print('=> eval success')


if __name__ == '__main__':
    main()
