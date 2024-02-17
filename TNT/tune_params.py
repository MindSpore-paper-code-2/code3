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
"""Tune hyperparameters for training"""

import argparse
from typing import List

import mindspore as ms
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed

import hyperopt as hy
from hyperopt import hp
import typer

from src.tools import callbacks
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import (
    get_dataset, set_device, get_model, get_train_one_step
)
from src.tools.optimizer import get_optimizer


app = typer.Typer(pretty_exceptions_show_locals=False)


def objective(dict_args):
    """Target function for optimization: try single set of params"""
    args = argparse.Namespace(**dict_args)

    # get model and cast amp_level
    net = get_model(args)
    cast_amp(net, args)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)

    data = get_dataset(argparse.Namespace(
        run_modelarts=False,
        set=args.ds_set,
        **dict_args
    ))
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    # if pretrained:
    #     pretrained(argparse.Namespace(
    #         run_modelarts=False,
    #         **locals()
    #     ), net_with_loss, exclude_epoch_state)

    eval_network = nn.WithEvalCell(net, criterion,
                                   args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)
    cb = [
        callbacks.TrainTimeMonitor(data_size=data.train_dataset.get_dataset_size()),
        callbacks.EvalTimeMonitor(data_size=data.val_dataset.get_dataset_size()),
        ms.LossMonitor(1)
    ]

    print('Number of samples in dataset:'
          ' train={}, val={}'.format(data.train_dataset.get_dataset_size(),
                                     data.val_dataset.get_dataset_size()))
    model.fit(args.epochs, data.train_dataset,
              data.val_dataset, callbacks=cb, dataset_sink_mode=args.ds_sink_mode)
    metrics = model.eval(data.val_dataset)
    acc = metrics['acc']
    return -acc


@app.command()
def main(arch: str = 'tnt_s_patch16_224',
         amp_level: str = "O2",
         batch_size: int = 128,
         beta: List[float] = (0.9, 0.99),
         clip_global_norm_value: float = 5.,
         ds_train: str = './data/train',
         ds_val: str = './data/val',
         device_id: List[int] = (0,),
         accumulation_step: int = 1,
         device_num: int = 1,
         device_target: str = 'GPU',
         epochs: int = 5,
         eps: float = 1e-8,
         file_format: str = 'MINDIR',
         in_channel: int = 3,
         is_dynamic_loss_scale: int = 1,
         keep_checkpoint_max: int = 20,
         optimizer: str = 'adamw',
         ds_set: str = 'ImageNet',
         pynative_mode: int = 1,
         auto_augment: str = 'rand-m9-mstd0.5-inc1',
         interpolation: str = 'bicubic',
         re_prob: float = 0.25,
         re_mode: str = 'pixel',
         re_count: int = 1,
         mix_up: float = 0.,
         mixup_prob: float = 1.,
         switch_prob: float = 0.5,
         mixup_mode: str = 'batch',
         cutmix: float = 1.0,
         mlp_ratio: float = 4.,
         num_parallel_workers: int = 1,
         start_epoch: int = 0,
         warmup_length: int = 0,
         warmup_lr: float = 5e-7,
         weight_decay: float = 0.05,
         loss_scale: int = 1024,
         min_lr: float = 0.000006,
         base_lr: float = 0.0005,
         lr: float = 5e-4,
         lr_scheduler: str = 'cosine_lr',
         lr_adjust: int = 30,
         lr_gamm: float = 0.97,
         momentum: float = 0.9,
         num_classes: int = 1000,
         exclude_epoch_state: bool = False,
         label_smoothing: float = 0.1,
         image_size: int = 224,
         ds_sink_mode: bool = True,
         drop_path_rate: float = 0.0,
         drop_out: float = 0.0,
         max_evals: int = 10):
    set_seed(0)
    print('Base arguments:',
          (arch, amp_level, batch_size, beta, clip_global_norm_value,
           ds_train, ds_val, device_id, accumulation_step, device_num,
           device_target, epochs, eps,
           file_format, in_channel, is_dynamic_loss_scale,
           keep_checkpoint_max,
           optimizer, ds_set, pynative_mode, auto_augment, interpolation,
           re_prob, re_mode, re_count, mix_up, mixup_prob,
           switch_prob, mixup_mode, cutmix, mlp_ratio, num_parallel_workers,
           start_epoch, warmup_length, warmup_lr, weight_decay, loss_scale,
           min_lr, base_lr, lr, lr_scheduler, lr_adjust, lr_gamm,
           momentum, num_classes, exclude_epoch_state, label_smoothing,
           image_size, ds_sink_mode, drop_path_rate, drop_out, max_evals))

    def convert_val(key, val):
        if isinstance(val, (int, float, str, list, tuple)):
            return val
        assert isinstance(val, dict), (val, type(val))
        if 'bounds' in val.keys():
            return hp.uniform(key + '_b', *val['bounds'])
        if 'choice' in val.keys():
            return hp.choice(key + '_c', val['choice'])
        if 'range' in val.keys():
            return hp.quniform(key + '_r', *val['range'], 1)
        raise AssertionError('Unknown type specification: {}: {}'
                             .format(key, val))

    search_space = locals()
    search_space.pop('convert_val')
    search_space.update({
        'weight_decay': {'bounds': [0.99, 0.999]},
        'lr_scheduler': 'constant_lr',
        'warmup_length': 0,
        'base_lr': {'bounds': [1e-6, 1e-3]},
        're_prob': {'bounds': [0.1, 0.5]},
        'switch_prob': {'bounds': [0.1, 0.9]},
    })
    space = hp.choice('a', [
        {
            key: convert_val(key, val)
            for key, val in search_space.items()
        }
    ])

    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[pynative_mode],
                        device_target=device_target)
    context.set_context(enable_graph_kernel=False)
    if device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    _ = set_device(argparse.Namespace(**locals()))

    best = hy.fmin(objective, space, algo=hy.tpe.suggest, max_evals=max_evals)

    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print(hy.space_eval(space, best))
    # -> ('case 2', 0.01420615366247227}


if __name__ == '__main__':
    app()
