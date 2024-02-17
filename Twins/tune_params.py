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
"""Debug script for tuning PCPVT-L hyperparameters"""

import argparse
import datetime
import time
import json
from pathlib import Path

import yaml
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import hyperopt as hy

import src.tools.callbacks as cb_fn
from src.data.imagenet import init_dataset
from src.tools.misc import get_model, get_train_one_step, set_device
from src.tools.optimizers import get_learning_rate
from src.tools.criterion import NetWithLoss, SoftTargetCrossEntropy


rank = None


def parse_args():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', '-c', type=Path, required=True,
                        help='Path to hyperparameter ranges (YAML)')
    parser.add_argument('--iters', '-n', type=int, default=10,
                        help='Number of trials')

    return parser.parse_args()


def sample_args(hp_params):
    """Prepare base arguments for optimization"""
    args = argparse.Namespace(**hp_params)

    for key, val in {
            'ds_train': '/mindspore/data/small/train/',
            'ds_val': '/mindspore/data/small/val/',
            'dir_summary': '/mindspore/save/1/summary/',
            'device_id': [0],
            'device_target': 'GPU',
            'model': 'pcpvt_s',
            'params': json.dumps({"model": "pcpvt_large_v0", "num_classes": 1000,
                                  "drop_rate": 0.0, "drop_path_rate": 0.0,
                                  "device": "cpu:0"}),
            'is_dynamic_loss_scale': True,
            'loss_scale': 1.0,
            'clip_global_norm_value': 5.0,
            'num_parallel_workers': 1,
            'image_size': 224,
            'interpolation': 'bicubic',
            'auto_augment': 'rand-m9-mstd0.5-inc1',
            'remode': 'pixel',
            'recount': 1,
            'resplit': False,
            'num_classes': 1000,
            'cutmix-minmax': None,
            'mixup-mode': 'batch',

            'epochs': 5,
            'lr_scheduler': 'constant_lr',
            'warmup_lr': 1e-4,
            'warmup_length': 0,
            'min_lr': 0.0
    }.items():
        setattr(args, key.replace('-', '_'), val)
    return args


def collect_hyperopt_params(config):
    """PRepare a list of HyperOpt bounds/ranges by configuration"""
    def convert_single_param(cur_key, cur_elem):
        if 'bounds' in cur_elem.keys():
            low, high = cur_elem['bounds']
            if cur_elem.get('kind') == 'log':
                return hy.hp.loguniform(cur_key, np.log(low), np.log(high))
            return hy.hp.uniform(cur_key, low, high)
        if 'choice' in cur_elem.keys():
            return hy.hp.choice(cur_key, cur_elem['choice'])
        raise AttributeError()

    res = {}
    for key, elem in config.items():
        res[key] = convert_single_param(key, elem)
    return res


def check_opts(hp_params):
    """
    Run training on a single set of parameters
    and evaluate the model
    """
    global rank
    args = sample_args(hp_params)
    print('Current params', args)

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target=args.device_target)
    ms.context.set_context(enable_graph_kernel=False)
    # rank = set_device(args)
    if rank is None:
        rank = set_device(args)

    # Replace batch_size with params.batch_size.
    ds_params = (
        args.batch_size,
        args.num_parallel_workers,
        args.image_size,
        args.interpolation,
        args.auto_augment,
        args.reprob,
        args.remode,
        args.recount,
        args.num_classes,
        args.mixup,
        args.mixup_prob,
        args.mixup_mode,
        args.switch_prob,
        args.cutmix,
        args.label_smoothing
    )
    ds_train = init_dataset(args.ds_train, *ds_params,
                            training=True, preloaded_ds=None)
    ds_val = init_dataset(args.ds_val, *ds_params,
                          training=False, preloaded_ds=None)

    ds_size = ds_train.get_dataset_size()

    net = get_model(args.model, json.loads(args.params))
    # Replace cfg.learning_rate with params.learning_rate.
    lr = get_learning_rate(args, ds_size)
    # Replace cfg.momentum with params.momentum.
    weight_decay = 0.99
    optimizer = nn.Momentum(
        net.trainable_params(),
        learning_rate=lr,
        momentum=args.momentum,
        weight_decay=weight_decay
    )

    # Instantiate SummaryCollector and add it to callback to automatically collect training information.
    dump_graph = False
    collect_input_data = False
    criterion = SoftTargetCrossEntropy()
    net_with_loss = NetWithLoss(net, criterion)
    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, False)
    eval_indexes = [0, 1, 2]
    model = ms.Model(net_with_loss, metrics={"acc", "loss"},
                     eval_network=eval_network,
                     eval_indexes=eval_indexes)
    cur_name = datetime.datetime.now().strftime('%y-%m-%d_%H%M%S')
    summary_dir = "{}/{}".format(args.dir_summary, cur_name)
    summary_cb = cb_fn.SummaryCallbackWithEval(
        summary_dir=summary_dir,
        collect_specified_data={
            "collect_metric": True,
            'collect_train_lineage': True,
            'collect_eval_lineage': True,
            # "histogram_regular": "^network.*weight.*",
            "collect_graph": dump_graph,
            # "collect_dataset_graph": True,
            'collect_input_data': collect_input_data,
        },
        collect_freq=1,
        keep_default_action=False,
        collect_tensor_freq=100
    )
    loss_cb = ms.train.LossMonitor(1)
    model.fit(args.epochs, ds_train, ds_val,
              callbacks=[loss_cb, summary_cb])
    eval_metrics = model.eval(ds_val)

    return {
        'loss': -eval_metrics['acc'],
        'status': hy.STATUS_OK,
        'eval_time': time.time(),
        'other_stuff': {},
        'attachments': {}
    }


def main():
    args = parse_args()

    with open(args.config, 'r') as fp:
        hp_config = yaml.load(fp, Loader=yaml.FullLoader
                              )['parameters']
    hp_params = collect_hyperopt_params(hp_config)
    best = hy.fmin(fn=check_opts,
                   space=hp_params,
                   algo=hy.tpe.suggest,
                   max_evals=args.iters)
    print(best)


if __name__ == '__main__':
    main()
