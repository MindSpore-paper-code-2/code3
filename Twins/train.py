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
"""Train PCPVT/SVT model"""

import os
import json
import datetime
import functools

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed

from src.args import parse_args
from src.tools.cell import cast_amp
from src.tools.common import get_callbacks
from src.tools.criterion import get_criterion, NetWithLoss
from src.data.imagenet import init_dataset
from src.tools.misc import (set_device, get_model,
                            pretrained, get_train_one_step)
from src.tools.optimizers import get_optimizer


def main():
    args = parse_args()

    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.pynative_mode],
                        device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)

    # get model and cast amp_level
    net = get_model(args.model, json.loads(args.params))
    cast_amp(net, args)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)

    # data = get_dataset(args)
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
    batch_num = ds_train.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    if args.pretrained:
        pretrained(args, net_with_loss, args.exclude_epoch_state)

    eval_network = nn.WithEvalCell(net, criterion)  # , args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    cur_name = datetime.datetime.now().strftime('%y-%m-%d_%H%M%S')
    ckpt_save_dir = "{}/{}_{}".format(args.dir_ckpt, cur_name, rank)
    ckpt_best_save_dir = "{}/{}_{}".format(args.dir_best_ckpt, cur_name, rank)
    summary_dir = "{}/{}".format(args.dir_summary, cur_name)
    cb = get_callbacks(
        args.model, rank, ds_train.get_dataset_size(),
        ds_val.get_dataset_size(), ckpt_save_dir, ckpt_best_save_dir,
        summary_dir, args.save_ckpt_every_step, args.save_ckpt_every_sec,
        args.save_ckpt_keep, print_loss_every=100,
        collect_graph=args.dump_graph
    )

    print("begin train")
    print('Number of parameters:',
          sum(functools.reduce(lambda x, y: x * y, params.shape)
              for params in net.trainable_params()))
    print('Number of samples in dataset:'
          ' train={}, val={}'.format(ds_train.get_dataset_size(),
                                     ds_val.get_dataset_size()))

    model.fit(int(args.epochs - args.start_epoch), ds_train, ds_val,
              callbacks=cb, dataset_sink_mode=True)
    print("train success")

    if args.run_modelarts:
        import moxing as mox
        mox.file.copy_parallel(
            src_url=ckpt_save_dir,
            dst_url=os.path.join(args.train_url, "ckpt_" + str(rank))
        )


if __name__ == '__main__':
    main()
