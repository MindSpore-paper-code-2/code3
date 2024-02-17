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
"""Training script for TNT model"""
import time
import datetime
import functools

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed

from src.tools.common import get_callbacks
from src.tools.cell import cast_amp
# from src.tools.callbacks import StopAtEpoch
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import (
    get_dataset, set_device, get_model, pretrained, get_train_one_step
)
from src.tools.optimizer import get_optimizer


def main():
    from src.args import args
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
    net = get_model(args)
    cast_amp(net, args)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    if args.pretrained:
        pretrained(args, net_with_loss, args.exclude_epoch_state)

    eval_network = nn.WithEvalCell(net, criterion,
                                   args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    cur_name = datetime.datetime.now().strftime('%y-%m-%d_%H%M%S')
    ckpt_save_dir = "{}/{}_{}".format(args.dir_ckpt, cur_name, rank)
    ckpt_best_save_dir = "{}/{}_{}".format(args.dir_best_ckpt, cur_name, rank)
    summary_dir = "{}/{}".format(args.dir_summary, cur_name)
    # if args.run_modelarts:
    #     ckpt_save_dir = "/cache/ckpt_" + str(rank)

    cb = get_callbacks(
        args.arch, rank, data.train_dataset.get_dataset_size(),
        data.val_dataset.get_dataset_size(), ckpt_save_dir, ckpt_best_save_dir,
        summary_dir, args.save_ckpt_every_step, args.save_ckpt_every_sec,
        args.save_ckpt_keep, print_loss_every=100,
        collect_graph=args.dump_graph
    )

    print("begin train")
    print('Number of parameters:',
          sum(functools.reduce(lambda x, y: x * y, params.shape)
              for params in net.trainable_params()))
    print('Number of samples in dataset:'
          ' train={}, val={}'.format(data.train_dataset.get_dataset_size(),
                                     data.val_dataset.get_dataset_size()))
    # cb.append(StopAtEpoch(summary_dir, 1, args.epochs - args.start_epoch))

    sink_mode = True
    t1 = time.time()
    model.fit(int(args.epochs - args.start_epoch), data.train_dataset,
              data.val_dataset, callbacks=cb, dataset_sink_mode=sink_mode)
    t2 = time.time()
    dt = 1000 * (t2 - t1)
    print('Total training time: {:.3f} ms, time per epoch: {:.3f} ms,'
          ' time per batch: {:.3f} ms, time per element: {:.3f} ms'
          .format(dt, dt / args.epochs,
                  dt / args.epochs / data.train_dataset.get_dataset_size(),
                  dt / args.epochs /
                  data.train_dataset.get_dataset_size() / args.batch_size))
    print("train success")


if __name__ == '__main__':
    main()
