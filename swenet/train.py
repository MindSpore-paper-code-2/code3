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
"""train net."""
import os
import time
import random
import numpy as np
import mindspore as ms
from mindspore import context, Model, load_checkpoint, load_param_into_net
import mindspore.common.initializer as weight_init
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.communication.management import init, get_rank, get_group_size
import mindspore.nn as nn
from model_utils.device_adapter import get_device_id
from model_utils.config import config
from src.dataset import create_dataset, create_patches_dataset
from src.Losses import WithLossCell
from src.model.SWEnet import SWEnet

ms.common.set_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)


def poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * ((1.0 - step_ / decay_steps) ** power) + end_lr


def train_net(network_model, train_dataset, epoch_size, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    is_train = True
    ds_train = create_dataset(config.batch_size, 4, train_dataset, is_train, target=config.device_target,
                              distribute=config.is_distributed)
    print(ds_train.get_dataset_size())
    config_ckt = CheckpointConfig(ds_train.get_dataset_size(), keep_checkpoint_max=30)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint", directory=config.ckpt_save_dir, config=config_ckt)
    since = time.time()
    cb = [ckpoint_cb, LossMonitor()]
    if config.is_distributed and device_id != 0:
        cb = [cb[1]]
    network_model.train(epoch_size, ds_train, callbacks=cb, dataset_sink_mode=sink_mode)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def init_weight(net):
    """init_weight"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))


if __name__ == '__main__':

    ckpt_save_dir = config.ckpt_save_dir + "/"

    if config.modelArts_mode:
        import moxing as mox
        # download dataset from obs to cache
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=config.data_url, dst_url=local_data_url)
        local_train_url = '/cache/checkpoint/'

        if "obs://" in config.checkpoint_path:
            local_checkpoint_url = "/cache/" + config.checkpoint_path.split("/")[-1]
            mox.file.copy_parallel(config.checkpoint_path, local_checkpoint_url)
            config.checkpoint_path = local_checkpoint_url
        config.dataroot = local_data_url
        config.ckpt_save_dir = local_train_url

    device_id = get_device_id()
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True,
                                          device_num=config.group_size)

    # create the network
    network = SWEnet(2)
    if config.pre_trained:
        param_dict = load_checkpoint(config.checkpoint_path)
        load_param_into_net(network, param_dict)
    init_weight(net=network)

    features1, features2, labels, _ = create_patches_dataset(root_dir=config.dataroot, mode='train')
    train_data = (features1, features2, labels)
    # define the loss function
    loss_net = WithLossCell(network)
    # learning rate setting
    lr = poly_lr(config.lr, config.epochs * train_data[0].shape[0]//config.batch_size//config.group_size,
                 config.epochs * train_data[1].shape[0]//config.batch_size//config.group_size, end_lr=0.0, power=0.9)

    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), lr, config.momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=train_data[0].shape[0]//config.batch_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)

    model = Model(network=loss_net, optimizer=net_opt)

    train_net(model, train_data, config.epochs, sink_mode=config.dataset_sink_mode)

    if config.modelArts_mode:
        # copy train result from cache to obs
        if config.rank == 0:
            mox.file.copy_parallel(src_url=config.ckpt_save_dir, dst_url=config.train_url)
