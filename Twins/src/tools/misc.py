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
"""misc functions for program"""
import os
import logging

from mindspore import (nn, context,
                       CheckpointConfig, ModelCheckpoint, LossMonitor)
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.serialization import (
    load_checkpoint, load_param_into_net)

from ..models.twins import pcpvt, svt
from .trainer import TrainClipGrad
from .callbacks import (
    SummaryCallbackWithEval, BestCheckpointSavingCallback,
    TrainTimeMonitor, EvalTimeMonitor
)


MS_MODELS = {
    'pcpvt_s': pcpvt.pcpvt_small_v0,
    'pcpvt_b': pcpvt.pcpvt_base_v0,
    'pcpvt_l': pcpvt.pcpvt_large_v0,
    'svt_s': svt.alt_gvt_small,
    'svt_b': svt.alt_gvt_base,
    'svt_l': svt.alt_gvt_large
}


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))

    if device_target == "Ascend":
        if device_num > 1:
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            # context.set_auto_parallel_context(pipeline_stages=2, full_batch=True)

            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id[rank])
    elif device_target == "GPU":
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id[rank])
    else:
        raise ValueError("Unsupported platform.")

    return rank


# def get_dataset(args, training=True) -> data.ImageNet:
#     """"Get model according to args.set"""
#     print(f"=> Getting {ds_type} dataset")
#     dataset = getattr(data, ds_type)(args, training)
#
#     return dataset


# def get_model(args):
#     """"Get model according to args.arch"""
#     print("==> Creating model '{}'".format(args.arch))
#     model = models.__dict__[args.arch](args)
#
#     return model


def get_model(model_name, model_args) -> pcpvt.CPVTV2:
    model_args = dict(model_args)
    model_args.pop('model')
    model_args.pop('device')
    return MS_MODELS[model_name](**model_args)


def get_callbacks(arch, rank, train_data_size, val_data_size, ckpt_dir,
                  best_ckpt_dir, summary_dir, ckpt_save_every_step=0,
                  ckpt_save_every_sec=0, ckpt_keep_num=10, print_loss_every=1,
                  collect_freq=0, collect_tensor_freq=None,
                  collect_input_data=False, keep_default_action=False,
                  logging_level=logging.INFO,
                  logging_format='%(levelname)s: %(message)s'):
    """Get common callbacks."""
    logging.basicConfig(format=logging_format, level=logging_level)
    if collect_freq == 0:
        collect_freq = train_data_size
    if ckpt_save_every_step == 0 and ckpt_save_every_sec == 0:
        ckpt_save_every_step = train_data_size
    config_ck = CheckpointConfig(
        # To save every epoch use data.train_dataset.get_data_size(),
        save_checkpoint_steps=ckpt_save_every_step,
        save_checkpoint_seconds=ckpt_save_every_sec,
        keep_checkpoint_max=ckpt_keep_num,
        append_info=['epoch_num', 'step_num']
    )
    train_time_cb = TrainTimeMonitor(data_size=train_data_size)
    eval_time_cb = EvalTimeMonitor(data_size=val_data_size)

    best_ckpt_save_cb = BestCheckpointSavingCallback(
        best_ckpt_dir, prefix=arch
    )

    ckpoint_cb = ModelCheckpoint(
        prefix=f'{arch}_{rank}',
        directory=ckpt_dir,
        config=config_ck
    )
    loss_cb = LossMonitor(print_loss_every)

    specified = {
        'collect_metric': True,
        'collect_train_lineage': True,
        'collect_eval_lineage': True,
        # "histogram_regular": "^network.*weight.*",
        # "collect_graph": True,
        # "collect_dataset_graph": True,
        'collect_input_data': collect_input_data,
    }
    summary_collector_cb = SummaryCallbackWithEval(
        summary_dir=summary_dir,
        collect_specified_data=specified,
        collect_freq=collect_freq,
        keep_default_action=keep_default_action,
        collect_tensor_freq=collect_tensor_freq
    )
    return [
        train_time_cb,
        eval_time_cb,
        ckpoint_cb,
        loss_cb,
        best_ckpt_save_cb,
        summary_collector_cb
    ]


def pretrained(args, model, exclude_epoch_state=True):
    """"Load pretrained weights if args.pretrained is given"""
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        if exclude_epoch_state:
            if 'epoch_num' in param_dict:
                param_dict.pop('epoch_num')
            if 'step_num' in param_dict:
                param_dict.pop('step_num')
            if 'learning_rate' in param_dict:
                param_dict.pop('learning_rate')
        load_param_into_net(model, param_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        print(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 24, scale_factor=2,
                                                                    scale_window=2000)
    else:
        print(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)
    net_with_loss = TrainClipGrad(net_with_loss, optimizer, scale_sense=scale_sense,
                                  clip_global_norm_value=args.clip_global_norm_value,
                                  use_global_norm=True)
    return net_with_loss
