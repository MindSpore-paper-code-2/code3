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
"""Optimizer creation."""
import logging

import numpy as np
from mindspore.nn.optim import AdamWeightDecay, Adam
from mindspore.nn.optim.momentum import Momentum

from src.tools.schedulers import get_policy


def get_learning_rate(args, batch_num):
    """Get learning rate"""
    return get_policy(args.lr_scheduler)(args, batch_num)


def get_optimizer(args, model, batch_num):
    """Get optimizer for training"""
    logging.info('When using train_wrapper, using optimizer %s',
                 args.optimizer)
    optim_type = args.optimizer.lower()
    params = get_param_groups(model)
    learning_rate = get_learning_rate(args, batch_num)
    step = int(args.start_epoch * batch_num)
    train_step = len(learning_rate)
    learning_rate = learning_rate[step:]
    logging.info('Get LR from epoch: %d', args.start_epoch)
    logging.info('Start step: %d', step)
    logging.info('Total step: %d', train_step)

    logging.info('learning_rate %f', np.max(learning_rate))
    if optim_type == 'momentum':
        optim = Momentum(
            params=params,
            learning_rate=learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif optim_type == 'adamw':
        optim = AdamWeightDecay(
            params=params,
            learning_rate=learning_rate,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif optim_type == 'adam':
        optim = Adam(
            params=params,
            learning_rate=learning_rate,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps
        )
    else:
        raise ValueError(f'optimizer {optim_type} is not supported')

    return optim


def get_param_groups(network):
    """get param groups"""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.weight'):
            # Dense or Conv's weight using weight decay
            decay_params.append(x)
        else:
            # all bias not using weight decay
            # bn weight bias not using weight decay, be carefully for now x
            # not include LN
            no_decay_params.append(x)

    return [
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': decay_params}
    ]
