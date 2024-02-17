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
"""Parse CLI arguments for train/eval scripts"""

import argparse
from pathlib import Path

import yaml
from src.tools.misc import MS_MODELS


def add_model_args(parser: argparse.ArgumentParser):
    """Add model/base configuration"""
    parser.add_argument('--config', type=Path, required=False,
                        help='Path to configuration file (YAML)')
    parser.add_argument('--model', type=str, choices=MS_MODELS.keys(),
                        required=False,
                        help='Model type')
    parser.add_argument('--params', type=str, required=False,
                        help='Model arguments (JSON-encoded)')
    parser.add_argument('--pretrained', type=str, required=False,
                        help='Source checkpoint (.pt)')
    parser.add_argument('--exclude_epoch_state', action='store_true',
                        help='exclude epoch state and learning rate')


def add_ds_args(parser: argparse.ArgumentParser):
    """Add dataset configuration"""
    parser.add_argument('--ds-train', type=str, required=False,
                        help='Path to training dataset (ImageNet format)')
    parser.add_argument('--ds-val', type=str, required=False,
                        help='Path to validation dataset (ImageNet format)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num-parallel-workers', type=int, default=1,
                        help='Number of dataset parallel reading workers')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (px, each side)')
    parser.add_argument('--interpolation', type=str, default='bilinear',
                        help='Interpolation method')
    parser.add_argument('--auto-augment', type=str, default='rand-m9-mstd0.5-inc1',
                        help='Augmentation set')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes')
    parser.add_argument('--switch-prob', type=float, default=0.5,
                        help='Switch augmentation probability')


def add_environment_args(parser: argparse.ArgumentParser):
    """Add training environment configuration (optimization, LR)"""
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='Optimization function')
    parser.add_argument('--beta', type=float, nargs='*', default=[0.9, 0.999],
                        help='Optimization function beta')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Optimization functions epsilon')
    parser.add_argument('--lr-scheduler', type=str, default='cosine_lr',
                        help='LR scheduling algorithm')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='First epoch index')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--warmup-length', type=int, default=20,
                        help='Number of warmup iterations for LR scheduler')
    parser.add_argument('--base-lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--accumulation-step', type=int, default=1,
                        help='Interval for logging data')
    parser.add_argument('--is-dynamic-loss-scale', type=int, default=1,
                        help='Use dynamic loss scale')
    parser.add_argument('--clip-global-norm-value', type=float, default=5.0,
                        help='Limit for gradient norm value')
    parser.add_argument('--amp-level', type=str, default='O0',
                        help='AMP optimization level')

    parser.add_argument('--pynative-mode', type=int, default=0,
                        help='Use pynative mode for GPU'
                             ' (if 0, then graph mode)')
    parser.add_argument('--device-target', type=str, choices=('CPU', 'GPU', 'Ascend'),
                        default='GPU',
                        help='Target accelerator')
    parser.add_argument('--device-id', type=int, nargs='+', default=[0],
                        help='Device IDs')
    parser.add_argument('--device-num', type=int, default=1,
                        help='Number of accelerator devices')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')


def add_aug_args(parser: argparse.ArgumentParser):
    """Add augmentation parameters"""
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--label-smoothing', type=float, default=0.5,
                        help='Label smoothing intensity')


def add_dirs_args(parser: argparse.ArgumentParser):
    """Add directories as parameters: checkpoints, logs"""
    parser.add_argument('--dir-ckpt', type=Path, required=False,
                        help='Directory for periodical checkpoints (.ckpt)')
    parser.add_argument('--dir-best-ckpt', type=Path, required=False,
                        help='Directory for best checkpoints (.ckpt)')
    parser.add_argument('--dir-summary', type=Path, required=False,
                        help='Directory for summary logs')
    parser.add_argument("--dump_graph", action="store_true",
                        help="Dump model graph to MindInsight")
    parser.add_argument("--collect_input_data", action="store_true",
                        help="Dump input images to MindInsight")
    parser.add_argument('--save-ckpt-every-step', type=int, default=0,
                        help='Interval for saving checkpoints (training steps)')
    parser.add_argument('--save-ckpt-every-sec', type=int, default=1800,
                        help='Interval for saving checkpoints (seconds)')
    parser.add_argument('--save-ckpt-keep', type=int, default=20,
                        help='Number of saved checkpoints'
                             ' (old checkpoints are removed)')


def add_export_args(parser: argparse.ArgumentParser):
    """Add parameters for model export"""
    parser.add_argument('--seed', type=int, required=False, default=1,
                        help='Random seed initial value')
    parser.add_argument('--run-modelarts', action='store_true',
                        help='Run inside the ModelArts infrastructure')
    parser.add_argument('--export-format', type=str, choices=['MINDIR', 'ONNX'],
                        default='ONNX',
                        help='Format of exported model')
    parser.add_argument('--src', type=str,
                        help='Source checkpoint for weight conversion')
    parser.add_argument('--dst', type=str,
                        help='Destination path for converted weights')


def parse_args():
    """Parse all groups of arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_model_args(parser)
    add_ds_args(parser)
    add_environment_args(parser)
    add_aug_args(parser)
    add_dirs_args(parser)
    add_export_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_data = yaml.load(fp, Loader=yaml.FullLoader)
            for key, val in config_data.items():
                args.__setattr__(key, val)
    from pprint import pprint
    pprint(vars(args))

    return args
