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
"""global args for Transformer in Transformer(TNT)"""
import argparse
import ast
import os
import sys

import yaml

from src.configs import parser as _parser

args = None


def parse_arguments():
    """parse_arguments"""
    global args
    parser = argparse.ArgumentParser(description="MindSpore TNT Training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture")
    parser.add_argument("--accumulation_step", default=1, type=int, help="accumulation step")
    parser.add_argument("--amp_level", default="O2", choices=["O0", "O2", "O3"], help="AMP Level")
    parser.add_argument("--batch_size", default=128, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--beta", default=[0.9, 0.999], type=lambda x: [float(a) for a in x.split(",")],
                        help="beta for optimizer")
    parser.add_argument("--clip_global_norm_value", default=5., type=float, help="Clip grad value")
    parser.add_argument('--ds_train', default="./data/train", help='Training dataset')
    parser.add_argument('--ds_val', default="./data/val", help='validation dataset')
    parser.add_argument("--device_id", default=[0], type=int, nargs='+', help="Device Ids")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--device_target", default="GPU", choices=["GPU", "Ascend", "CPU"], type=str)
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR", "ONNX"],
                        default="MINDIR", help="file format")
    parser.add_argument("--in_channel", default=3, type=int)
    parser.add_argument("--is_dynamic_loss_scale", default=1, type=int, help="is_dynamic_loss_scale ")
    parser.add_argument("--keep_checkpoint_max", default=20, type=int, help="keep checkpoint max num")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument("--pynative_mode", default=0, type=int, help="graph mode with 0, python with 1")
    parser.add_argument("--mix_up", default=0., type=float, help="mix up")
    parser.add_argument("--mlp_ratio", help="mlp ", default=4., type=float)
    parser.add_argument("-j", "--num_parallel_workers", default=20, type=int, metavar="N",
                        help="number of data loading workers (default: 20)")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--warmup_length", default=0, type=int, help="Number of warmup iterations")
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warm up learning rate")
    parser.add_argument("--wd", "--weight_decay", default=0.05, type=float, metavar="W",
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--loss_scale", default=1024, type=int, help="loss_scale")
    parser.add_argument("--lr", "--learning_rate", default=5e-4, type=float, help="initial lr", dest="lr")
    parser.add_argument("--lr_scheduler", default="cosine_annealing", help="Schedule for the learning rate.")
    parser.add_argument("--lr_adjust", default=30, type=float, help="Interval to drop lr")
    parser.add_argument("--lr_gamma", default=0.97, type=int, help="Multistep multiplier")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--exclude_epoch_state", action="store_true", help="exclude epoch state and learning rate")
    parser.add_argument("--tnt_config", help="Config file to use (see configs dir)", default=None, required=True)
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--save_ckpt_every_step", default=0, type=int, help="Save checkpoint every N batches")
    parser.add_argument("--save_ckpt_every_sec", default=1800, type=int, help="Save checkpoint every N seconds")
    parser.add_argument("--save_ckpt_keep", default=20, type=int, help="Keep N checkpoints")
    parser.add_argument("--label_smoothing", type=float, help="Label smoothing to use, default 0.0", default=0.1)
    parser.add_argument("--image_size", default=224, help="Image Size.", type=int)
    parser.add_argument("--img_mean", nargs=3, type=float, default=(0.5, 0.5, 0.5), help="Image mean (model input)")
    parser.add_argument("--img_std", nargs=3, type=float, default=(0.5, 0.5, 0.5), help="Image std (model input)")
    parser.add_argument('--train_url', default="./", help='Location of training outputs.')
    parser.add_argument("--run_modelarts", type=ast.literal_eval, default=False, help="Whether run on modelarts")

    parser.add_argument("--dir_ckpt", default="ckpt", help="Root directory for checkpoints.")
    parser.add_argument("--dir_best_ckpt", default="best_ckpt", help="Root directory for best (acc) checkpoints.")
    parser.add_argument("--dir_summary", default="summary", help="Root directory for summary logs.")
    parser.add_argument("--dump_graph", action="store_true",
                        help="Dump model graph to MindInsight")
    parser.add_argument("--collect_input_data", action="store_true",
                        help="Dump input images to MindInsight")

    parser.add_argument(
        "--tnt_pt_implementation",
        default="/mindspore/Efficient-AI-Backbones/tnt_pytorch",
        help="Directory with existing implementation of TNT model (PyTorch)"
             " (see https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch)."
    )
    parser.add_argument(
        "--tnt_pt_pretrained",
        default=(
            # '/mindspore/pt_weights/tnt_s_81.5.pth.tar'
            '/mindspore/pt_weights/tnt_b_82.9.pth.tar'
        ),
        help="Arguments to PyTorch implementation (JSON-encoded list)."
    )
    parser.add_argument("--tnt_ms_export", help="Path to exported weights in MindSpore format (.ckpt).")
    parser.add_argument("--pred_output", default="preds.json", help="Path to output predictions (JSON)")
    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config()


def get_config():
    """get_config"""
    global args
    override_args = _parser.argv_to_vars(sys.argv)
    # load yaml file
    if args.run_modelarts:
        import moxing as mox
        if not args.tnt_config.startswith("obs:/"):
            args.tnt_config = "obs:/" + args.tnt_config
        with mox.file.File(args.tnt_config, 'r') as f:
            yaml_txt = f.read()
    else:
        yaml_txt = open(args.tnt_config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.tnt_config}")

    args.__dict__.update(loaded_yaml)
    print(args)

    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(args.device_num)
        os.environ["RANK_SIZE"] = str(args.device_num)


def run_args():
    """run and get args"""
    global args
    if args is None:
        parse_arguments()


run_args()
