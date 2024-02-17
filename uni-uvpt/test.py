
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
# Modified from https://github.com/lhoyer/DAFormer/

# !/usr/bin/env python3
# -*- coding:utf-8 -*-


import argparse

from datasets import build_dataloader, build_dataset
from models import build_segmentor

from utils import Config, ProgressBar

import mindspore as ms

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--data_url', type=str, help="path to dataset")
    args = parser.parse_args()
    return args


def run_test(model, data_loader, no_return_feat=False):
    """Test with single GPU.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for _, data in enumerate(data_loader):

        prog_bar.update()
        data['img_metas'] = data['img_metas'][0].data
        result = model(return_loss=False, **data)

        if isinstance(result, tuple) and len(result) == 2 and no_return_feat:
            feat, out_seg = result
            feat = None
            result = (feat, out_seg)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

    return results

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if not args.data_url is None:
        cfg.data_root = args.data_url
        cfg.data.train.data_root = args.data_url
        cfg.data.val.data_root = args.data_url
        cfg.data.test.data_root = args.data_url

    cfg = update_legacy_cfg(cfg)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.gpu_ids = range(1)
    cfg.data.workers_per_gpu = 2

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    ms.load_checkpoint(args.checkpoint, model, strict_load=True)

    outputs = run_test(model, data_loader, no_return_feat=True)
    dataset.evaluate(outputs, args.eval)


if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    main()
