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
"""
Data operations, will be used in train.py and eval.py
"""
import os
from dataclasses import dataclass
import math

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter

from src.data.augment.auto_augment import _pil_interp, rand_augment_transform
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing
from .data_utils.moxing_adapter import sync_data


class ImageNet:
    """ImageNet Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Download data.')
            local_data_path = '/cache/data'
            sync_data(args.data_url, local_data_path, threads=128)
            print('Create train and evaluate dataset.')
            train_dir = os.path.join(local_data_path, "train")
            val_ir = os.path.join(local_data_path, "val")
            self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)
        else:
            # train_dir = os.path.join(args.data_url, "train")
            # val_ir = os.path.join(args.data_url, "val")
            if training:
                self.train_dataset = create_dataset_imagenet(args.ds_train, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(args.ds_val, training=False, args=args)


def create_dataset_imagenet(dataset_dir, args, repeat_num=1, training=True,
                            preloaded_ds=None
                            ) -> ds.ImageFolderDataset:
    """
    create a train or eval imagenet2012 dataset for TNT

    Args:
        dataset_dir(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()
    if device_num is None:
        device_num = 1
    shuffle = bool(training)
    ds.config.set_prefetch_size(args.batch_size)
    if preloaded_ds is not None:
        data_set = preloaded_ds
    else:
        shard_args = {}
        if device_num > 1 and training:
            shard_args = {'num_shards': device_num,
                          'shard_id': rank_id}
        data_set = ds.ImageFolderDataset(
            dataset_dir, num_parallel_workers=args.num_parallel_workers,
            shuffle=shuffle, **shard_args
        )

    image_size = args.image_size

    # define map operations
    # BICUBIC: 3

    mean, std = args.img_mean, args.img_std  # ImageNet: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if training:
        aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        interpolation = args.interpolation
        auto_augment = args.auto_augment
        assert auto_augment.startswith('rand')
        aa_params['interpolation'] = _pil_interp(interpolation)

        transform_img = [
            vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                                          interpolation=Inter.BICUBIC),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.ToPIL()
        ]
        transform_img += [rand_augment_transform(auto_augment, aa_params)]
        transform_img += [
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
            RandomErasing(args.re_prob, mode=args.re_mode, max_count=args.re_count)
        ]
    else:
        mean = (np.array(mean) * 255).tolist()
        std = (np.array(std) * 255).tolist()

        # As in the initial repo.
        crop_pct = 0.9
        if isinstance(image_size, tuple):
            assert len(image_size) == 2
            if image_size[-1] == image_size[-2]:
                # fall-back to older behaviour so Resize scales to shortest edge if target is square
                scale_size = int(math.floor(image_size[0] / crop_pct))
            else:
                scale_size = tuple([int(x / crop_pct) for x in image_size])
        else:
            scale_size = int(math.floor(image_size / crop_pct))

        transform_img = [
            vision.Decode(),
            vision.Resize(scale_size, interpolation=Inter.BICUBIC),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    transform_label = C.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    if (args.mix_up > 0. or args.cutmix > 0.) and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = C.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=one_hot)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=True,
                              num_parallel_workers=args.num_parallel_workers)

    if (args.mix_up > 0. or args.cutmix > 0.) and training:
        mixup_fn = Mixup(
            mixup_alpha=args.mix_up, cutmix_alpha=args.cutmix, cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smoothing, num_classes=args.num_classes)

        data_set = data_set.map(operations=mixup_fn, input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id


@dataclass
class DatasetParams:
    """Dataset arguments as a namespace"""
    batch_size: int
    num_parallel_workers: int
    image_size: int
    img_mean: list
    img_std: list
    interpolation: str
    auto_augment: str
    re_prob: float
    re_mode: str
    re_count: int
    num_classes: int
    mix_up: float  # alpha
    mixup_prob: float  # prob
    mixup_mode: str
    switch_prob: float
    cutmix: float
    label_smoothing: float


def init_dataset(
        dataset_dir, batch_size: int,
        num_parallel_workers: int,
        image_size: int,
        img_mean: list,
        img_std: list,
        interpolation: str,
        auto_augment: str,
        re_prob: float,
        re_mode: str,
        re_count: int,
        num_classes: int,
        mix_up: float,
        mixup_prob: float,
        mixup_mode: str,
        switch_prob: float,
        cutmix: float,
        label_smoothing: float, repeat_num=1, training=True,
        preloaded_ds=None,
        **kwargs
) -> ds.ImageFolderDataset:
    """Initialize dataset with explicit parameter names"""
    _ = kwargs
    args = DatasetParams(
        batch_size,
        num_parallel_workers,
        image_size,
        img_mean,
        img_std,
        interpolation,
        auto_augment,
        re_prob,
        re_mode,
        re_count,
        num_classes,
        mix_up,
        mixup_prob,
        mixup_mode,
        switch_prob,
        cutmix,
        label_smoothing
    )
    return create_dataset_imagenet(
        dataset_dir, args, repeat_num=repeat_num, training=training,
        preloaded_ds=preloaded_ds
    )


def get_transforms(
        image_size: int, training: bool, **aug: dict
):
    """Get images preprocessing according mode and augmentations settings.

    Parameters
    ----------
    image_size: int
        Target image size.
    training: bool
        Mode. If True augmentations may be applied.
    aug: Dict
        Augmentation settings (type, auto aug, random erase).

    Returns
    -------
        List of transforms.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    aug = {} if aug is None else aug
    if training:
        if aug['type'] == 'weak':
            transform = [
                vision.ToPIL(),
                vision.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                    interpolation=Inter.BILINEAR
                ),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.ToTensor(),
                vision.Normalize(mean, std, is_hwc=False),
            ]
        elif aug['type'] == 'none':
            transform = [
                vision.ToPIL(),
                vision.Resize(image_size, interpolation=Inter.BILINEAR),
                vision.CenterCrop(image_size),
                vision.ToTensor(),
                vision.Normalize(mean, std, is_hwc=False),
            ]
        elif aug['type'] == 'auto':
            aa_params = dict(
                translate_const=int(image_size * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in mean]),
                interpolation=_pil_interp(aug['interpolation'])
            )
            auto_augment = aug['auto_augment']

            transform = [
                vision.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                    interpolation=Inter.BILINEAR
                ),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.ToPIL()
            ]
            if auto_augment is not None:
                transform += [rand_augment_transform(auto_augment, aa_params)]
            transform += [
                vision.ToTensor(),
                vision.Normalize(mean=mean, std=std, is_hwc=False),
                RandomErasing(
                    aug['re_prob'], mode=aug['re_mode'],
                    max_count=aug['re_count']),
            ]
        else:
            raise ValueError('???' + aug.get('type', 'Unknown'))
    else:
        transform = [
            vision.ToPIL(),
            vision.Resize(
                int((256 / 224) * image_size), interpolation=Inter.BILINEAR
            ),
            vision.CenterCrop(image_size),
            vision.ToTensor(),
            vision.Normalize(mean, std, is_hwc=False),
        ]

    return transform
