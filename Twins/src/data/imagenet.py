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
"""Read dataset in ImageNet format"""

import os
import math
from dataclasses import dataclass

from PIL import Image
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import Tensor

from src.data.augment.auto_augment import (_pil_interp, rand_augment_transform,
                                           augment_and_mix_transform, auto_augment_transform)
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)


class ImageList:
    """Simple list of images and labels"""

    def __init__(self, root, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.initialized = False

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels:
            target = self.labels[idx]
            return img, target
        return img


class ImageNet(ImageList):
    """ImageNet synonym for image list"""


def transforms_noaug_train(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
):
    """Get list of transformations for validation (no random augmentations)"""
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        vision.Resize(img_size, _pil_interp(interpolation)),
        vision.CenterCrop(img_size)
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [vision.ToNumpy()]
    else:
        tfl += [
            vision.ToTensor(),
            vision.Normalize(mean=Tensor(mean), std=Tensor(std))
        ]
    return tfl


def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    Get list of ImageNet transformations for training.
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = [
        vision.RandomResizedCrop(img_size, scale=scale, ratio=ratio)]
    if hflip > 0.:
        primary_tfl += [vision.RandomHorizontalFlip(prob=hflip)]
    if vflip > 0.:
        primary_tfl += [vision.RandomVerticalFlip(prob=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [vision.ToNumpy()]
    else:
        final_tfl += [
            vision.ToTensor(),
            vision.Normalize(
                mean=mean,
                std=std)
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits))

    if separate:
        return primary_tfl, secondary_tfl, final_tfl
    return primary_tfl + secondary_tfl + final_tfl


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD
):
    """Get list of ImageNet transformations for validation (no augmentations)"""
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        vision.Resize(scale_size, _pil_interp(interpolation)),
        vision.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [vision.ToNumpy()]
    else:
        tfl += [
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std)
        ]

    return tfl


def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False
) -> list:
    """Initialize train/val transformations"""

    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation)
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std)
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                auto_augment=auto_augment,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                separate=separate)
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                crop_pct=crop_pct)

    return transform


def build_transform(is_train, args):
    """Build transformation+tensor preparation pipeline"""
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            auto_augment=args.auto_augment,
            interpolation=args.interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform[0] = vision.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            vision.Resize(size),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(vision.CenterCrop(args.input_size))

    t.append(vision.ToTensor())
    t.append(vision.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return t


def build_dataset(is_train, args):
    """Create dataset object with embedded transformations"""
    transform = build_transform(is_train, args)

    from torch.utils.data import Dataset

    class ClassificationDataset(Dataset):
        """Dataset for classification.
        """

        def __init__(self, split='train', pipeline=None):
            if split == 'train':
                self.data_source = ImageNet(root='data/imagenet/train',
                                            list_file='data/imagenet/meta/train.txt')
            else:
                self.data_source = ImageNet(root='data/imagenet/val',
                                            list_file='data/imagenet/meta/val.txt')
            self.pipeline = pipeline

        def __len__(self):
            return self.data_source.get_length()

        def __getitem__(self, idx):
            img, target = self.data_source.get_sample(idx)
            if self.pipeline is not None:
                img = self.pipeline(img)

            return img, target

    dataset = ClassificationDataset(
        'train' if is_train else 'val',
        pipeline=transform
    )
    nb_classes = 1000

    return dataset, nb_classes


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
    shuffle = bool(training)
    ds.config.set_prefetch_size(args.batch_size)
    if preloaded_ds is not None:
        data_set = preloaded_ds
    else:
        shard_args = {}
        if device_num is not None and training:
            shard_args = {'num_shards': device_num,
                          'shard_id': rank_id}
        data_set = ds.ImageFolderDataset(
            dataset_dir,
            num_parallel_workers=args.num_parallel_workers,
            shuffle=shuffle,
            **shard_args
        )

    # define map operations
    # BICUBIC: 3

    # transform_img = build_transform(training, args)
    image_size = args.image_size
    if training:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
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
            RandomErasing(args.reprob, mode=args.remode, max_count=args.recount)
        ]
    else:
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        # test transform complete
        transform_img = [
            vision.Decode(),
            vision.Resize(int(256 / 224 * image_size), interpolation=Inter.BICUBIC),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    transform_label = C.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    if (args.mixup > 0. or args.cutmix > 0.)  and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = C.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=one_hot)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=True,
                              num_parallel_workers=args.num_parallel_workers)

    if (args.mixup > 0. or args.cutmix > 0.) and training:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smoothing, num_classes=args.num_classes)

        data_set = data_set.map(operations=mixup_fn, input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


@dataclass
class DatasetParams:
    """Namespace with dataset arguments (only useful)"""
    batch_size: int
    num_parallel_workers: int
    image_size: int
    input_size: int
    interpolation: str
    auto_augment: str
    reprob: float
    remode: str
    recount: int
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
    """Initialize dataset (explicit list of arguments)"""
    _ = kwargs  # suppress other keys
    args = DatasetParams(
        batch_size,
        num_parallel_workers,
        image_size,
        image_size,
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
