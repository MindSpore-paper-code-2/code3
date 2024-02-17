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
"""
create train or eval dataset.
"""
import os
import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.communication.management import init, get_rank, get_group_size


def generate_data(i1, i2, label):
    neibor_size = 5
    add_size = neibor_size // 2
    rows, cols = i1.shape

    rows_array1 = i1[:, 0:add_size]
    rows_array2 = i2[:, 0:add_size]
    i1 = np.hstack((rows_array1, i1, rows_array1))
    i2 = np.hstack((rows_array2, i2, rows_array2))

    cols_array1 = i1[0:add_size, :]
    cols_array2 = i2[0:add_size, :]
    i1 = np.vstack((cols_array1, i1, cols_array1))
    i2 = np.vstack((cols_array2, i2, cols_array2))

    n = 0
    train_x_i1 = np.zeros((rows * cols, neibor_size, neibor_size))
    train_x_i2 = np.zeros((rows * cols, neibor_size, neibor_size))
    train_label = np.zeros((rows * cols))
    for i in range(rows):
        for j in range(cols):
            neibor_1 = i1[i:neibor_size + i, j:neibor_size + j]
            neibor_2 = i2[i:neibor_size + i, j:neibor_size + j]
            train_x_i1[n] = neibor_1
            train_x_i2[n] = neibor_2
            train_label[n] = int(label[i, j])
            n = n + 1

    print('num_patches: ', n)
    train_x_i1 = np.stack((train_x_i1, train_x_i1, train_x_i1), axis=-1)
    train_x_i2 = np.stack((train_x_i2, train_x_i2, train_x_i2), axis=-1)
    return train_x_i1, train_x_i2, train_label


def create_patches_dataset(root_dir='../dataset/', mode="train"):
    if mode == "train":
        # --------------------------------train data------------------------------
        # dataset river
        i1_1 = open_image(os.path.join(root_dir, "Yellow River/Yellow River I-SAR/im1.bmp"))
        i2_1 = open_image(os.path.join(root_dir, "Yellow River/Yellow River I-SAR/im2.bmp"))
        label_gt_1 = open_image(os.path.join(root_dir, "Yellow River/Yellow River I-SAR/YR-I-ref.bmp")) / 255

        # farmlamdD
        i1_2 = open_image(os.path.join(root_dir, "Yellow River/Yellow River II-SAR/im1.bmp"))
        i2_2 = open_image(os.path.join(root_dir, "Yellow River/Yellow River II-SAR/im2.bmp"))
        label_gt_2 = open_image(os.path.join(root_dir, "Yellow River/Yellow River II-SAR/YR-II-ref.bmp")) / 255

        # coasline
        i1_3 = open_image(os.path.join(root_dir, "Yellow River/Yellow River III-SAR/im1.bmp"))
        i2_3 = open_image(os.path.join(root_dir, "Yellow River/Yellow River III-SAR/im2.bmp"))
        label_gt_3 = open_image(os.path.join(root_dir, "Yellow River/Yellow River III-SAR/YR-III-ref.bmp")) / 255

        features1_1, features2_1, labels_1 = generate_data(i1_1, i2_1, label_gt_1)
        features1_2, features2_2, labels_2 = generate_data(i1_2, i2_2, label_gt_2)
        features1_3, features2_3, labels_3 = generate_data(i1_3, i2_3, label_gt_3)

        features1 = np.concatenate((features1_1, features1_2, features1_3), axis=0)
        features2 = np.concatenate((features2_1, features2_2, features2_3), axis=0)
        labels = np.concatenate((labels_1, labels_2, labels_3))
        label_gt = None

    elif mode == "eval":
        # -----------------------------validate data -------------------------------
        # farmlandC
        i1 = open_image(os.path.join(root_dir, "Yellow River/Yellow River IV-SAR/im1.bmp"))
        i2 = open_image(os.path.join(root_dir, "Yellow River/Yellow River IV-SAR/im2.bmp"))
        label_gt = open_image(os.path.join(root_dir, "Yellow River/Yellow River IV-SAR/YR-IV-ref.bmp")) / 255

        # ottawa
        # i1 = open_image(os.path.join(root_dir, "Ottawa/1997.05.bmp"))
        # i2 = open_image(os.path.join(root_dir, "Ottawa/1997.08.bmp"))
        # label_gt = open_image(os.path.join(root_dir, "Ottawa/Ottawa_ref.bmp")) / 255

        # RR
        # i1 = open_image(os.path.join(root_dir, "Red River-SAR/RedRiver_1.bmp"))
        # i2 = open_image(os.path.join(root_dir, "Red River-SAR/RedRiver_2.bmp"))
        # label_gt = open_image(os.path.join(root_dir, "Red River-SAR/RedRiver_ref.bmp")) / 255

        features1, features2, labels = generate_data(i1, i2, label_gt)

    elif mode == 'test':
        # -----------------------------test data -------------------------------

        i1 = open_image(os.path.join(root_dir, "De-Gaule airport-SAR/GauleAirport_1.bmp"))
        i2 = open_image(os.path.join(root_dir, "De-Gaule airport-SAR/GauleAirport_2.bmp"))
        label_gt = open_image(os.path.join(root_dir, "De-Gaule airport-SAR/GauleAirport_ref.bmp")) / 255

        features1, features2, labels = generate_data(i1, i2, label_gt)
    else:
        raise ValueError('invalid mode')

    return features1, features2, labels, label_gt


def UniformLabelSampler(labels):
    N = len(labels)
    images_lists = [[] for i in range(2)]

    for i in range(N):
        images_lists[int(labels[i])].append(i)  #

    size_per_pseudolabel = int(N / len(images_lists)) + 1

    res = np.zeros(size_per_pseudolabel * len(images_lists))

    for i in range(len(images_lists)):
        indexes = np.random.choice(
            images_lists[i],
            size_per_pseudolabel,
            replace=(len(images_lists[i]) <= size_per_pseudolabel)
        )

        res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

    np.random.shuffle(res)
    return res[:N].astype('int')


def create_dataset(batch_size, num_parallel_workers, data, is_train, target="Ascend", distribute=True):
    """ create dataset for train or test
    Args:
        dataset path(string): the path of dataset.
        batch_size: The number of data records in each groups
        num_parallel_workers: The number of parallel workers
    """

    hwc2chw_op = CV.HWC2CHW()  # change shape from (height, width, channel) to (channel, height, width) to fit network.
    type_cast_op = C.TypeCast(mstype.int32)  # change data type of label to int32 to fit network
    normalize_op = CV.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms_list_data = [normalize_op, hwc2chw_op]

    if target != "Ascend":
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
            rank_id = 0

    else:
        device_num, rank_id = get_rank_information()
        print(device_num, rank_id)

    if is_train:

        #train_sample = UniformLabelSampler(data[2])

        dataset = ds.NumpySlicesDataset(data, num_parallel_workers=4, sampler=None,
                                        column_names=["data1", "data2", "label"], shuffle=True, num_shards=device_num,
                                        shard_id=rank_id)

    else:
        dataset = ds.NumpySlicesDataset(data, column_names=["data1", "data2", "label"], shuffle=False)

    dataset = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)

    dataset = dataset.map(operations=transforms_list_data, input_columns="data1",
                          num_parallel_workers=num_parallel_workers)

    dataset = dataset.map(operations=transforms_list_data, input_columns="data2",
                          num_parallel_workers=num_parallel_workers)

    buffer_size = 10000
    if is_train:
        dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.batch(batch_size, drop_remainder=False)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset


def open_image(img_path):
    return np.array(Image.open(img_path).convert('L'))


def get_rank_information():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    print('rank size', rank_size)
    if rank_size <= 1:
        rank_size = 1
        rank_id = 0

    else:
        rank_size = get_group_size()
        rank_id = get_rank()

    return rank_size, rank_id
