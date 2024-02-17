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
"""eval net."""

import os
import time
from mindspore import context, load_checkpoint, load_param_into_net
import mindspore.nn as nn
from src.model.SWEnet import SWEnet
from src.dataset import create_dataset, create_patches_dataset
from src.EvalMetrics import kappa
import numpy as np
import scipy.signal as signal
from PIL import Image
from model_utils.config import config
from model_utils.device_adapter import get_device_id


class CustomWithEvalCell(nn.Cell):
    def __init__(self, net):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self._network = net

    def construct(self, data1, data2):
        outputs = self._network(data1, data2)
        return outputs


def test(network_model, eval_dataset, eval_label_gt):
    rows, cols = eval_label_gt.shape
    N = rows * cols
    print("batch num-----------", eval_dataset.get_dataset_size())

    since = time.time()
    i = 0
    for d in eval_dataset.create_dict_iterator():
        aux = network_model(d["data1"], d["data2"])
        pred = aux.argmax(1).asnumpy()
        if i == 0:
            features = np.zeros((N))
        if i < eval_dataset.get_dataset_size() - 1:
            features[i * eval_dataset.get_batch_size(): (i + 1) * eval_dataset.get_batch_size()] = pred
        else:
            # special treatment for final batch
            features[i * eval_dataset.get_batch_size():] = pred
        i = i + 1

    time_elapsed = time.time() - since
    print('Inner Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return features


def calcu_changemap_predict(B, i1, epoch_i):
    if not os.path.exists(config.save_pred_path):
        os.makedirs(config.save_pred_path)

    print('calcu_changemap_predict-------------------')

    B_labels = B.reshape(i1.shape)
    output_medfilt = signal.medfilt(B_labels, kernel_size=3)

    print('======================evaluate predicted===============')
    kappa0 = kappa(B_labels, i1 / 255)
    max_x = np.max(B_labels)
    min_x = np.min(B_labels)
    B_labels = (B_labels - min_x) / (max_x - min_x) * 255
    print('io===============')
    B_labels = Image.fromarray(B_labels.astype('uint8'))
    B_labels.save(config.save_pred_path + "/index%d_%s_%.4f.tiff" % (epoch_i, "predict", kappa0))

    kappa0 = kappa(output_medfilt, i1 / 255)
    max_x = np.max(output_medfilt)
    min_x = np.min(output_medfilt)
    output_medfilt = (output_medfilt - min_x) / (max_x - min_x) * 255
    output_medfilt = Image.fromarray(output_medfilt.astype(np.uint8))
    output_medfilt.save(config.save_pred_path + "/index%d_%s_%.4f.tiff" % (epoch_i, "medfilt_predict", kappa0))


def eval_net(network, eval_data, label_GroundTruth, epoch_i):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    param_dict = load_checkpoint(config.ckpt_save_dir + "/checkpoint-{}_644.ckpt".format(epoch_i))
    # load parameter to the network
    load_param_into_net(network, param_dict)
    # load testing dataset

    eval_network = CustomWithEvalCell(network)
    eval_network.set_train(False)
    since = time.time()

    predicted_label = test(eval_network, eval_data, label_GroundTruth)

    time_elapsed = time.time() - since

    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    calcu_changemap_predict(predicted_label, label_GroundTruth, epoch_i)


if __name__ == '__main__':
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    if config.modelArts_mode:
        import moxing as mox
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=config.data_url, dst_url=local_data_url)
        local_checkpoint_url = '/cache/checkpoint'
        mox.file.copy_parallel(src_url=config.checkpoint_url, dst_url=local_checkpoint_url)
        save_pred_path = '/cache/pred'

        config.ckpt_save_dir = local_checkpoint_url
        config.dataroot = local_data_url
        config.save_pred_path = save_pred_path
    device_id = get_device_id()

    features1, features2, labels, label_gt = create_patches_dataset(root_dir=config.dataroot, mode='eval')
    eval_ds = (features1, features2, labels)
    eval_ds = create_dataset(config.eval_batch_size, 1, eval_ds, False)
    # create the network
    SWE_net = SWEnet(2)
    for epoch in range(1, config.epochs+1):
        eval_net(SWE_net, eval_ds, label_gt, epoch)

    if config.modelArts_mode:
        # copy eval result from cache to obs
        if config.rank == 0:
            mox.file.copy_parallel(src_url=config.save_pred_path, dst_url=config.eval_url)
