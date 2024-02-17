# Contents

- [Contents](#contents)
- [TNT Description](#tnt-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Eval process](#eval-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
            - [Result](#result-2)
        - [Infer with ONNX](#infer-with-onnx)
            - [Result](#result-3)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TNT Description](#contents)

TNT is a deep neural network for use as a classifier model or as a deep feature extractor.

[Paper](https://arxiv.org/abs/2103.00112) Transformer in Transformer

# [Model architecture](#contents)

TNT model is a tranformer network. The overall network architecture of TNT is show below:

[Link](https://arxiv.org/abs/2103.00112) Transformer in Transformer

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 224*224 colorful images in 1000 classes
    - Train: 120G, 1281167 images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（GPU/CPU）
    - Prepare hardware environment with GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
├── TNT
  ├── README.md              # descriptions about TNT
  ├── configs
  │   ├── tnt_b_patch16_224_imagenet.yaml  # example of configuration for TNT-B
  │   ├── tnt_s_patch16_224_imagenet.yaml  # example of configuration for TNT-B
  ├── scripts
  │   ├── run_distribute_train_gpu.sh  # shell script for distributed training on multiple GPUs
  │   ├── run_eval_gpu.sh              # shell script for MindSpore model evaluation
  │   ├── run_eval_onnx.sh             # shell script for ONNX model evaluation
  │   ├── run_infer_gpu.sh             # shell script for MindSpore model inference
  │   ├── run_infer_onnx.sh            # shell script for ONNX model inference
  │   ├── run_standalone_train_gpu.sh  # example of shell script for training on 1 GPU
  ├── src
  │   ├── configs
  |   │   ├── config.py          # YAML configuration parser
  │   ├── data
  │   |   ├── augment
  |   │   |   ├── auto_augment.py    # augmentation set builder
  |   │   |   ├── mixup.py           # MixUp augmentation
  |   │   |   ├── random_erasing.py  # 'random erasing' augmentation
  │   |   ├── data_utils
  |   │   |   ├── moxing_adapter.py  # DS synchronization for distributed training
  |   |   ├── imagenet.py            # wrapper for reading ImageNet dataset
  │   ├── models
  │   |   ├── tnt
  │   |   |   ├── layers
  |   │   |   |   ├── attention.py  # Attention layer
  |   │   |   |   ├── misc.py  # extra tools and layers
  |   │   |   |   ├── patch_embed.py  # PatchEmbed layer
  |   │   |   |   ├── unfold_kernel.py  # Unfold
  |   │   |   ├── tnt.py  # implementation of TNT architecture
  │   ├── tools
  │   |   ├── callbacks.py  # callback functions (implementation)
  │   |   ├── cell.py  # tune model layers/parameters
  │   |   ├── common.py  # initialize callbacks
  │   |   ├── criterion.py  # model training objective function (implementation)
  │   |   ├── get_misc.py  # initialize optimizers and other arguments for training process
  │   |   ├── optimizer.py  # model optimizer function (implementation)
  │   |   ├── schedulers.py  # training (LR) scheduling function (implementation)
  │   ├── trainers
  │   |   ├── train_one_step_with_scale_and_clip_global_norm.py  # example training script

  ├── convert_from_pt.py  # convert model weights from PyTorch model
  ├── eval.py             # evaluation script (MindSpore)
  ├── eval_onnx.py        # evaluation script (ONNX)
  ├── export.py           # export MindIR/ONNX script
  ├── infer.py            # run inference on single file/folder (MindSpore)
  ├── infer_onnx.py       # run inference on single file/folder (ONNX)
  ├── train.py            # training script
  ├── tune_params.py      # tune hyperparameters for trainable model
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- GPU: bash scripts/run_distribute_train_gpu.sh GPU [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]

### Launch

```shell
# training example
  python:
      GPU: python train.py --tnt_config configs/tnt_s.yaml \
             --dir_ckpt /mindspore/save/ckpt/ --dir_best_ckpt /mindspore/save/best_ckpt/ \
             --dir_summary /mindspore/save/summary/
  shell:
      GPU: bash scripts/run_train.sh
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `--dir_ckpt / --dir_best_ckpt` by default.

```bash
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## [Eval process](#contents)

### Usage

You can start evaluation using python script. The usage of script is as follows:

- GPU: python eval.py --tnt_config configs/tnt_s.yaml --pretrained save/ckpt/.../tnt_s_patch16_224-0.yaml

### Launch

```shell
# infer example
  python:
    GPU: python eval.py --dataset_path ~/imagenet/val/ --pretrained pretrained.ckpt --device_target GPU
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the following in `val.log`.

```bash
result: {'acc': 0.7252} ckpt=/path/to/checkpoint/tnt_s_patch16_224-200_625.ckpt
```

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --pretrained [CKPT_PATH] --tnt_config configs/tnt_s.yaml --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`DEVICE` should be in ['GPU']
`FILE_FORMAT` should be in "MINDIR"

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | TNT-S                     |
| -------------------------- | ------------------------- |
| Model Version              | large                     |
| Resource                   | 4 * NV RTX 3090           |
| uploaded Date              |  |
| MindSpore Version          | 1.8.0/1.9.0               |
| Dataset                    | ImageNet                  |
| Training Parameters        | configs/tnt_s_patch16_224_imagenet.yaml |
| Optimizer                  | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy       |
| outputs                    | probability               |
| Loss                       | 2.0..3.0                  |
| Accuracy                   | ACC1[0.7252]              |
| Total time                 | ~660 h                    |
| Params (M)                 | 23768584                  |
| Checkpoint for Fine tuning | 91 M                      |
| Scripts                    |  |

# [Description of Random Situation](#contents)

We use fixed seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
