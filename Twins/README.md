# Contents

- [Contents](#contents)
- [Twins Description](#twins-description)
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

# [Twins Description](#contents)

Twins is a deep neural network for use as a classifier model or as a deep feature extractor.

[Paper](https://arxiv.org/abs/2103.00112) Transformer in Transformer

# [Model architecture](#contents)

Twins model is a tranformer network. The overall network architecture of Twins is show below:

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
├── Twins
  ├── README.md  # description of Twins
  ├── configs  # example of configuration:
  │   ├── optimize.yaml  # ... for hyperparameter tuning
  │   ├── pcpvt_l.yaml   # ... for Twins PCPVT-L
  │   ├── svt_s.yaml     # ... for Twins SVT-S
  ├── scripts  # shell scripts:
  │   ├── run_distribute_training_gpu.sh  # train model on multiple GPUs
  │   ├── run_eval_gpu.sh                 # evaluate accuracy (MindSpore)
  │   ├── run_eval_onnx.sh                # evaluate accuracy (ONNX)
  │   ├── run_infer_gpu.sh                # run single file/directory inference (MindSpore)
  │   ├── run_infer_onnx.sh               # run single file/directory inference (ONNX)
  │   ├── run_standalone_training_gpu.sh  # train model on a single GPU
  ├── src
  │   ├── data
  │   |   ├── augment  # augmentations:
  |   │   |   ├── auto_augment.py    # AutoAugment set builder
  |   │   |   ├── mixup.py           # MixUp augmentation
  |   │   |   ├── random_erasing.py  # Random Erasing augmentation
  │   |   ├── data_utils
  |   │   |   ├── moxing_adapter.py  # DS synchronization for distributed training
  |   |   ├── imagenet.py            # wrapper for reading ImageNet dataset
  │   ├── models
  │   |   ├── twins
  │   |   |   ├── layers  # Twins layers
  |   │   |   |   ├── attention.py      # Attention
  |   │   |   |   ├── drop_path.py      # DropPath
  |   │   |   |   ├── misc.py           # extra tools and layers
  |   │   |   |   ├── patch_embed.py    # PatchEmbed
  |   │   |   |   ├── unfold_kernel.py  # Unfold
  |   │   |   ├── pcpvt.py  # base PCPVT model (architecture + common block)
  |   │   |   ├── svt.py  # SVT model changes (architecture + SVT block)
  │   ├── tools
  │   |   ├── callbacks.py   # callback functions (implementation)
  │   |   ├── cell.py        # tune model layers/parameters
  │   |   ├── common.py      # initialize callbacks
  │   |   ├── criterion.py   # model training objective function (implementation)
  │   |   ├── misc.py        # initialize optimizers and other arguments for training process
  │   |   ├── optimizer.py   # model optimizer function (implementation)
  │   |   ├── schedulers.py  # training (LR) scheduling function (implementation)
  │   |   ├── trainer.py     # extra training tools
  │   ├── args.py  # arguments for other scripts

  ├── convert_from_pt.py  # convert model weights from PyTorch model
  ├── eval.py             # evaluation script (MindSpore)
  ├── eval_onnx.py        # evaluation script (ONNX)
  ├── export.py           # export model to MindIR/ONNX
  ├── infer.py            # run inference on single file/directory (MindSpore)
  ├── infer_onnx.py       # run inference on single file/directory (ONNX)
  ├── train.py            # training script
  ├── tune_params.py      # tune hyperparameters for trainable model
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- GPU: bash scripts/run_distributed_training_gpu.sh GPU [DEVICE_NUM] [CONFIG_PATH] "[DEVICE_IDS]" [EXTRA_ARGS...]

### Launch

```shell
# training example
  python:
      GPU: python train.py --config configs/svt_s.yaml \
             --dir-ckpt /mindspore/save/ckpt/ --dir-best-ckpt /mindspore/save/best_ckpt/ \
             --dir-summary /mindspore/save/summary/
  shell:
      GPU: bash scripts/run_standalone_training_gpu.sh
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `--dir-ckpt / --dir-best-ckpt` by default.

```bash
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## [Eval process](#contents)

### Usage

You can start evaluation using python script. The usage of script is as follows:

- GPU: python eval.py --config configs/tnt_s.yaml --pretrained save/ckpt/.../svt_s-0.yaml

### Launch

```shell
# infer example
  python:
    GPU: python eval.py --config configs/svt_s.yaml --pretrained pretrained.ckpt --device-target GPU
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the following in `val.log`.

```bash
result: {'acc': 0.705} ckpt=/path/to/checkpoint/svt_s-0.ckpt
```

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --pretrained [CKPT_PATH] --tnt_config configs/svt_s.yaml --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`DEVICE` should be in ['GPU']
`FILE_FORMAT` should be in "MINDIR, ONNX"

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Twins-S                     |
| -------------------------- | ------------------------- |
| Model Version              | large                     |
| Resource                   | 4 * NV RTX 3090           |
| uploaded Date              |                           |
| MindSpore Version          | 1.8.0/1.9.0               |
| Dataset                    | ImageNet                  |
| Training Parameters        | configs/svt_s.yaml        |
| Optimizer                  | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy       |
| outputs                    | probability               |
| Loss                       | 2.0..3.0                  |
| Accuracy                   | ACC1[0.705]               |
| Total time                 | ~630 h                    |
| Params (M)                 | 24060776                  |
| Checkpoint for Fine tuning | 92 M                      |
| Scripts                    |  |

# [Description of Random Situation](#contents)

We use fixed seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).