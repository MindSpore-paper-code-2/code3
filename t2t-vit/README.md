# Contents

<!-- TOC -->

* [Contents](#contents)
* [T2T-ViT Description](#t2t-vit-description)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Environment Requirements](#environment-requirements)
* [Quick Start](#quick-start)
    * [Prepare the model](#prepare-the-model)
    * [Run the scripts](#run-the-scripts)
* [Script Description](#script-description)
    * [Script and Sample Code](#script-and-sample-code)
        * [Directory structure](#directory-structure)
        * [Script Parameters](#script-parameters)
    * [Training Process](#training-process)
        * [Training on GPU](#training-on-gpu)
            * [Training on multiple GPUs](#training-on-multiple-gpus)
            * [Training on single GPU](#training-on-single-gpu)
            * [Arguments description](#arguments-description)
        * [Training with CPU](#training-with-cpu)
        * [Transfer training](#transfer-training)
    * [Evaluation](#evaluation)
        * [Evaluation process](#evaluation-process)
            * [Evaluation with checkpoint](#evaluation-with-checkpoint)
        * [Evaluation results](#evaluation-results)
    * [Inference](#inference)
        * [Inference with checkpoint](#inference-with-checkpoint)
        * [Inference results](#inference-results)
    * [Export](#export)
        * [Export process](#export-process)
        * [Export results](#export-results)
* [Model Description](#model-description)
    * [Performance](#performance)
        * [Training Performance](#training-performance)
* [Description of Random Situation](#description-of-random-situation)
* [ModelZoo Homepage](#modelzoo-homepage)

<!-- TOC -->

# [T2T-ViT Description](#contents)

Transformers, which are popular for language modeling, have been explored for
solving vision tasks recently, e.g., the Vision Transformer (ViT) for image
classification. The ViT model splits each image into a sequence of tokens with
fixed length and then applies multiple Transformer layers to model their global
relation for classification. However, ViT achieves inferior performance to
CNNs when trained from scratch on a midsize dataset like ImageNet.
We find it is because:
1) the simple tokenization of input images fails to model the important
   local structure such as edges and lines among neighboring pixels,
   leading to low training sample efficiency;
2) the redundant attention backbone design of ViT leads to limited feature
   richness for fixed computation budgets and limited training samples.

To overcome such limitations, we propose a new Tokens-To-Token
Vision Transformer (T2T-ViT), which incorporates
1) a layer wise Tokens-to-Token (T2T) transformation to progressively
   structurize the image to tokens by recursively aggregating neighboring
   Tokens into one Token (Tokens-to-Token), such that local structure
   represented by surrounding tokens can be modeled and tokens length can be reduced;
2) an efficient backbone with a deep-narrow structure for vision transformer
   motivated by CNN architecture design after empirical study. Notably,
   T2T-ViT reduces the parameter count and MACs of vanilla ViT by half,
   while achieving more than 3.0% improvement when trained from scratch on
   ImageNet. It also outperforms ResNets and achieves comparable performance
   with MobileNets by directly training on ImageNet. For example, T2T-ViT
   with comparable size to ResNet50 (21.5M parameters) can achieve 83.3%
   top1 accuracy in image resolution 384×384 on ImageNet.

[Paper](https://arxiv.org/pdf/2101.11986.pdf): ZLi Yuan, Yunpeng Chen,
Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis E.H. Tay, Jiashi Feng,
Shuicheng Yan. 2021.

# [Model Architecture](#contents)

T2T-ViT consists of two main components:
1) a layer-wise “Tokens-to-Token module” (T2T module) to model the local structure information of the image and reduce the length of tokens progressively;
2) an efficient “T2T-ViT backbone” to draw the global attention relation on tokens from the T2T module.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [ImageNet2012](http://www.image-net.org/)

* Dataset size：146.6G
    * Train：139.3G，1281167 images
    * Val：6.3G，50000 images
    * Annotations：each image is in label folder
* Data format：images sorted by label folders
    * Note：Data will be processed in imagenet.py

# [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset ImageNet dataset.
* We use ImageNet2012 as training dataset in this example by default, and you
  can also use your own datasets.

For ImageNet-like dataset the directory structure is as follows:

```shell
 .
 └─imagenet
   ├─train
     ├─class1
       ├─image1.jpeg
       ├─image2.jpeg
       └─...
     ├─...
     └─class1000
   ├─val
     ├─class1
     ├─...
     └─class1000
   └─test
```

# [Quick Start](#contents)

## Prepare the model

1. Chose the model by changing the `arch` in `configs/t2t_vit_.yaml`, `XXX` is the corresponding model architecture configuration.
   Allowed options are: `t2t_vit_19`, `t2t_vit_t_14`.
2. Change the dataset config in the corresponding config. `configs/t2t_vit_XXX.yaml`.
   Especially, set the correct path to data.
3. Change the hardware setup.
4. Change the artifacts setup to set the correct folders to save checkpoints and mindinsight logs.

Note, that you also can pass the config options as CLI arguments, and they are
preferred over config in YAML.
Also, all possible options must be defined in `yaml` config file.

## Run the scripts

After installing MindSpore via the official website,
you can start training and evaluation as follows.

```shell
# distributed training on GPU
bash run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# standalone training on GPU
bash run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# run eval on GPU
bash run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

### Directory structure

```shell
t2t-vit
├── scripts
│   ├── run_distribute_train_gpu.sh                          # shell script for distributed training on GPU
│   ├── run_eval_gpu.sh                                      # shell script for evaluation on GPU
│   ├── run_infer_gpu.sh                                     # shell script for inference on GPU
│   └── run_standalone_train_gpu.sh                          # shell script for training on GPU
├── src
│  ├── configs
│  │  ├── t2t_vit_19.yaml                                    # example of configuration for T2T-ViT-19
│  │  └── t2t_vit_t_14.yaml                                  # example of configuration for T2T-ViT-t-14
│  ├── data
│  │  ├── augment
│  │  │  ├── __init__.py
│  │  │  ├── auto_augment.py                                 # augmentation set builder
│  │  │  ├── mixup.py                                        # MixUp augmentation
│  │  │  └── random_erasing.py                               # Random Erasing augmentation
│  │  ├── __init__.py
│  │  └── imagenet.py                                        # wrapper for reading ImageNet dataset
│  ├── layers                                                # layers used in T2T-ViT implementation
│  │  ├── __init__.py
│  │  ├── attention.py                                       # Attention layer
│  │  ├── drop_path_timm.py                                  # Implementation of drop path the same way as in TIMM
│  │  ├── mlp.py                                             # MLP block implementation
│  │  ├── t2t_module.py                                      # Tokens-to-token block
│  │  ├── token_performer.py                                 # Token performer block
│  │  ├── token_transformer.py                               # Token transformer block
│  │  ├── transformer_block.py                               # Transformer block implementation
│  │  └── unfold.py                                          # Custom implementation of Unfold block
│  ├── tools
│  │  ├── __init__.py
│  │  ├── callback.py                                        # callback functions (implementation)
│  │  ├── cell.py                                            # tune model layers/parameters
│  │  ├── criterion.py                                       # model training objective function (implementation)
│  │  ├── get_misc.py                                        # initialize optimizers and other arguments for training process
│  │  ├── metrics.py                                         # wrapper class for metric
│  │  ├── optimizer.py                                       # model optimizer function (implementation)
│  │  └── schedulers.py                                      # training (LR) scheduling function (implementation)
│  ├── trainer
│  │  ├── __init__.py
│  │  ├── ema.py                                             # EMA implementation
│  │  ├── train_one_step_with_ema.py                         # utils for training with EMA
│  │  └── train_one_step_with_scale_and_clip_global_norm.py  # utils for training with gradient clipping
│  ├── __init__.py
│  ├── config.py                                             # YAML and CLI configuration parser
│  ├── utils.py                                              # Auxiliary functions for T2T-ViT architecture
│  └── t2t_vit.py                                            # T2T-ViT architecture
├── eval.py                                                  # evaluation script
├── export.py                                                # export checkpoint files into MINDIR and AIR formats
├── infer.py                                                 # inference script
├── README.md                                                # T2T-ViT descriptions
├── requirements.txt                                         # python requirements
└── train.py                                                 # training script
```

### [Script Parameters](#contents)

```yaml
# ===== Dataset ===== #
dataset: ImageNet
data_url: /data/imagenet/ILSVRC/Data/CLS-LOC/
train_dir: train
val_dir: validation_preprocess
train_num_samples: -1
val_num_samples: -1

# ===== Augmentations ==== #
auto_augment: rand-m9-mstd0.5-inc1
aa_interpolation: bilinear
re_mode: pixel
re_prob: 0.25
re_count: 1
cutmix: 1.0
mixup: 0.8
mixup_prob: 1.0
mixup_mode: batch
mixup_off_epoch: 0.0
switch_prob: 0.5
label_smoothing: 0.1
min_crop: 0.08
crop_pct: 0.9

# ===== Optimizer ======== #
optimizer: adamw
beta: [ 0.9, 0.999 ]
eps: 1.0e-8
base_lr: 0.001
min_lr: 1.0e-5
lr_scheduler: cosine_lr
lr_adjust: 30
lr_gamma: 0.97
momentum: 0.9
weight_decay: 0.05


# ===== Network training config ===== #
epochs: 300
batch_size: 100
is_dynamic_loss_scale: True
loss_scale: 1024
num_parallel_workers: 8
start_epoch: 0
warmup_length: 2
warmup_lr: 0.000007
# Gradient clipping
use_clip_grad_norm: True
clip_grad_norm: 1.0
# Load pretrained setup
exclude_epoch_state: True
seed: 0
# EMA
with_ema: False
ema_decay: 0.9999

pynative_mode: False
dataset_sink_mode: True

# ==== Model arguments ==== #
arch: t2t_vit_19
amp_level: O0
file_format: MINDIR
pretrained: ''
image_size: 224
num_classes: 1000
drop: 0.0
drop_block: 0.0
drop_path: 0.1
disable_approximate_gelu: False

# ===== Hardware setup ===== #
device_id: 0
device_num: 1
device_target: GPU

# ===== Callbacks setup ===== #
summary_root_dir: /experiments/summary_dir/
ckpt_root_dir: /experiments/checkpoints/
best_ckpt_root_dir: /experiments/best_checkpoints/
logs_root_dir: /experiments/logs/
ckpt_keep_num: 10
best_ckpt_num: 5
ckpt_save_every_step: 0
ckpt_save_every_seconds: 1800
print_loss_every: 100
summary_loss_collect_freq: 20
model_postfix: 0
collect_input_data: False
dump_graph: False
```

## [Training Process](#contents)

In the examples below the only required argument is YAML config file.

### Training on GPU

#### Training on multiple GPUs

Usage

```shell
# distributed training on GPU
run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments
bash run_distribute_train.sh ../src/configs/t2t_vit_19.yaml --num_devices 4 --device_ids 0,1,2,3

# With extra arguments
bash run_distribute_train.sh ../src/configs/t2t_vit_19.yaml --num_devices 4 --device_ids 0,1,2,3 --extra --amp_level O0 --batch_size 48 --start_epoch 0 --num_parallel_workers 8
```

#### Training on single GPU

Usage

```shell
# standalone training on GPU
run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments:
bash run_standalone_train.sh ../src/configs/t2t_vit_19.yaml --device 0
# With extra arguments:
bash run_standalone_train.sh ../src/configs/t2t_vit_19.yaml --device 0 --extra --amp_level O0 --batch_size 48 --start_epoch 0 --num_parallel_workers 8
```

Running the Python scripts directly is also allowed.

```shell
# show help with description of options
python train.py --help

# standalone training on GPU
python train.py --config_path path/to/config.yaml [OTHER OPTIONS]
```

#### Arguments description

`bash` scripts have the following arguments

* `CONFIG`: path to YAML file with configuration.
* `--num_devices`: the device number for distributed train.
* `--device_ids`: ids of devices to train.
* `--checkpoint`: path to checkpoint to continue training from.
* `--extra`: any other arguments of `train.py`.

By default, training process produces four folders (configured):

* Best checkpoints
* Current checkpoints
* Mindinsight logs
* Terminal logs

### Training with CPU

**It is recommended to run models on GPU.**

### Transfer training

You can train your own model based on pretrained classification
model. You can perform transfer training by following steps.

1. Convert your own dataset to ImageFolderDataset style. Otherwise, you have to add your own data preprocess code.
2. Change `t2t_vit_XXX.yaml` according to your own dataset, especially the `num_classes`.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by `--pretrained` argument.
4. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation](#contents)

### Evaluation process

#### Evaluation with checkpoint

Usage

```shell
run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Examples

```shell

# Without extra args
bash run_eval_gpu.sh  ../src/configs/t2t_vit_19.yaml --checkpoint /data/models/t2t_vit_19.ckpt

# With extra args
bash run_eval_gpu.sh  ../src/configs/t2t_vit_19.yaml --checkpoint /data/models/t2t_vit_19.ckpt --extra --data_url /data/imagenet/ --val_dir validation_preprocess
```

Running the Python script directly is also allowed.

```shell
# run eval on GPU
python eval.py --config_path path/to/config.yaml [OTHER OPTIONS]
```

The Python script has the same arguments as the training script (`train.py`),
but it uses only validation subset of dataset to evaluate.
Also, `--pretrained` is expected.

### Evaluation results

Results will be printed to console.

```shell
# checkpoint evaluation result
eval results: {'Loss': 0.8323014795482159, 'Top1-Acc': 0.81674, 'Top5-Acc': 0.95638}
```

## [Inference](#contents)

Inference may be performed with checkpoint or ONNX model.

### Inference with checkpoint

Usage

```shell
run_infer_gpu.sh DATA [--checkpoint CHECKPOINT] [--arch ARCHITECTURE] [--output OUTPUT_JSON_FILE (default: predictions.json)]
```

Example for folder

```shell
bash run_infer_gpu.sh /data/images/cheetah/ --checkpoint /data/models/t2t_vit_19.ckpt --arch t2t_vit_19
```

Example for single image

```shell
bash run_infer_gpu.sh /data/images/American\ black\ bear/ILSVRC2012_validation_preprocess_00011726.JPEG --checkpoint /data/models/t2t_vit_19.ckpt --arch t2t_vit_19
```

### Inference results

Predictions will be output in logs and saved in JSON file. File content is
dictionary where key is image path and value is class number. It's supported
predictions for folder of images (png, jpeg file in folder root) and single image.

Results for single image in console

```shell
/data/images/cheetah/ILSVRC2012_validation_preprocess_00011726.JPEG (class: 295)
```

Results for single image in JSON file

```json
{
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00011726.JPEG": 295
}
```

Results for directory in console

```shell
/data/images/American black bear/ILSVRC2012_validation_preprocess_00011726.JPEG (class: 295)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00014576.JPEG (class: 294)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00000865.JPEG (class: 295)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00007093.JPEG (class: 295)
/data/images/American black bear/ILSVRC2012_validation_preprocess_00014029.JPEG (class: 295)

```

Results for directory in JSON file

```json
{
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00011726.JPEG": 295,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00014576.JPEG": 294,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00000865.JPEG": 295,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00007093.JPEG": 295,
 "/data/images/American black bear/ILSVRC2012_validation_preprocess_00014029.JPEG": 295
}
```

## [Export](#contents)

### Export process

Trained checkpoints may be exported to `MINDIR` and `AIR` (currently not checked).

Usage

```shell
python export.py --config path/to/config.yaml --file_format FILE_FORMAT --pretrained path/to/checkpoint.ckpt --arch ARCHITECTURE_NAME
```

Example

```shell

# Export to MINDIR
python export.py --config src/configs/t2t_vit_19.yaml --file_format MINDIR --pretrained /data/models/t2t_vit_19.ckpt --arch t2t_vit_19
```

### Export results

Exported models saved in the current directory with name the same as architecture.

# [Model Description](#contents)

## Performance

### Training Performance

| Parameters                 | GPU                             |
|----------------------------|---------------------------------|
| Model Version              | T2T-ViT-19                      |
| Resource                   | 4xGPU (NVIDIA GeForce RTX 3090) |
| Uploaded Date              | 07/12/2023 (month/day/year)     |
| MindSpore Version          | 1.9.0                           |
| Dataset                    | ImageNet                        |
| Training Parameters        | src/configs/t2t_vit_19.yaml     |
| Optimizer                  | AdamW                           |
| Loss Function              | SoftmaxCrossEntropy             |
| Outputs                    | logits                          |
| Accuracy                   | ACC1 [0.451]                    |
| Total time                 | ~138.5 h                        |
| Params                     | 38638315                        |
| Checkpoint for Fine tuning | 457.3 M                         |
| Scripts                    |                                 |

| Parameters                 | GPU                               |
|----------------------------|-----------------------------------|
| Model Version              | T2T-ViT-19                        |
| Resource                   | 1xGPU (NVIDIA GeForce RTX 3090)   |
| Uploaded Date              | 07/12/2023 (month/day/year)       |
| MindSpore Version          | 1.9.0                             |
| Dataset                    | ImageNet                          |
| Training Parameters        | src/configs/t2t_vit_19_cifar.yaml |
| Optimizer                  | AdamW                             |
| Loss Function              | SoftmaxCrossEntropy               |
| Outputs                    | logits                            |
| Accuracy                   | ACC1 [0.985]                      |
| Total time                 | ~6.6 h                            |
| Params                     | 38638315                          |
| Checkpoint for Fine tuning | 446.6 M                           |
| Scripts                    |                                   |

| Parameters                 | GPU                             |
|----------------------------|---------------------------------|
| Model Version              | T2T-ViT-t14                     |
| Resource                   | 4xGPU (NVIDIA GeForce RTX 3090) |
| Uploaded Date              | 07/12/2023 (month/day/year)     |
| MindSpore Version          | 1.9.0                           |
| Dataset                    | ImageNet                        |
| Training Parameters        | src/configs/t2t_vit_t_14.yaml   |
| Optimizer                  | AdamW                           |
| Loss Function              | SoftmaxCrossEntropy             |
| Outputs                    | logits                          |
| Accuracy                   | ACC1 [0.795]                    |
| Total time                 | ~692.66 h                       |
| Params                     | 21082347                        |
| Checkpoint for Fine tuning | 251.1 M                         |
| Scripts                    |                                 |

# [Description of Random Situation](#contents)

We use fixed seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
