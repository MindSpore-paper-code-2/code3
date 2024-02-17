# 目录

<!-- TOC -->

- [目录](#目录)
- [SWEnet描述](#SWEnet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [评价指标](#评价指标)
    - [Kappa](#Kappa)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [Yellow River I II III-SAR上训练SWEnet](#Yellow River I II III-SAR上训练SWEnet)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SWEnet描述

## 概述

SWEnet基于变化检测网络的基础上引入使用边窗口增强模块，使得网络可以关注到更大范围与当前像素点相似的区域，增强了模型的保边降噪的能力。

## 论文:https://arxiv.org/pdf/1709.01507.pdf

[]()

# 模型架构

由孪生网络（1x1,3x3,5x5卷积）和se-resnet网络和SWE（side windows enhancement）模块组成。

# 数据集

使用的数据集：[Yellow River]

- 数据集大小：1.42MB，共4个类，每个类为一个文件夹，每个文件夹下有三张图片
- 数据格式：.bmp
    - 注：数据保存在dataset/Yellow River中。

使用的数据集：[Ottawa]

- 数据集介绍：该数据集是由RADARSAT SAR传感器在渥太华市上空采集的两个SAR图像的一部分。它们由渥太华的加拿大防研究与发展部提供。此数据集包括1997年5月和8月采集的两张影像，并显示了它们曾经遭受洪水侵袭的地区。并通过将先验信息与照片解释相结合而创建的图像和可用的基本事实。
- 数据集大小：290*350黑白图像
- 数据格式：.bmp
    - 注：数据保存在dataset/Ottawa中。

使用的数据集：[Red River-SAR]

- 数据集介绍：两张原始图像分别对应于非洪水和洪水，使用两张WRS-2 SAR图像进行实验，分别于1996年8月24日和1999年8月14日从越南红河拍摄。
- 数据集大小：512*512黑白图像
- 数据格式：.bmp
    - 注：数据保存在dataset/Red River-SAR中。

SARdataset:https://drive.google.com/file/d/1ziacqNiZLpwhQzQdIhzjk5jFQBk_Kvlx/view?usp=share_link

# 评价指标

## Kappa

采用[Kappa]作为模型训练评价指标。Kappa系数是一个用于一致性检验的指标，也可以用于衡量分类的效果。因为对于分类问题，所谓一致性就是模型预测结果和实际分类结果是否一致。kappa系数的计算是基于混淆矩阵的，取值为-1到1之间,通常大于0。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```yaml
  # 添加数据集路径
  dataroot:./dataset/
  val_data_path:./dataset/
  # 推理前添加checkpoint路径参数
  checkpoint_path:"./checkpoint/checkpoint_1_5149.ckpt"
  ```

  ```python
  # 运行训练示例
  python train.py > train.log 2>&1 &
  # 运行分布式训练示例
  bash scripts/run_train.sh  [DATA_PATH]
  # example: bash scripts/run_train.sh  ./dataset/
  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval.sh [DATA_PATH] [PATH_CHECKPOINT]
  # example: bash run_eval.sh ./dataset/ ./checkpoint/checkpoint_1_5149.ckpt
  ```

  对于分布式训练，也可以通过提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上训练数据集

      ```python
      # (1) 执行a或者b
      #       a. 在 config.yaml 文件中设置 "modelArts_mode=True"
      #          在 config.yaml 文件中设置 "dataroot='/cache/data/'"
      #          在 config.yaml 文件中设置 其他参数
      #       b. 在ModelArts网页上设置 "modelArts_mode=True"
      #          在ModelArts网页上设置 "dataroot='/cache/data/'"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到 S3 桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/path/SWEnet"
      # (4) 在ModelArts网页上设置启动文件为 "train.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

    - 在 ModelArts 上推理 liberty 数据集

      ```python
      # (1) 执行a或者b
      #       a. 在 config.yaml 文件中设置 "modelArts_mode=True"
      #          在 config.yaml 文件中设置 "dataroot='/cache/data/'"
      #          在 config.yaml 文件中设置 "checkpoint_path='/cache/checkpoint/'"
      #          在 config.yaml 文件中设置 "save_pred_path='/cache/pred/'"
      #          在 config.yaml 文件中设置 其他参数
      #       b. 在ModelArts网页上设置 "modelArts_mode=True"
      #          在ModelArts网页上设置 "dataroot='/cache/data/"
      #          在ModelArts网页上设置 "checkpoint_path='/cache/checkpoint/'"
      #          在ModelArts网页上设置 "save_pred_path='/cache/pred/'"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到 S3 桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/path/SWEnet"
      # (4) 在ModelArts网页上设置启动文件为 "eval.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

      其他数据集同理。

# 脚本说明

## 脚本及样例代码

```bash
├── SWEnet
    ├── model_utils
    │   ├──config.py                // 参数配置
    │   ├──device_adapter.py        // device adapter
    │   ├──local_adapter.py         // local adapter
    │   ├──hccl_tools.py            // hccl tools
    ├── scripts
    │   ├──run_train.sh             // 分布式到Ascend的shell脚本
    │   ├──run_eval.sh              // Ascend评估的shell脚本
    ├── src
    │   ├──model
    │   │   ├──SWEnet.py              // SWE模块
    │   │   ├──se.py              // se模块
    │   ├──dataset.py               // 数据处理
    │   ├──EvalMetrics.py           // 验证指标
    │   ├──Losses.py                // 损失函数
    ├── train.py                    // 训练脚本
    ├── eval.py                     // 评估脚本
    ├── export.py                   // 将checkpoint文件导出到air/mindir
    ├── README_CN.md                // 所有模型相关说明
    ├── requirements.txt                // 相关安装包
    ├── config.yaml             // 参数配置
```

## 脚本参数

在config.yaml中可以同时配置训练参数和评估参数。

  ```python
  'modelArts_mode': False    # 当使用model_arts云上环境，将其设置为True
  'is_distributed': False    # 进行分布式计算的时候，将其设置为True
  'lr': 0.001                   # 初始学习率
  'batch_size':64          # 训练批次大小
  'test_batch_size':1024     # 测试批次大小
  'epochs':15                # 总计训练epoch数
  'dataroot':'./data'        # 训练和评估数据集的绝对全路径
  'device_target':'Ascend'   # 运行设备
  'device_id':0              # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'ckpt_save_dir': "./checkpoint/" #存储权重路径
  'dataset_sink_mode': False #数据下沉
  'checkpoint_path':'./checkpoint/checkpoint_1_5149.ckpt'  # 推理时加载checkpoint文件的绝对路径
  ```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # train.log
    epoch: 1 step: 1, loss is 0.6920883655548096
    epoch: 1 step: 2, loss is 0.6916631460189819
    epoch: 1 step: 3, loss is 0.6910416483879089
    ...
    ...
    epoch: 15 step: 1, loss is 0.03258157894015312
    epoch: 15 step: 2, loss is 0.029147090390324593
    epoch: 15 step: 3, loss is 0.06458353996276855
  ```

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_train.sh ./dataset/
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  train_parallel0/log:epoch: 1 step: 1, loss is 0.6930392980575562
                      ...
                      epoch: 15 step: 1, loss is 0.06458353996276855
  train_parallel1/log:epoch:1, step: 1, loss is 0.6925740242004395
                      ...
                      epoch: 15 step: 1, loss is 0.03458353996276855
  ...
  ...
  train_parallel7/log:epoch:1, step: 1, loss is 0.6906054019927979
                      ...
                      epoch: 15 step: 1, loss is 0.0548625452354854
  ```

  训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估数据集

  在运行以下命令之前，请检查用于评估的检查点路径。例如“username/SWEnet/checkpoint/checkpoint_1_5149.ckpt”。

  ```bash
  python eval.py > eval.log 2>&1 &
  OR
  bash ./scripts/run_eval.sh ./dataset/ ./checkpoint/checkpoint_1_5149.ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
    ======================evaluate predicted===============
    ===evaluate====
    TN:82996.0 TP:4767.0 FP:780.0 FN:503.0 OE:1283.0 KC:0.873725754226216
    io===============
    ===evaluate====
    TN:83246.0 TP:4744.0 FP:530.0 FN:526.0 OE:1056.0 KC:0.8935455844226032
  ```

## 导出过程

### 导出MindIR

```shell
python export.py --ckpt_file=[CKPT_PATH] --file_format=[MINDIR, AIR]
```

# 模型描述

## 性能

### 训练性能

#### Yellow River I II III-SAR上训练SWEnet

- 在mindspore框架上使用训练的模型分别预测 IV-SAR,Ottawa,RedRiver-SAR。

  在执行预测命令之前，我们需要先修改参数。修改的项包括batch_size和save-path。

  预测的结果用kc值评估。

|参数|Ascend 910|
|------------------------------|------------------------------|
|模型版本|SWEnet|
|资源|Ascend 910；系统 Euler2.8|
|上传日期|2022-12-5|
|MindSpore版本|1.7.0|
|训练参数|epoch=15, steps per epoch=5149, batch_size = 64|
| 优化器 | Momentum|
|输出|概率|
|损失|0.00229|
|速度|50毫秒/步|
|总时长| 1p:64min 8p:15min|

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。