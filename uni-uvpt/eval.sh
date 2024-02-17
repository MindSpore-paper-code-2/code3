#!/bin/bash

python test.py  configs/to_cityscapes_swin_daformer_prompt.py \
                checkpoints/GtA_Swin-B_GTAV_Cityscapes_Uni-UVPT_LP_56.9.ckpt  \
               --eval mIoU \
               --data_url /home/ma-user/work/datasets/Cityscapes 

python test.py  configs/to_cityscapes_swin_daformer_prompt.py \
                checkpoints/GtA_Swin-B_Synthia_Cityscapes_Uni-UVPT_LP_53.8_60.4.ckpt  \
               --eval mIoU \
               --data_url /home/ma-user/work/datasets/Cityscapes 

python test.py  configs/to_cityscapes_swin_daformer_prompt.py \
                checkpoints/SSS_Swin-B_GTAV_Cityscapes_Uni-UVPT_LP_56.2.ckpt  \
               --eval mIoU \
               --data_url /home/ma-user/work/datasets/Cityscapes 

python test.py  configs/to_cityscapes_swin_daformer_prompt.py \
                checkpoints/SSS_Swin-B_Synthia_Cityscapes_Uni-UVPT_LP_52.6_59.4.ckpt  \
               --eval mIoU \
               --data_url /home/ma-user/work/datasets/Cityscapes 