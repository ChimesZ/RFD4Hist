#!/bin/bash
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher.py" \
#     --model ResNet50 \
#     --batch_size 64 \
#     --learning_rate 0.01 \
#     --dataset ivygap \
#     > 12_25_ResNet50_lr0.01_teacher.log &
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher.py" \
#     --model ResNet18 \
#     --batch_size 64 \
#     --learning_rate 0.1 \
#     --dataset ivygap \
#     > 12_25_ResNet18_lr0.1_teacher.log &
python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher.py" \
    --model vgg8 \
    --batch_size 64 \
    --learning_rate 0.01 \
    --dataset ivygap \
    > 12_27_vgg8_lr0.01_teacher.log & 
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher.py" \
#     --model MobileNetV2 \
#     --batch_size 64 \
#     --learning_rate 0.01 \
#     --dataset ivygap \
#     > 12_27_MobileNetV2_lr0.01_teacher.log &   
