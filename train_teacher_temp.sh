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
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet8 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/12_28_resnet8_lr0.005_teacher.log & 
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet32 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/12_28_resnet32_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model vgg8 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/12_28_vgg8_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet32x4 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/12_29_resnet32x4_lr0.005_teacher.log &
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model wrn_40_2 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
    #  > logs/1_2_wrn_40_2_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model ShuffleV2 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/12_29_wrn_shuffleV2_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model MobileNetV2 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/12_31_MobileNetV2_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model ShuffleV1 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_2_ShuffleV1_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model ResNet18 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/12_31_ResNet18_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model ResNet50 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_1_ResNet50_lr0.005_teacher.log &   

# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model vgg8 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_5_vgg8_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model vgg19 \
#     --batch_size 64 \
#   i  --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_5_vgg19_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet110 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_9_resnet110_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet32 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_10_resnet32_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet8 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_12_resnet8_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet20 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_12_resnet20_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet32x4 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_17_resnet32x4_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model resnet8x4 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_17_resnet8x4_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model ShuffleV1 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_17_ShuffleV1_lr0.005_teacher.log &   
# python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
#     --model ShuffleV2 \
#     --batch_size 64 \
#     --learning_rate 0.005 \
#     --dataset ivygap \
#     > logs/1_19_ShuffleV2_lr0.005_teacher.log &   
python -u "/home/lthpc/zhongzh/RFD4Hist/train_teacher_temp.py" \
    --model resnet32 \
    --batch_size 128 \
    --learning_rate 0.005 \
    --dataset ivygap_norm \
    --device cuda:0 \
    --device_id 0 \
    --epochs 200 \
    > log_norm/1_30_resnet32_lr0.005_teacher_norm.log &   
