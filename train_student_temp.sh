#!/bin/bash
TEACHER="resnet110"
STUDENT="resnet8"
TEACHER_1="resnet110"
STUDENT_1="resnet20"
TEACHER_2="vgg19"
STUDENT_2="vgg8"
A=0.9
B=0
A1=0
B1=0.7
A11=0
B11=0.7
A2=1
B2=0.7
DATE="1_30"
cd /home/lthpc/zhongzh/RFD4Hist
python "/home/lthpc/zhongzh/RFD4Hist/train_student.py" \
    --model_s $STUDENT \
    --path_t "/home/lthpc/zhongzh/RFD4Hist/save/models/${TEACHER}_ivygap_lr_0.005_decay_0.0005_trial_0/${TEACHER}_best.pth" \
    --distill kd \
    --learning_rate 0.005 \
    -r 0.1 \
    -a $A \
    -b $B \
    > logs_st/${DATE}_${TEACHER}_${STUDENT}_lr0.005_kd_a:${A}_b:${B}.log & 

python "/home/lthpc/zhongzh/RFD4Hist/train_student.py" \
    --model_s $STUDENT_1 \
    --path_t "/home/lthpc/zhongzh/RFD4Hist/save/models/${TEACHER_1}_ivygap_lr_0.005_decay_0.0005_trial_0/${TEACHER_1}_best.pth" \
    --distill kd \
    --learning_rate 0.005 \
    -r 0.1 \
    -a $A \
    -b $B \
    > logs_st/${DATE}_${TEACHER_1}_${STUDENT_1}_lr0.005_kd_a:${A}_b:${B}.log & 

python "/home/lthpc/zhongzh/RFD4Hist/train_student.py" \
    --model_s $STUDENT_2 \
    --path_t "/home/lthpc/zhongzh/RFD4Hist/save/models/${TEACHER_2}_ivygap_lr_0.005_decay_0.0005_trial_0/${TEACHER_2}_best.pth" \
    --distill kd \
    --learning_rate 0.005 \
    -r 0.1 \
    -a $A \
    -b $B \
    > logs_st/${DATE}_${TEACHER_2}_${STUDENT_2}_lr0.005_kd_a:${A}_b:${B}.log & 


# python "/home/lthpc/zhongzh/RFD4Hist/train_student.py" \
#     --path_t "/home/lthpc/zhongzh/RFD4Hist/save/models/${TEACHER}_ivygap_lr_0.005_decay_0.0005_trial_0/${TEACHER}_best.pth" \
#     --distill md_relation_pyr \
#     --model_s $STUDENT \
#     --learning_rate 0.005 \
#     -r 1 \
#     -a $A1 \
#     -b $B1 \
#     --feature --spatial \
#     > logs_st/${DATE}_${TEACHER}_${STUDENT}_lr0.005_RFD_a:${A1}_b:${B1}.log &

# python "/home/lthpc/zhongzh/RFD4Hist/train_student.py" \
#     --path_t "/home/lthpc/zhongzh/RFD4Hist/save/models/${TEACHER_1}_ivygap_lr_0.005_decay_0.0005_trial_0/${TEACHER_1}_best.pth" \
#     --distill md_relation_pyr \
#     --model_s $STUDENT_1 \
#     --learning_rate 0.005 \
#     -r 1 \
#     -a $A11 \
#     -b $B11 \
#     --feature --spatial \
#     > logs_st/${DATE}_${TEACHER_1}_${STUDENT_1}_lr0.005_RFD_a:${A11}_b:${B11}.log &

# python "/home/lthpc/zhongzh/RFD4Hist/train_student.py" \
#     --path_t "/home/lthpc/zhongzh/RFD4Hist/save/models/${TEACHER}_ivygap_lr_0.005_decay_0.0005_trial_0/${TEACHER}_best.pth" \
#     --distill md_relation_pyr \
#     --model_s $STUDENT \
#     --learning_rate 0.005 \
#     -r 1 \
#     -a $A2 \
#     -b $B2 \
#     --feature --spatial \
#     > logs_st/${DATE}_${TEACHER}_${STUDENT}_RFD+KD_a:${A2}_b:${B2}.log &  
