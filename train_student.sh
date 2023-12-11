#! /bin/zsh
python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
    --model_s resnet32 \
    --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
    --distill kd \
    -a 0.9 \
    -b 0 \
    --device cuda:4 \
    --device_id 4 5 7 \
    > 11_1_kd.log & \