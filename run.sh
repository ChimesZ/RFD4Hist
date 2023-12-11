# python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
#     --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
#     --device cuda:4 --device_id 4 \
#     --distill kd --model_s resnet32 -a 0.9 -b 0 --trial 1 \
#     > 12_5_kd_resnet32.log &
# # python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
# #     --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
# #     --device cuda:5 --device_id 5 \
# #     --distill kd --model_s resnet8 -a 0.9 -b 0 --trial 1\
# #     > 12_5_kd_resnet32.log &
# python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
#     --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
#     --device cuda:5 --device_id 5 \
#     --distill md_relation_pyr --model_s resnet32 -a 0 -b 0.3 --feature --spatial --trial 1 \
#     > 12_5_RFD_0.3_resnet32.log & 
# python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
#     --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
#     --device cuda:6 --device_id 6 \
#     --distill md_relation_pyr --model_s resnet32 -a 0 -b 0.5 --feature --spatial --trial 1 \
#     > 12_5_RFD_0.5_resnet32.log & 
# python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
#     --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
#     --device cuda:7 --device_id 7 \
#     --distill md_relation_pyr --model_s resnet32 -a 0 -b 0.7 --feature --spatial --trial 1 \
#     > 12_5_RFD_0.7_resnet32.log & 
# TODO alpha -> 0

python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
    --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
    --device cuda:4 --device_id 4 \
    --distill kd --model_s resnet32 -a 0.1 -b 0.3 --trial 1 \
    > 12_6_RFD_a:0.1_resnet32.log &
python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
    --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
    --device cuda:5 --device_id 5 \
    --distill md_relation_pyr --model_s resnet32 -a 0.3 -b 0.3 --feature --spatial --trial 1 \
    > 12_6_RFD_a:0.3_resnet32.log & 
python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
    --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
    --device cuda:6 --device_id 6 \
    --distill md_relation_pyr --model_s resnet32 -a 0.5 -b 0.3 --feature --spatial --trial 1 \
    > 12_6_RFD_a:0.5_resnet32.log & 
python "/home/zhong/Experiment/RFD_base_crd/train_student.py" \
    --path_t "/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/resnet110_best.pth" \
    --device cuda:7 --device_id 7 \
    --distill md_relation_pyr --model_s resnet32 -a 0.7 -b 0.3 --feature --spatial --trial 1 \
    > 12_6_RFD_a:0.7_resnet32.log & 
