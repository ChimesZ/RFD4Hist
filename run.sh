# CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s wrn_16_2 -a 0.9 -b 1 --trial 1 --feature --spatial > 8_12_1.log &
# CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s wrn_40_1 -a 0.9 -b 1 --trial 1 --feature --spatial > 8_12_2.log &
# CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s resnet20 -a 0.9 -b 1 --trial 1 --feature --spatial > 8_12_3.log &
# CUDA_VISIBLE_DEVICES=5 python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s resnet20 -a 0.9 -b 1 --trial 1 --feature --spatial > 8_12_4.log &
# CUDA_VISIBLE_DEVICES=6 python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s resnet32 -a 0.9 -b 1 --trial 1 --feature --spatial > 8_12_5.log &
# CUDA_VISIBLE_DEVICES=7 python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s resnet8x4 -a 0.9 -b 1 --trial 1 --feature --spatial > 8_12_6.log &
# CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s vgg8 -a 0.9 -b 1 --trial 1 --feature --spatial > 8_12_7.log &
# CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s MobileNetV2 -a 0.9 -b 1 --trial 1 --feature --spatial --learning_rate 0.01 > 8_10_8.log &
# CUDA_VISIBLE_DEVICES=0 python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s vgg8 -a 0 -b 1 --trial 1 --feature --spatial > 8_13_1.log &
# CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s MobileNetV2 -a 0 -b 1 --trial 1 --feature --spatial > 8_13_2.log &
# CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s MobileNetV2 -a 0 -b 1 --trial 1 --feature --spatial > 8_13_3.log &
# CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s vgg8 -a 0 -b 1 --trial 1 --feature --spatial > 8_13_4.log & 
# CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s vgg8 -a 0.1 -b 1 --trial 1 --feature --spatial > 8_18_1.log &
# CUDA_VISIBLE_DEVICES=6 python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill md_relation_pyr --model_s vgg8 -a 0.1 -b 1 --trial 1 --feature --spatial > 8_18_2.log & 
# CUDA_VISIBLE_DEVICES=0 python train_teacher.py --model vgg16 > 1_11.log &
python '/home/zhong/Experiment/RFD_base_crd/train_teacher.py' > 10_29.log &
