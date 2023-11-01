代码中：
先看train_student.py里面主要看 md_relation_pyr 对应的部分，opt.feature 代表使用S-RFD过程, opt.spatial 代表使用R-RFD过程， SampleAttentionModule对应 S-RFD模块，SpatialAttentionModule对应 R-RFD模块。老师网络已经预训练好了，不用再训练，直接可以load参数，这里只放了resnet56的teacher参数。我们主要通过train_student.py训练学生网络的性能。

python命令：看run.sh中的运行命令
```
FitNet
python train student.py --path t ./save/models/resnet32x4 vanilla/ckpt epoch 240.pth --distill hint --model s resnet8x4 -a 0 -h 100 --trial 1#AT
--distill attention --model s resnet8x4 -a  -b 1000--trial 1python train student.py --path t ./save/models/resnet32x4vanilla/ckpt epoch 240.pth
#SP
-distill similaritypython train student.py--path t .save/models/resnet32epoch 240.pth--model s resnet8x4 -a  -b 3000--trial 1#CC
python train student.py--path t ./save/models/resnet32x4vanilla/ckpt epoch 240.pth-distill correlation --model s resnet8x4 -a 0 -b 0.02 --trial 1
# VID
python train student.py --path t./save/models/resnet32x--distill vid --model s resnet8x4 -a 0 -b 1 --trial 1anilla/ckptepoch 240.pth
#RKDpython train student.py --path t ./save/models/resnet32x4 vanilla/ckpt epoch 240.pth--model s resnet8x4 -a  -b 1 --trial 1--distill rkd# PKT
python train student.py --path t.save/models/resnet32xvanilla/ckptepoch 240.pth-distill pkt-model s resnet8x4 -a  -b 30000--trial 1#ABpython train student.py--path t-distill abound--model s resnet8x4 -a 0 -b 1 --trial 1save240.pth
#FT
python train student.py --path t-distill factor --model s resnet8x4 -a 0 -b 200 --trial 1vanilla/ckpt epoch 240.pthsave/models/resnet32x# FSPpython train student.py./save/models/resnet32-distill fsp-model s resnet8x4 -a  -b 50 --trial 1--path tepoch 240.pth
井NST
python train student.py --path t ./save/models/resnet32x4 vanilla/ckpt epoch 240.pth-distill nst-model s resnet8x4 -a 0 -b 50 --trial 1#CRD
python train student.py --path t ./save/models/resnet32x4--distill crd --model s resnet8x4 -a  -b 0.8 --trial 1vanilla/ckpt epoch 240.pth
# CRD+KD
python train student.py --path t ./save/models/resnet32x4 vanilla/ckpt epoch 240.pth--distill crd --model s resnet8x4 -a 1 -b 0.8 --trial 1
```
