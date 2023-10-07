代码中：
先看train_student.py里面主要看 md_relation_pyr 对应的部分，opt.feature 代表使用S-RFD过程, opt.spatial 代表使用R-RFD过程， SampleAttentionModule对应 S-RFD模块，SpatialAttentionModule对应 R-RFD模块。老师网络已经预训练好了，不用再训练，直接可以load参数，这里只放了resnet56的teacher参数。我们主要通过train_student.py训练学生网络的性能。

python命令：看run.sh中的运行命令