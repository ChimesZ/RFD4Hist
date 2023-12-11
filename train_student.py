"""
the general training framework
"""

from __future__ import print_function
from email.policy import strict

import os
import argparse
import socket
import time
from aiohttp import TraceResponseChunkReceivedParams


import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger
# TUNE_DISABLE_AUTO_CALLBACK_LOGGERS=1
# torch.cuda.current_device()
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '4'

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser, SampleAttentionModule, SpatialAttentionModule, Tofd

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.ivygap import get_GAP_dataloaders

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss, AFD
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init
import numpy as np

def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='ivygap_5', choices=['cifar100','ivygap', 'ivygap_5', 'ivygap_6'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet32',  # Resnet32 
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default="/home/zhong/Experiment/RFD_base_crd/save/models/resnet110_ivygap_5_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth", help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', # kd md_relation_pyr hint rkd
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst', 'md_relation_pyr', 'tofd','afd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=0.1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # relation feature distillation
    parser.add_argument('--gamma_1', default=0.1, type=float)
    parser.add_argument('--gamma_2', default=0.1, type=float)
    parser.add_argument('--theta_1', default=0.05, type=float)
    parser.add_argument('--theta_2', default=0.01, type=float)
    parser.add_argument('--feature', action='store_true')
    parser.add_argument('--spatial', action='store_true')
    parser.add_argument('--num_stage', default=3, type=int)

    # tofd 
    parser.add_argument('--alpha_tofd', default=0.05, type=float)
    parser.add_argument('--beta_tofd', default=0.4, type=float)

    # AFD
    parser.add_argument('--qk_dim', default=128, type=int)
    
    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # Device 
    parser.add_argument('--device', type=str, default=None, help='cuda:number')
    parser.add_argument('--device_id', nargs='+', type=int, default=[4, 5, 6, 7])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls, opt):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'], strict=False)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path, map_location=torch.device(opt.device))['model'].items()}) #Because parallel was used in training
    # model = torch.load(model_path)
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    if opt.dataset == 'ivygap_5': 
        train_loader, val_loader, n_data = get_GAP_dataloaders(batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers,
                                                               is_instance=True)
        n_cls = 5
    else:
        raise NotImplementedError(opt.dataset)

    # model

    model_t = load_teacher(opt.path_t, n_cls, opt)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model_t = model_t.to(device)
        model_t = nn.DataParallel(model_t, device_ids=opt.device_id)
        model_s = model_s.to(device)
        model_s = nn.DataParallel(model_s, device_ids=opt.device_id)
    data = torch.randn(2, 3, 150, 150).to(device)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    elif opt.distill == 'md_relation_pyr':
        criterion_kd = None
        num_auxiliary_classifier = 3
        link_1 = []
        link_2 = []
        t_feat_list_1 = []
        feat_list_1 = []
        t_feat_list_2 = []
        feat_list_2 = []
        if opt.feature:
            for m in range(1,opt.num_stage+1):
                t = torch.nn.functional.adaptive_avg_pool2d(feat_t[m], 1)
                s = torch.nn.functional.adaptive_avg_pool2d(feat_s[m], 1)
                t_feat_list_1.append(t)
                feat_list_1.append(s)
            for m in range(0,opt.num_stage):
                t_feat = t_feat_list_1[m].view(t_feat_list_1[m].size(0), -1)
                s_feat = feat_list_1[m].view(feat_list_1[m].size(0), -1)
                teacher_feature_size = t_feat.size(1)
                student_feature_size = s_feat.size(1)
                link_1.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
            SAM = SampleAttentionModule(nn.ModuleList(link_1), opt, opt.num_stage)
            module_list.append(SAM)
            trainable_list.append(SAM)

        if opt.spatial:
            for m in range(1,opt.num_stage+1):
                t = torch.nn.functional.adaptive_avg_pool2d(feat_t[m], 4)
                s = torch.nn.functional.adaptive_avg_pool2d(feat_s[m], 4)
                t_feat_list_2.append(t)
                feat_list_2.append(s)
            for m in range(0,opt.num_stage):
                t_feat = t_feat_list_2[m].view(t_feat_list_2[m].size(0), -1)
                s_feat = feat_list_2[m].view(feat_list_2[m].size(0), -1)
                teacher_feature_size = t_feat.size(1)
                student_feature_size = s_feat.size(1)
                link_2.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
            SPAM = SpatialAttentionModule(nn.ModuleList(link_2), opt, opt.num_stage)
            module_list.append(SPAM)
            trainable_list.append(SPAM)
    elif opt.distill == 'tofd':
        criterion_kd = None
        teacher_feature_size = feat_t[0].size(1)
        student_feature_size = feat_s[0].size(1)
        link = []
        for j in range(opt.num_stage):
            link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
        tofd = Tofd(nn.ModuleList(link), opt, opt.num_stage, opt.alpha_tofd, opt.beta_tofd)
        module_list.append(tofd)
        trainable_list.append(tofd)
    elif opt.distill == 'afd':
        LAYER = {'resnet20': np.arange(1, (20 - 2) // 2 + 1),  # 9
         'resnet56': np.arange(1, (56 - 2) // 2 + 1),  # 27
         'resnet110': np.arange(2, (110 - 2) // 2 + 1, 2),  # 27
         'wrn40x2': np.arange(1, (40 - 4) // 2 + 1),  # 18
         'wrn28x2': np.arange(1, (28 - 4) // 2 + 1),  # 12
         'wrn16x2': np.arange(1, (16 - 4) // 2 + 1),  # 6
         'resnet34': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet34im': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18im': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet32': np.arange(1, (32 - 2) // 2 + 1),  # 15
         }
        opt.guide_layers = LAYER[opt.model_t]
        opt.hint_layers = LAYER[opt.model_s]
        opt.s_shapes = [feat_s[i].size() for i in opt.hint_layers]
        opt.t_shapes = [feat_t[i].size() for i in opt.guide_layers]
        opt.n_t, opt.unique_t_shapes = unique_shape(opt.t_shapes)
        criterion_kd = AFD(opt)
        module_list.append(criterion_kd)
        trainable_list.append(criterion_kd)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        # model_t = model_t.to(device)
        # model_t = nn.DataParallel(model_t, device_ids=opt.device_id)
        # model_s = model_s.to(device)
        # model_s = nn.DataParallel(model_s, device_ids=opt.device_id)
        # criterion = criterion.to(device)
        module_list.to(device)
        trainable_list.to(device)
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            # torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    # torch.save(state, save_file)


if __name__ == '__main__':
    main()
