from __future__ import print_function, division

import sys
import time
from sqlalchemy import false
import torch
import pickle

from .util import AverageMeter, accuracy




def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            input = input.to(device)
            target = target.to(device)

        # ===================forward=====================
        _,output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    sum_spkd_loss_1 = 0.0
    sum_spkd_loss_2 = 0.0
    sum_spkd_pixel_loss_1 = 0.0
    sum_spkd_pixel_loss_2 = 0.0
    for idx, data in enumerate(train_loader):
        length = len(train_loader)
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            input = input.to(device)
            target = target.to(device)
            index = index.to(device)
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.to(device)
        # if torch.cuda.is_available():
        #     input = input.cuda()
        #     target = target.cuda()
        #     index = index.cuda()
        #     if opt.distill in ['crd']:
        #         contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact, label=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact, label=True)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        if opt.distill == 'tofd':
            loss_cls = criterion_cls(logit_s[-1], target)
            loss_div = criterion_div(logit_s[-1], logit_t[-1])
        else: 
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.tensor([0])
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        elif opt.distill == 'md_relation_pyr':
            if opt.spatial and opt.feature: 
                a_1, b_1 = module_list[1](feat_t, feat_s) # SAM
                sum_spkd_loss_1 += a_1.item()
                sum_spkd_pixel_loss_1 += b_1.item()
                loss_kd = a_1 * opt.gamma_1 + b_1 * opt.theta_1

                a_2, b_2 = module_list[2](feat_t, feat_s) # SPAM
                sum_spkd_loss_2 += a_2.item()
                sum_spkd_pixel_loss_2 += b_2.item()

                loss_kd += a_2 * opt.gamma_2 + b_2 * opt.theta_2
            elif opt.feature:
                a_1, b_1 = module_list[1](feat_t, feat_s) # SAM
                sum_spkd_loss_1 += a_1.item()
                sum_spkd_pixel_loss_1 += b_1.item()
                loss_kd = a_1 * opt.gamma_1 + b_1 * opt.theta_1
            elif opt.spatial: 
                a_2, b_2 = module_list[1](feat_t, feat_s) # SPAM
                sum_spkd_loss_2 += a_2.item()
                sum_spkd_pixel_loss_2 += b_2.item() 
                loss_kd = a_2 * opt.gamma_2 + b_2 * opt.theta_2
            else: 
                raise NotImplementedError('feature or spatial')
        elif opt.distill == 'tofd':
            loss_kd = module_list[1](feat_t, feat_s, logit_s, logit_t, target, epoch) # tofd
        elif opt.distill == 'afd':
            loss_kd = criterion_kd(feat_s, feat_t)
        else:
            raise NotImplementedError(opt.distill)
        
        if opt.distill != 'kd': 
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        else: 
            loss = opt.gamma * loss_cls + opt.alpha * loss_div

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_kd.update(loss_kd.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss@kd {losskd.val:.4f} ({losskd.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, losskd=losses_kd ,top1=top1, top5=top5))
            sys.stdout.flush()
        
        # if idx % 50 == 0:
        #         print('[epoch:%d, iter:%d] Loss: %.03f, %.03f, %.03f, %.03f, %.03f'
        #               % (epoch + 1, (idx + 1 + epoch * length),
        #                  sum_spkd_loss_1 / (idx + 1), sum_spkd_loss_2 / (idx + 1), sum_spkd_pixel_loss_1 / (idx + 1), sum_spkd_pixel_loss_2 / (idx + 1), loss_cls))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # neuron_records = { 'act_records': [], 'labels': [],}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target,_) in enumerate(val_loader):

            input = input.float()
            device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                input = input.to(device)
                target = target.to(device)

            # output, act_record = model(input)
            # neuron_records['act_records'].extend(act_record.cpu().numpy())
            # neuron_records['labels'].extend(target.cpu().numpy())

            # compute output
            # output = model(input, target)
            _, output = model(input, label=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                      

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    # pickle.dump(neuron_records, open('/home/lthpc/crd/neural_record_init/vgg16_cifar_neuron_records.pkl', 'wb'))
    return top1.avg, top5.avg, losses.avg
