'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, List, Dict, Any, cast
import numpy as np


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
                nn.Linear(in_features=channel, out_features=mid_channel),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.classifier = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.cbam = CBAM(channel=512)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    # def forward(self, x, is_feat=False, preact=False):
    def forward(self, x, y, is_feat=False, preact=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
        
        
        
        #####OPERATION#####
        # x = self.cbam(x)
        # x_record = x
        # act_record = self.avgpool(x_record).reshape(x.size(0),512)
        # print(x.size())

        # ****OPERATION****
        batch = x.size(0)
        # print(batch)
        # output = x  # output (32,512,7,7)
        # output = self.avgpool(output)
        # output = output.reshape(batch,512,-1)
        # output = output.transpose(1,2) # ouput (32,49,512)

        # print(output.size())

        # _, a = torch.sort(output, descending=True, dim=-1) # a (32,1,512)
        # a = a.reshape(batch,-1)
        # a_split = [67,445]
        # a ,_ = torch.split(a,a_split,dim=1)
        # a = a.cpu().numpy().tolist()
        # b = self.b.cpu().numpy().tolist()
        # print('b', len(b))
        # ranking_final = []
        # for i in range(batch):
            # max_length = 0
            # ranking = []
            # for j in range(100):
                # print('b', len(b[j]))
                # if len(b[j]) == 67:
                # intersection = [v for v in a[i] if v in b[j]]
                # intersection = list(set(intersection))
                # length = len(intersection)
                # if length >= max_length:
                    # ranking.extend([v for v in b[j]])
                    # if length > max_length:
                        # max_length = length
                        # ranking = b[j]
            # print('overoverover')
            # print(max_length)
            # ranking_final.append(ranking)
                

        # for i in range(batch):
            # for j in range(len(ranking_final[i])):
                # mask_index = ranking_final[i][j]
                # self.mask[i,int(mask_index),] = 1 # mask (32,512,7,7)
        # mask_split = [batch,(64-batch)]
        # mask_final ,_ = torch.split(self.mask,mask_split,dim=0)

        # print('x', x.size())
        # print('mask', mask.size())
        # output_final = x * mask_final  # output_final (32,512,7,7)
        # print(output_final)
        # x = output_final






        # r = torch.matmul(a, torch.t(self.b)) # r (32,1,1000)
        # _, index = torch.max(r,-1) # index (32,1)
        y = y.reshape(batch,1)
        index = y








        # r = torch.matmul(output, torch.t(self.records)) # r (32,1,1000)
        # _, index = torch.max(r,-1) # index (32,1)
        # _, index = torch.topk(r,10) # index (32,1,5)
        # index = index.transpose(1,2) # index (32,5,1)
        # for i in range(batch):
            # for j in range(10):
                # sub_index = index[i,j,].reshape(-1)
                # sub_ranking = self.b[sub_index,]
                # split_list = [67,445]
                # ranking_top ,_  = torch.split(sub_ranking,split_list,dim=1)
                # if j == 0:
                    # ranking = ranking_top
                # else:
                    # ranking = torch.cat([ranking,ranking_top],dim=1)
            # if i == 0:
                # ranking_final = ranking
            # else:
                # ranking_final = torch.cat([ranking_final,ranking],dim=0)
        # ranking_final ,_ = torch.sort(ranking_final,descending=False,dim=-1)

        # for i in range(batch):
            # for j in range(670):
                # mask_index = ranking_final[i,j]
                # print(mask_index)
                # self.mask[i,mask_index.int(),] = 1 # mask (32,512,7,7)
        # mask_split = [batch,(32-batch)]
        # mask_final ,_ = torch.split(self.mask,mask_split,dim=0)

        # print('x', x.size())
        # print('mask', mask.size())
        # output_final = x * mask_final  # output_final (32,512,7,7)
        # print(output_final)
        # x = output_final
        
        
        
        engram_number = self.b.size(1)
        for i in range(batch):
            sub_index = index[i,]
            sub_index = sub_index.reshape(-1)
            sub_ranking = self.b[sub_index,]
            if i == 0 :
                ranking = sub_ranking
            else:
                ranking = torch.cat([ranking,sub_ranking],dim=0)
        # print(ranking.size()) # ranking (32,512)
        split_list = [engram_number,0]
        ranking_seven ,_  = torch.split(ranking,split_list,dim=1)  # ranking_seven (32,7)
        # print(ranking_seven.size())
        ranking_seven ,_  = torch.sort(ranking_seven,descending=False, dim=-1)
        # print(ranking_seven)

        for i in range(batch):
            for j in range(engram_number):
                mask_index = ranking_seven[i,j]
                # print(mask_index)
                self.mask[i,mask_index.int(),] = 1 # mask (32,512,7,7)
        mask_split = [batch,(64-batch)]
        mask_final ,_ = torch.split(self.mask,mask_split,dim=0)

        # print('x', x.size())
        # print('mask', mask.size())
        output_final = x * mask_final  # output_final (32,512,7,7)
        # print(output_final)
        x = output_final
        
        # count = 0
        # for j in range(512):
            # if torch.mean(torch.mean(self.mask[0,j,],dim=0,keepdim=True),dim=1,keepdim=True) == self.cal:
                # count = count + 1
        # print



        self.mask[:,:,:,:] = 0
        self.mask = self.mask.float()
        
        #####END#####


        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
            # return x, act_record

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = vgg19_bn(num_classes=100)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
