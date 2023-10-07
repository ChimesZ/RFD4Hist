from __future__ import print_function

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SampleAttentionModule(nn.Module):
    def __init__(self, link_1, num_stage=3):
        super(SampleAttentionModule, self).__init__()
        self.alpha_1 = nn.Parameter(torch.zeros(1)).cuda()
        self.link_1 = link_1
        self.num_stage = num_stage
    def forward(self, t, s):
        a, b = 0.0, 0.0
        for j in range(1,self.num_stage+1):
            t_1 = torch.nn.functional.adaptive_avg_pool2d(t[j], 1)
            reshape_t_feature = t_1.view(t_1.size(0), -1)
            relation_t_feature = torch.mm(reshape_t_feature, torch.t(reshape_t_feature))
            relation_t_feature = F.normalize(relation_t_feature)
            weight_t_feature = torch.mm(relation_t_feature, reshape_t_feature)
            reshape_t_feature = reshape_t_feature + self.alpha_1 * weight_t_feature
            
            s_1 = torch.nn.functional.adaptive_avg_pool2d(s[j], 1)
            reshape_s_feature = s_1.view(s_1.size(0), -1)
            relation_s_feature = torch.mm(reshape_s_feature, torch.t(reshape_s_feature))
            relation_s_feature = F.normalize(relation_s_feature)
            weight_s_feature = torch.mm(relation_s_feature, reshape_s_feature)
            reshape_s_feature = reshape_s_feature + self.alpha_1 * weight_s_feature
            reshape_s_feature = self.link_1[j-1](reshape_s_feature)

            a += torch.dist(relation_t_feature, relation_s_feature)
            b += torch.dist(reshape_t_feature, reshape_s_feature, p=2)
        return a, b

class SpatialAttentionModule(nn.Module):
    def __init__(self, link_2, num_stage=3):
        super(SpatialAttentionModule, self).__init__()
        self.alpha_2 = nn.Parameter(torch.zeros(1)).cuda()
        self.link_2 = link_2
        self.num_stage = num_stage
    def forward(self, t, s):
        a, b = 0.0, 0.0

        for j in range(1,self.num_stage+1):
            q = int(4)
            r = int(1)
            t_2 = torch.nn.functional.adaptive_avg_pool2d(t[j], 4)
            re_t_feature = t_2.view(t_2.size(0), t_2.size(1), -1)
            s_2 = torch.nn.functional.adaptive_avg_pool2d(s[j], 4)
            re_s_feature = s_2.view(s_2.size(0), s_2.size(1), -1)
            for x in range(0,3):
                t_2 = torch.nn.functional.adaptive_avg_pool2d(t[j], q)
                reshape_t_feature = t_2.view(t_2.size(0), t_2.size(1), -1)
                relation_t_feature = torch.bmm(reshape_t_feature.permute(0,2,1), reshape_t_feature)
                relation_t_feature = F.normalize(relation_t_feature)
                weight_t_feature = torch.bmm(reshape_t_feature, relation_t_feature)
                Upsample = torch.nn.Upsample(scale_factor=r)
                weight_t_feature = Upsample(weight_t_feature)
                # print(re_t_feature.size())
                # print(weight_t_feature.size())
                re_t_feature = re_t_feature + self.alpha_2 * weight_t_feature

                    
                s_2 = torch.nn.functional.adaptive_avg_pool2d(s[j], q)
                reshape_s_feature = s_2.view(s_2.size(0), s_2.size(1), -1)
                relation_s_feature = torch.bmm(reshape_s_feature.permute(0,2,1), reshape_s_feature)
                relation_s_feature = F.normalize(relation_s_feature)
                weight_s_feature = torch.bmm(reshape_s_feature, relation_s_feature)
                Upsample = torch.nn.Upsample(scale_factor=r)
                weight_s_feature = Upsample(weight_s_feature) 
                # print(re_s_feature.size())
                # print(weight_t_feature.size())
                re_s_feature = re_s_feature + self.alpha_2 * weight_s_feature

                q = int(q / 2)
                r = int(r * 4) 
                a += torch.dist(relation_t_feature, relation_s_feature) 

            re_s_feature = re_s_feature.contiguous().view(s_2.size(0),-1)
            re_t_feature = re_t_feature.contiguous().view(t_2.size(0), -1)
            re_s_feature = self.link_2[j-1](re_s_feature)
                
            b += torch.dist(re_t_feature, re_s_feature, p=2)
        return a, b

def CrossEntropy(outputs, targets, T=3):
    log_softmax_outputs = F.log_softmax(outputs/T, dim=1)
    softmax_targets = F.softmax(targets/T, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

class Tofd(nn.Module):
    def __init__(self, link, num_stage=3, alpha_tofd=0.05, beta_tofd=0.4):
        super(Tofd, self).__init__()
        self.link = link
        self.num_stage = num_stage
        self.alpha_tofd = alpha_tofd
        self.beta_tofd = beta_tofd
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, t, s, outputs, teacher_logits, labels, epoch):
        loss = torch.FloatTensor([0.]).cuda()
        #   Distillation Loss + Task Loss
        for index in range(0, self.num_stage):
            s[index] = self.link[index](s[index])
            #   task-oriented feature distillation loss
            loss += torch.dist(s[index], t[index], p=2) * self.alpha_tofd
            #   task loss (cross entropy loss for the classification task)
            loss += self.criterion(outputs[index], labels)
            #   logit distillation loss, CrossEntropy implemented in utils.py.
            loss += CrossEntropy(outputs[index], teacher_logits[index], 1 + (3.0/250) * float(1+epoch))

        # Orthogonal Loss
        for index in range(self.num_stage):
            weight = list(self.link[index].parameters())[0]
            weight_trans = weight.permute(1, 0)
            ones = torch.eye(weight.size(0)).cuda()
            ones2 = torch.eye(weight.size(1)).cuda()
            loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * self.beta_tofd
            loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * self.beta_tofd
        loss /= 10

        return loss
        

class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""
    def __init__(self, t_shape, k=0.5, use_bn=False):
        super(Paraphraser, self).__init__()
        in_channel = t_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        factor = self.encoder(f_s)
        if is_factor:
            return factor
        rec = self.decoder(factor)
        return factor, rec


class Translator(nn.Module):
    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        super(Translator, self).__init__()
        in_channel = s_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        return self.encoder(f_s)


class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"""
    def __init__(self, s_shapes, t_shapes):
        super(Connector, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        connectors = []
        for s, t in zip(s_shapes, t_shapes):
            if s[1] == t[1] and s[2] == t[2]:
                connectors.append(nn.Sequential())
            else:
                connectors.append(ConvReg(s, t, use_relu=False))
        return connectors

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""
    def __init__(self, s_shapes, t_shapes):
        super(ConnectorV2, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    def _make_conenctors(self, s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        t_channels = [t[1] for t in t_shapes]
        s_channels = [s[1] for s in s_shapes]
        connectors = nn.ModuleList([self._build_feature_connector(t, s)
                                    for t, s in zip(t_channels, s_channels)])
        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class Regress(nn.Module):
    """Simple Linear Regression for hints"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
