# This file contains modules common to various models

import math
from pathlib import Path
from torch.autograd import Variable, Function
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list, plot_one_box
from torch.nn import init
import torch.nn.functional as F
import math
from .SE_weight_module import SEWeightModule
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


#SINET：隐藏目标检测
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()

        self.relu = nn.ReLU(True)
        #out_channel=32
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        #print('x0',x0.shape)
        x1 = self.branch1(x)
        #print('x1', x1.shape)
        x2 = self.branch2(x)
        #print('x2', x2.shape)
        x3 = self.branch3(x)
        #print('x3', x3.shape)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class PDC_SM(nn.Module):
    # Partial Decoder Component (Search Module)
    def __init__(self, channel):
        super(PDC_SM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2)), x4), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class PDC_IM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDC_IM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class softmaxattention(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,c1, k=1, s=1, p=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(softmaxattention, self).__init__()
        # hidden channels
        #print('c1',c1/8)
        self.cv1 = Conv(512,64, 3, 1,1)#640*640*9
        self.cv2 = Conv(512,64, 3, 1,1)#640*640*6
        #self.cv3 = Conv(6, 3,3, 1,1)  #640*640*3# act=FReLU(c2)
        #Conv(3, 3, 1, 1, 0)
        #self.m 6= nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        #print('x1',x.shape)
        y1=self.cv1(x)  ##[B,C,H,W]
        #print('y1', y1.shape)
        y2=self.cv2(x)  ##
        #print('y2', y2.shape)##[B,C,H,W]
        x_= x.view([x.size()[0], x.size()[1], -1]) #[B,C,H*w]
        #print('x_', x_.shape)  ##[B,C,H,W]

        y2_=y2.view([y2.size()[0],y2.size()[1],-1]) #[B,C,H*w] 0 1 2
        #print('y2_', y2_.shape)  ##[B,C,H*w] 0 1 2
        y1_ = y1.view([y1.size()[0], y1.size()[1], -1])  # [B,C,H*w] 0 1 2
        #print('y1_ ', y1_ .shape)  ##[B,C,H*w] 0 1 2

        y2_T = y2_.permute([0,2,1])  # [B,H*w,C]  0 2 1
        #print('y2_T ', y2_T.shape)# [B,H*w,C]
        #c=y1_*y2_T  #[B,C,H*w]  * [B,H*w,C]
        c=torch.matmul(y2_T,y1_ ) #[B,H*w,H*w]
        #print('c ', c.shape)  # [B,H*w,H*w]
        C_weight=F.softmax(c,dim=-1)   ##[B,H*w,H*w]
        #print('C_weight', C_weight.shape)  # [B,H*w,H*w]
        #y=C*x
        #y2_ = y2.view([y2.size()[0], y2.size()[1], -1])  #
        y_last = torch.matmul(x_,C_weight) #[B,C,H*w] *[B,H*w,H*w]
        #print('y_last', y_last.shape)   #[B,C,H*w] *[B,H*w,H*w]
        y_last_ = y_last.view([y_last.size()[0], y_last.size()[1],  x.size()[2], x.size()[3]])  # [B,C,H*w] 0 1 2
        #print('y1', y1.shape)
        #y2=self.cv2(y1)
        #print('y_last_', y_last_.shape)
        return y_last_


class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer
    as proposed in the Jaderberg paper.
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator
    3. A roi pooled module.
    The current implementation uses a very small convolutional net with
    2 convolutional layers and 2 fully connected layers. Backends
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map.
    """
    def __init__(self,c1, k=1, s=1, p=None):
        super(SpatialTransformer, self).__init__()
        kernel_size=3
        use_dropout = False
        #self._h, self._w = 20,20#spatial_dims
        self._in_ch = c1
        self._ksize = kernel_size
        self.dropout = use_dropout
        # localization net
        self.conv1 = nn.Conv2d(c1, 32, kernel_size=self._ksize, stride=1, padding=1,
                               bias=False)  # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        #self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(32 * 1* 1, 128) #input/8 4 4
        self.fc2 = nn.Linear(128, 6)
    def forward(self, x):
        """
        Forward pass of the STN module.
        x -> input feature map
        """
        #print('x.shape',x.shape)
        batch_images = x
        b_s, c, h, w = x.shape
        #print('x.shape',x.shape)
        x = F.relu(self.conv1(x.detach()))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        #print('x2.shape', x.shape)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        #print('x3.shape', x.shape)
        x = x.view(-1, 32 * x.shape[2] * x.shape[3]) #input/8
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x)  # params [Nx6]

        x = x.view(-1, 2, 3)  # change it to the 2x3 matrix
        #print(x.size())
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, h,w)),align_corners=False)
        assert (affine_grid_points.size(0) == batch_images.size(
            0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points,align_corners=False)
        #print("rois found to be of size:{}".format(rois.size()))
        return rois #, affine_grid_points


class NonLocal(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,c1, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(NonLocal, self).__init__()
        # hidden channels
        #print('c1',c1/8)
        self.cv1 = Conv(c1,c1//8, 1, 1,0)#640*640*9
        self.cv2 = Conv(c1,c1//8, 1, 1,0)#640*640*6
        self.cv3 = Conv(c1,c1//8, 1, 1,0)  #640*640*3# act=FReLU(c2)
        self.cv4 = Conv(c1//8, c1, 1, 1, 0)  # 640*640*3# act=FReLU(c2)

        #Conv(3, 3, 1, 1, 0)
        #self.m 6= nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        #print('x1',x.shape)
        y1=self.cv1(x)  ##[B,C,H,W]
        #print('y1', y1.shape)
        y2=self.cv2(x)  ##
        y3 = self.cv3(x)  ##
        #print('y2', y2.shape)##[B,C,H,W]
        y1_= y1.view([y1.size()[0], y1.size()[1], -1]) #[B,C,H*w]
        #print('x_', x_.shape)  ##[B,C,H,W]

        y2_=y2.view([y2.size()[0],y2.size()[1],-1]) #[B,C,H*w] 0 1 2
        #print('y2_', y2_.shape)  ##[B,C,H*w] 0 1 2
        y3_ = y3.view([y3.size()[0], y3.size()[1], -1])  # [B,C,H*w] 0 1 2
        #print('y1_ ', y1_ .shape)  ##[B,C,H*w] 0 1 2

        y2_T = y2_.permute([0,2,1])  # [B,H*w,C]  0 2 1
        #print('y2_T ', y2_T.shape)# [B,H*w,C]
        #c=y1_*y2_T  #[B,C,H*w]  * [B,H*w,C]
        c=torch.matmul(y2_T,y1_ ) #[B,H*w,H*w]
        #print('c ', c.shape)  # [B,H*w,H*w]
        C_weight=F.softmax(c,dim=-1)   ##[B,H*w,H*w]
        #print('C_weight', C_weight.shape)  # [B,H*w,H*w]
        #y=C*x
        #y2_ = y2.view([y2.size()[0], y2.size()[1], -1])  #
        y_last = torch.matmul(y3_,C_weight) #[B,0.5*C,H*w] *[B,H*w,H*w]
           #[B,0.5*C,H*w] *[B,H*w,H*w]
        y_last_ = y_last.view([y_last.size()[0], y_last.size()[1],  x.size()[2], x.size()[3]])  # [B,0.5*C,H,w] 0 1 2
        #print('y_last', y_last_.shape)
        mask=self.cv4(y_last_)
        out=mask+x
        #print('y1', y1.shape)
        #y2=self.cv2(y1)
        #print('y_last_', y_last_.shape)
        return out


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        #print('c1',c1,c2)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def fuseforward(self, x):
        return self.act(self.conv(x))


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class ASFFV5(nn.Module):
    def __init__(self, level, multiplier=0.5, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 only.
        Since YoloV5 outputs 3 layer of feature maps with different channels
        which is different than YoloV3
        normally, multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you gonna need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]
        # print("dim:",self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)
            # print(self.dim)
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16

        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):  # s,m,l
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        print('x_level_0: ', x_level_0.shape)
        print('x_level_1: ', x_level_1.shape)
        print('x_level_2: ', x_level_2.shape)
        # x_level_0 = x[2]
        # x_level_1 = x[1]
        # x_level_2 = x[0]
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)

            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            # print('X¡ª¡ªlevel_0: ', level_2_downsampled_inter.shape)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #  level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class C3_psa(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_psa, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck_psa(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Bottleneck_psa(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck_psa, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = PSAModule(c_, c2, 3, 1)
        #self.cv2 = Conv(c_, c2, [3, 5], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PSAModule(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=3, stride=1, conv_groups=1):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=3, padding=3//2,
                            stride=stride, groups=1)
        self.conv_2 = conv(inplans, planes//4, kernel_size=5, padding=5//2,
                            stride=stride, groups=4)
        self.conv_3 = conv(inplans, planes//4, kernel_size=7, padding=7//2,
                            stride=stride, groups=8)
        self.conv_4 = conv(inplans, planes//4, kernel_size=9, padding=9//2,
                            stride=stride, groups=16)
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        return out


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3,stride =1, padding=1, bias=None):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride =stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.offsets = nn.Conv2d(inc, 18, kernel_size=kernel_size,  padding=1,stride=stride)
    # ×¢Òâ£¬offsetµÄTensor³ß´çÊÇ[b, 18, h, w]£¬offset´«ÈëµÄÆäÊµ¾ÍÊÇÃ¿¸öÏñËØµãµÄ×ø±êÆ«ÒÆ£¬Ò²¾ÍÊÇÒ»¸ö×ø±êÁ¿£¬×îÖÕÃ¿¸öµãµÄÏñËØ»¹ÐèÒªÕâ¸ö×ø±êÆ«ÒÆºÍÔ­Í¼½øÐÐ¶ÔÓ¦Çó³ö¡£
    def forward(self, x):
        #print('x', x.shape)
        offset = self.offsets(x)
        #print('offset', offset.shape)
        dtype = offset.data.type()
        ks = self.kernel_size
        # N=9=3x3
        N = offset.size(1) // 2

        # ÕâÀïÆäÊµÃ»±ØÒª£¬ÎÒÃÇ·´ÕýÕâ¸öË³ÐòÊÇÎÒÃÇ×Ô¼º¶¨ÒåµÄ£¬ÄÇÎÒÃÇÖ±½Ó°´ÕÕ[x1, x2, .... y1, y2, ...]¶¨Òå²»¾ÍºÃÁË¡£
        # ½«offsetµÄË³Ðò´Ó[x1, y1, x2, y2, ...] ¸Ä³É[x1, x2, .... y1, y2, ...]
        offsets_index = Variable(torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]),
                                 requires_grad=False).type_as(x).long()
        # torch.unsqueeze()ÊÇÎªÁËÔö¼ÓÎ¬¶È,Ê¹offsets_indexÎ¬¶ÈµÈÓÚoffset
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # ¸ù¾ÝÎ¬¶Èdim°´ÕÕË÷ÒýÁÐ±íindex½«offsetÖØÐÂÅÅÐò£¬µÃµ½[x1, x2, .... y1, y2, ...]ÕâÑùË³ÐòµÄoffset
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        # ¶ÔÊäÈëx½øÐÐpadding
        if self.padding:
            x = self.zero_padding(x)

        # ½«offset·Åµ½Íø¸ñÉÏ£¬Ò²¾ÍÊÇ±ê¶¨³öÃ¿Ò»¸ö×ø±êÎ»ÖÃ
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # Î¬¶È±ä»»
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floorÊÇÏòÏÂÈ¡Õû
        q_lt = Variable(p.data, requires_grad=False).floor()
        # +1Ïàµ±ÓÚÏòÉÏÈ¡Õû£¬ÕâÀïÎªÊ²Ã´²»ÓÃÏòÉÏÈ¡Õûº¯ÊýÄØ£¿ÊÇÒòÎªÈç¹ûÕýºÃÊÇÕûÊýµÄ»°£¬ÏòÉÏÈ¡Õû¸úÏòÏÂÈ¡Õû¾ÍÖØºÏÁË£¬ÕâÊÇÎÒÃÇ²»Ïë¿´µ½µÄ¡£
        q_rb = q_lt + 1

        # ½«ltÏÞÖÆÔÚÍ¼Ïñ·¶Î§ÄÚ£¬ÆäÖÐ[..., :N]´ú±íx×ø±ê£¬[..., N:]´ú±íy×ø±ê
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # ½«rbÏÞÖÆÔÚÍ¼Ïñ·¶Î§ÄÚ
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # »ñµÃlb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        # »ñµÃrt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # ÏÞÖÆÔÚÒ»¶¨µÄÇøÓòÄÚ,ÆäÊµÕâ²¿·Ö¿ÉÒÔÐ´µÄºÜ¼òµ¥¡£ÓÐµã»¨ÀïºúÉÚµÄ¸Ð¾õ¡£¡£ÔÚnumpyÖÐÕâÑùÐ´£º
        # p = np.where(p >= 1, p, 0)
        # p = np.where(p <x.shape[2]-1, p, x.shape[2]-1)

        # ²åÖµµÄÊ±ºòÐèÒª¿¼ÂÇÒ»ÏÂpadding¶ÔÔ­Ê¼Ë÷ÒýµÄÓ°Ïì
        # (b, h, w, N)
        # torch.lt() ÖðÔªËØ±È½ÏinputºÍother£¬¼´ÊÇ·ñinput < other
        # torch.rt() ÖðÔªËØ±È½ÏinputºÍother£¬¼´ÊÇ·ñinput > other
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        # ½ûÖ¹·´Ïò´«²¥
        mask = mask.detach()
        # p - (p - torch.floor(p))²»¾ÍÊÇtorch.floor(p)ÄØ¡£¡£¡£
        floor_p = p - (p - torch.floor(p))
        # ×ÜµÄÀ´Ëµ¾ÍÊÇ°Ñ³¬³öÍ¼ÏñµÄÆ«ÒÆÁ¿ÏòÏÂÈ¡Õû
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # ²åÖµµÄ4¸öÏµÊý
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # ²åÖµµÄ×îÖÕ²Ù×÷ÔÚÕâÀï
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # Æ«ÖÃµãº¬ÓÐ¾Å¸ö·½ÏòµÄÆ«ÖÃ£¬_reshape_x_offset() °ÑÃ¿¸öµã9¸ö·½ÏòµÄÆ«ÖÃ×ª»¯³É 3¡Á3 µÄÐÎÊ½£¬
        # ÓÚÊÇ¾Í¿ÉÒÔÓÃ 3¡Á3 stride=3 µÄ¾í»ýºË½øÐÐ Deformable Convolution£¬
        # ËüµÈ¼ÛÓÚÊ¹ÓÃ 1¡Á1 µÄÕý³£¾í»ýºË£¨°üº¬ÁËÕâ¸öµã9¸ö·½ÏòµÄ context£©¶ÔÔ­ÌØÕ÷Ö±½Ó½øÐÐ¾í»ý¡£
        x_offset = self._reshape_x_offset(x_offset, ks)

        out = self.conv_kernel(x_offset)
        #print('out', out.shape)
        return out

    # ÇóÃ¿¸öµãµÄÆ«ÖÃ·½Ïò
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    # ÇóÃ¿¸öµãµÄ×ø±ê
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h + 1), range(1, w + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    # Çó×îºóµÄÆ«ÖÃºóµÄµã=Ã¿¸öµãµÄ×ø±ê+Æ«ÖÃ·½Ïò+Æ«ÖÃ
    def _get_p(self, offset, dtype):
        # N = 9, h, w
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    # Çó³öpµãÖÜÎ§ËÄ¸öµãµÄÏñËØ
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)½«Í¼Æ¬Ñ¹Ëõµ½1Î¬£¬·½±ãºóÃæµÄ°´ÕÕindexË÷ÒýÌáÈ¡
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)Õâ¸öÄ¿µÄ¾ÍÊÇ½«indexË÷Òý¾ùÔÈÀ©Ôöµ½Í¼Æ¬Ò»ÑùµÄh*w´óÐ¡
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # Ë«ÏßÐÔ²åÖµ·¨¾ÍÊÇ4¸öµãÔÙ³ËÒÔ¶ÔÓ¦Óë p µãµÄ¾àÀë¡£»ñµÃÆ«ÖÃµã p µÄÖµ£¬Õâ¸ö p µãÊÇ 9 ¸ö·½ÏòµÄÆ«ÖÃËùÒÔ×îºóµÄ x_offset ÊÇ b¡Ác¡Áh¡Áw¡Á9¡£
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    # _reshape_x_offset() °ÑÃ¿¸öµã9¸ö·½ÏòµÄÆ«ÖÃ×ª»¯³É 3¡Á3 µÄÐÎÊ½
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        #print('x_offset',x_offset.shape)
        return x_offset


class DeformConv2dv2(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1,padding=1,  bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2dv2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        #print('x',x.shape)
        offset = self.p_conv(x)
        #print('offset', offset.shape)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        #print('out', out.shape)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)


        return x_offset


class process(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,c1, k=1, s=1, p=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(process, self).__init__()
        # hidden channels
        self.cv1 = Conv(c1, 9, 3, 1,1)#640*640*9
        self.cv2 = Conv(9, 6, 3,1, 1)#640*640*6
        self.cv3 = Conv(6, 3,3, 1,1)  #640*640*3# act=FReLU(c2)
        self.cv4 = Conv(3, 6, 3,1, 1)#  640*640*6
        self.cv5 = Conv(6, 9,3, 1, 1)  ##640*640*3
        self.cv6 = Conv(9, 3, 1,1, 0)
        self.cv7 = DWConv(3, 3, 3, 1)
        self.cv8 = Conv(3, 3, 1, 1, 0)
        self.cv9 = DWConv(3, 3, 3, 1)
        self.pool = nn.MaxPool2d(10, 1)
        #self.m 6= nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        print('x',x.shape)
        y1=self.cv1(x)
        print('y1', y1.shape)
        y2=self.cv2(y1)
        #print('y2', y2.shape)
        y3 = self.cv3(y2)
        #print('y3', y3.shape)
        y4 = self.cv4(y3)
        #print('x4', y4.shape)
        y5 = self.cv5(y4)
        #print('x5', y5.shape)
        y6= self.cv6(y5)
        #print('y6', y6.shape)
        y7 = self.cv7(y6)
        #print('y7', y7.shape)
        y8 = self.cv8(y7)
        #print('y8', y8.shape)
        y9 = self.cv9(y8)
        #print('y9', y9.shape)
        #from models import feature
        #try:
        #feature.feature_vision(y9, '/mnt/sda1/zfy/shuoshi_lunwen/data/1/out')
        #except:
            #print('2222222222222')
            #pass
        return y9


class SelfAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=0.01):
        super(SelfAttention, self).__init__()

        self.fc_q = nn.Linear(d_model, h * d_k)         # (512, 8 * 512)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)         # (8 * 512, 512)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        b_s, c, h, w = x.shape
        nq = c
        nk = c

        x = x.view(b_s, c, h*w)

        q = self.fc_q(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out.view(b_s, c, h, w)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        #self.cv2 = Conv(c_, c2, [3, 5], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3_PSA(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        #print('focus',c1,c2)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain
    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain
    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x):
        x0=x
        y=x
        a = torch.sigmoid(-y)
        x = self.convert(x)
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        #print('x0xxx',x0.shape,self.convs(x).shape)
        x =x0+ self.convs(x)
        return x


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)

#BIFPN
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Concat_BIFPN(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(Concat_BIFPN, self).__init__()
        # self.relu = nn.ReLU()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.swish = MemoryEfficientSwish()

        self.relu=nn.ReLU
    def forward(self, x):
        outs = self._forward(x)
        return outs

    def _forward(self, x):  # intermediate result
        if len(x) == 2:
            #w = self.relu(self.w1)
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:  # final result
            #w = self.relu(self.w2)
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im  # open
                im.filename = f  # for uri
            files.append(Path(im.filename).with_suffix('.jpg').name if isinstance(im, Image.Image) else f'image{i}.jpg')
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, files, self.names)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
        # 
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = Path(save_dir) / self.files[i]
                img.save(f)  # save
                print(f"{'Saving' * (i == 0)} {f},", end='' if i < self.n - 1 else ' done.\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='results/'):
        Path(save_dir).mkdir(exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
