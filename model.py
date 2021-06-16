#!/usr/bin/env python3
# encoding: utf-8

import torch,math
import torch.nn as nn

class dim2SEResNetBlock(nn.Module):
    def __init__(self, channels, channels_out, kernel_size, dilate_size):
        super().__init__()
        pad_val = dilate_size*(kernel_size-1)//2
        self.channels_out = channels_out
        self.layers = nn.Sequential(
            nn.Sequential(nn.InstanceNorm2d(channels), nn.Conv2d(channels, channels_out, kernel_size, dilation=dilate_size, padding=(pad_val,pad_val), bias=False)),
            nn.ELU(inplace=True),
            nn.Sequential(nn.InstanceNorm2d(channels), nn.Conv2d(channels, channels_out, kernel_size, dilation=dilate_size, padding=(pad_val,pad_val), bias=False)),
        )
        self.fc1 = nn.Linear(in_features=channels_out*2, out_features=round(channels_out / 16))
        self.fc2 = nn.Linear(in_features=round(channels_out / 16), out_features=channels_out)
        self.sigmoid = nn.Sigmoid()
        self.rpool = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.cpool = torch.nn.AdaptiveAvgPool2d((1, None))
        self.elu = nn.ELU(inplace=True)

    def forward(self, fea):
        residual = fea
        fea = self.layers(fea)
        out = fea 
        original_out = out
        row_fea = self.rpool(fea)
        row_fea = self.cpool(row_fea)
        col_fea = self.cpool(fea)
        col_fea = self.rpool(col_fea)
        out = torch.cat((row_fea, col_fea), dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out
        fea += residual
        fea = self.elu(fea)
        return fea    
        
class dim2SEResNetLayer(nn.Module):
    def __init__(self, channels, channels_out, kernel_size, dilate_size, layer_nums):
        super().__init__()
        self.layers = nn.Sequential(
            *[dim2SEResNetBlock(channels_out, channels_out, kernel_size, dilate_size) for _ in range(layer_nums)]
        )

    def forward(self, fea):
        fea = self.layers(fea)
        return fea
    
class dim2SEResNet(nn.Module):
    def __init__(self, channels, deepths, 
            kernel_size=5, channel_dim=64, dilate_size=2):
        super().__init__()
        self.convert_part = nn.Sequential(
            nn.Conv2d(channels, channel_dim, kernel_size=kernel_size, padding=2, bias=False),
            nn.InstanceNorm2d(channel_dim),
            nn.ELU(inplace=True),
        )
        self.layers = nn.ModuleList([
            *[dim2SEResNetLayer(channel_dim, channel_dim, kernel_size, dilate_size, layer_nums=n) for n in deepths]       
        ])
        
    def forward(self, fea):
        fea = self.convert_part(fea)
        for layer in self.layers:
            fea = layer(fea)
        return fea


class dim1SEResNetLayer(nn.Module):
    def __init__(self, channels, channels_out, groups, 
            kernel_size=5, dilate_size=2):
        super().__init__()
        pad_val = dilate_size*((kernel_size-1)//2)
        self.layers = nn.Sequential(
            nn.Sequential(nn.InstanceNorm1d(channels), nn.Conv1d(channels, channels_out, kernel_size, groups=groups, dilation=dilate_size, padding=pad_val, bias=False)),
            nn.ELU(inplace=True),
            nn.Sequential(nn.InstanceNorm1d(channels), nn.Conv1d(channels, channels_out, kernel_size, groups=groups, dilation=dilate_size, padding=pad_val, bias=False)),
        )
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(in_features=channels_out, out_features=round(channels_out / 16))
        self.fc2 = nn.Linear(in_features=round(channels_out / 16), out_features=channels_out)
        self.sigmoid = nn.Sigmoid()
        self.rpool = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.cpool = torch.nn.AdaptiveAvgPool2d((1, None))
        self.elu = nn.ELU(inplace=True)
        self.pool_1d = nn.AdaptiveAvgPool1d((1))


    def forward(self, fea):
        residual = fea
        fea = self.layers(fea)
        original_out = fea
        out = self.pool_1d(fea)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out.squeeze(2) 
        out = out * original_out
        fea += residual
        fea = self.elu(fea)
        return fea
        

class QARESEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim2seresnet = dim2SEResNet(44, [2,3,3,2])
        self.dim2seresnet_out = self.dim2seresnet.layers[-1].layers[-1].channels_out
        self.rpool = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.cpool = torch.nn.AdaptiveAvgPool2d((1, None))
        
        dim1seresnet_group = self.dim2seresnet_out*2 + 52
        self.dim1seresnet = nn.ModuleList([
            *[dim1SEResNetLayer(dim1seresnet_group, dim1seresnet_group, groups=dim1seresnet_group)
                for _ in range(8)]
        ])

        self.pool_1d = nn.AdaptiveAvgPool1d((1))
        self.linear = nn.Linear(dim1seresnet_group, 1)

        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
           if isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
           elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
             nn.init.constant_(m.weight, 1)
             nn.init.constant_(m.bias, 0)
    
    def forward(self, fea1, fea2):
        fea = self.dim2seresnet(fea2)
        row_fea = self.rpool(fea)
        row_fea = row_fea.permute(0,1,3,2).reshape(row_fea.size()[0], row_fea.size()[1]*row_fea.size()[3], row_fea.size()[2])
        col_fea = self.cpool(fea)
        col_fea = col_fea.reshape(col_fea.size()[0], col_fea.size()[1]*col_fea.size()[2], col_fea.size()[3])
        fea = torch.cat((row_fea, col_fea, fea1), dim=1)
        for layer in self.dim1seresnet: fea = layer(fea)
        
        gdt_val = self.linear(self.pool_1d(fea).view(fea.size()[0], -1))
        gdt_val = self.sigmoid(gdt_val)
        gdt_val = gdt_val.half()
        return gdt_val

