# -*- coding: utf-8 -*-
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
class ReOrgLayer(nn.Module):
    def __init__(self, stride = 2):
        super(ReOrgLayer, self).__init__()
        self.stride= stride
        
    def forward(self,x):
        assert(x.data.dim() == 4)
        B,C,H,W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert(H % hs == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        assert(W % ws == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        x = x.view(B,C, H // hs, hs, W // ws, ws).transpose(-2,-3).contiguous()
        x = x.view(B,C, H // hs * W // ws, hs, ws)
        x = x.view(B,C, H // hs * W // ws, hs*ws).transpose(-1,-2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).transpose(1,2).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws)
        '''
        x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H/hs, W/ws)
        '''
        return x


class Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Block, self).__init__()
        self.double_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.double_conv(x)
class DoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.double_conv(x)
class Block1x1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Block1x1, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.conv1x1(x)
class DilatedBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DilatedBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, middle_channels, 3, padding=2,dilation=2),
            nn.BatchNorm2d(middle_channels),
            nn.Conv2d(middle_channels, out_channels, 3, padding=2,dilation=2),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.double_conv(x)


class NestedUNet(nn.Module):
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = Block(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = Block(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = Block(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = Block(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = Block(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = Block(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = Block(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output

class UNet(nn.Module):
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = Block(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0],n_classes , kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
    
class NestedWNet(nn.Module): # only maxpooling
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = Block(nb_filter[0]+nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = Block(nb_filter[1]+nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = Block(nb_filter[2]+nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = Block(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = Block(nb_filter[0]+nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = Block(nb_filter[1]+nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = Block(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = Block(nb_filter[0]+nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = Block(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)


    def forward(self, input):
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.pool(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.pool(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.pool(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.pool(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.pool(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.pool(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output
    

class NestedWNet_v2(nn.Module): #downsample + maxpooling
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_1d= Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_1 = Block(nb_filter[1]+nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_1d= Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv2_1 = Block(nb_filter[2]+nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_1d= Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv3_1 = Block(nb_filter[3]+nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = Block(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2d= Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_2 = Block(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_2d= Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv2_2 = Block(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = Block(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3d= Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_3 = Block(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = Block(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1d= self.conv0_1d(self.pool(x0_1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), x0_1d], 1))
        x1_1d= self.conv1_1d(self.pool(x1_1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2d= self.conv0_2d(self.pool(x0_2)) 

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),x1_1d], 1))
        x2_1d= self.conv2_1d(self.pool(x2_1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),x0_2d], 1))
        x1_2d= self.conv1_2d(self.pool(x1_2))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3d= self.conv0_3d(self.pool(x0_3))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),x2_1d], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),x1_2d], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),x0_3d], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output

class NestedWNet_v3(nn.Module): #downsample + maxpooling
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.up1_0 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(nb_filter[3], nb_filter[3], kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(nb_filter[4], nb_filter[4], kernel_size=2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[3], kernel_size=2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=2, stride=2)
        
        
        self.conv0_0 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1_1x1 = Block1x1(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_1 = Block(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv0_1d= Block(nb_filter[0], nb_filter[1], nb_filter[1])
        
        self.conv1_1_1x1 = Block1x1(nb_filter[1]+nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_1 = Block(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv1_1d= Block(nb_filter[1], nb_filter[2], nb_filter[2])
        
        self.conv2_1_1x1 = Block1x1(nb_filter[2]+nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_1 = Block(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv2_1d= Block(nb_filter[2], nb_filter[3], nb_filter[3])
        
        self.conv3_1_1x1 = Block1x1(nb_filter[3]+nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv3_1 = Block(nb_filter[3], nb_filter[3], nb_filter[3])

        self.conv0_2_1x1 = Block1x1(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2 = Block(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv0_2d= Block(nb_filter[0], nb_filter[1], nb_filter[1])
        
        self.conv1_2_1x1 = Block1x1(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_2 = Block(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv1_2d= Block(nb_filter[1], nb_filter[2], nb_filter[2])
        
        self.conv2_2_1x1 = Block1x1(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_2 = Block(nb_filter[2], nb_filter[2], nb_filter[2])
        
        self.conv0_3_1x1 = Block1x1(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3 = Block(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv0_3d= Block(nb_filter[0], nb_filter[1], nb_filter[1])
            
        self.conv1_3_1x1 = Block1x1(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_3 = Block(nb_filter[1], nb_filter[1], nb_filter[1])

        self.conv0_4_1x1 = Block1x1(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4 = Block(nb_filter[0], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        x0_1 = self.conv0_1_1x1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1 = self.conv0_1(x0_1)
        x0_1d= self.conv0_1d(self.pool(x0_1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1_1x1(torch.cat([x1_0, self.up(x2_0), x0_1d], 1))
        x1_1 = self.conv1_1(x1_1)
        
        x1_1d= self.conv1_1d(self.pool(x1_1))
        x0_2 = self.conv0_2_1x1(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.conv0_2(x0_2)
        x0_2d= self.conv0_2d(self.pool(x0_2)) 

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1_1x1(torch.cat([x2_0, self.up(x3_0),x1_1d], 1))
        x2_1 = self.conv2_1(x2_1)
        x2_1d= self.conv2_1d(self.pool(x2_1))
       
        x1_2 = self.conv1_2_1x1(torch.cat([x1_0, x1_1, self.up(x2_1),x0_2d], 1))
        x1_2 = self.conv1_2(x1_2)
        x1_2d= self.conv1_2d(self.pool(x1_2))
            
        x0_3 = self.conv0_3_1x1(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.conv0_3(x0_3)
        x0_3d= self.conv0_3d(self.pool(x0_3))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1_1x1(torch.cat([x3_0, self.up(x4_0),x2_1d], 1))
        x3_1 = self.conv3_1(x3_1)
        x2_2 = self.conv2_2_1x1(torch.cat([x2_0, x2_1, self.up(x3_1),x1_2d], 1))    
        x2_2 = self.conv2_2(x2_2)
        x1_3 = self.conv1_3_1x1(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),x0_3d], 1))
        x1_3 = self.conv1_3(x1_3)
        x0_4 = self.conv0_4_1x1(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.conv0_4(x0_4)
        
        output = self.final(x0_4)
        return output



class stochasticUNet(nn.Module):
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.multiplier = 2
        self.layers = 4
        self.downmodule = nn.ModuleList()
        self.upmodule = nn.ModuleList()
        nb_filter = [32, 64, 128, 256, 512]
        self.inifilter = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = DoubleBlock(input_channels, nb_filter[0])

        for layer in range(self.layers):
            layermodule = DoubleBlock(self.inifilter * np.power(self.multiplier,layer) , self.inifilter * np.power(self.multiplier,layer+1))
            self.downmodule.append(layermodule)
        for layer in range(self.layers):
            layermodule = DoubleBlock(self.inifilter * np.power(self.multiplier,4-layer) + self.inifilter * np.power(self.multiplier,3-layer)
                                     , self.inifilter * np.power(self.multiplier,3-layer))
            self.upmodule.append(layermodule)

            layermodulenc = DoubleBlock(self.inifilter * np.power(self.multiplier,4-layer), self.inifilter * np.power(self.multiplier,3-layer))
            self.upmodule.append(layermodulenc)


        self.final = nn.Conv2d(self.inifilter,n_classes , kernel_size=1)


    def forward(self, input):
        #normalized_betas = torch.randn(self._num_layers, 1, 3).cuda()
        c1=1
        c2=1
        c3=1
        c4=0
        x0_0 = self.conv0_0(input)
        x1_0 = self.downmodule[0](self.pool(x0_0))
        x2_0 = self.downmodule[1](self.pool(x1_0))
        x3_0 = self.downmodule[2](self.pool(x2_0))
        x4_0 = self.downmodule[3](self.pool(x3_0))
        
        if c1 == 1 :
            x3_1 = self.upmodule[0](torch.cat([x3_0, self.up(x4_0)], 1))
        if c1 == 0 :
            x3_1 = self.upmodule[1](self.up(x4_0))
        if c2 == 1 :
            x2_2 = self.upmodule[2](torch.cat([x2_0, self.up(x3_1)], 1))
        if c2 == 0 : 
            x2_2 = self.upmodule[3](self.up(x3_1))
        if c3 == 1 :
            x1_3 = self.upmodule[4](torch.cat([x1_0, self.up(x2_2)], 1))
        if c3 == 0 :
            x1_3 = self.upmodule[5](self.up(x2_2))
        if c4 == 1 :
            x0_4 = self.upmodule[6](torch.cat([x0_0, self.up(x1_3)], 1))
        if c4 == 0 :
            x0_4 = self.upmodule[7](self.up(x1_3))
        
        output = self.final(x0_4)
        return output
class NestedUNet_RO(nn.Module):
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = ReOrgLayer()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = Block(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = Block(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = Block(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = Block(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = Block(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = Block(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = Block(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output

if __name__ == '__main__':
    x = torch.rand(size=(1,1,256,256))
    model = stochasticUNet(1,2)
    output = model(x)
    print(output)
    