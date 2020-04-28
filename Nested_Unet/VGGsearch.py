# -*- coding: utf-8 -*-
import numpy as np
from torch import nn ,optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch
from torchvision import models
import torchvision
from torchsummary import summary
from thop import profile
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

class DoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)
    
class CBBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBBlock, self).__init__()
        self.CBBlock_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.CBBlock_conv(x)
    
class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatBlock, self).__init__()
        self.Concat_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.Concat_conv(x)
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



    
class ALLSearch(nn.Module):
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.multiplier = 2
        self.layers = 5
        self.module = nn.ModuleList()
        nb_filter = [32, 64, 128, 256, 512]
        self.inifilter = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final = nn.Conv2d(self.inifilter,n_classes , kernel_size=1)
        #self.routes = [nn.Parameter(torch.randn(2*(self.layers-1-i)-1).cuda(), requires_grad=True) for i in range(self.layers-1)] #7 5 3 1
        self._arch_param_names = ["routesdw0","routesdw1","routesdw2",
                                  "routesup0","routesup1","routesup2",
                                  "routessk01","routessk12","routessk23","routessk34",
                                  "routessk02","routessk13","routessk24",
                                  "routessk03","routessk14","routessk04"]
        self._initialize_alphas ()

        for layer in range(self.layers):
            if layer == 0:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: 
                        self.layermodule.append(DoubleBlock(input_channels, nb_filter[0]))                           
                    else :
                        self.layermodule.append(DoubleBlock(self.inifilter * np.power(self.multiplier,stage-1) , self.inifilter * np.power(self.multiplier,stage)))
                self.module.append(self.layermodule)
            else :
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: #第一層 (32+64,32),(32*2+64,32)
                        self.layermodule.append(ConcatBlock(self.inifilter * np.power(self.multiplier,stage) * layer +self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                    if stage > 0 : #(64*2+128,64),(64*3+128,64),(64*4+128,64)
                        self.layermodule.append(ConcatBlock(self.inifilter * np.power(self.multiplier,stage) * (layer+1) +self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                for stage in range(self.layers-layer-1): #ex layer1 有四個stage,只有三個downsampling
                    self.layermodule.append(ConcatBlock(self.inifilter * np.power(self.multiplier,stage),self.inifilter * np.power(self.multiplier,stage+1))) # for downsampling + maxpooling
                self.module.append(self.layermodule)
                
    def _initialize_alphas(self):
        routesdw0 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[0], nn.Parameter(routesdw0))
  
        routesdw1 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[1], nn.Parameter(routesdw1))
        
        routesdw2 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[2], nn.Parameter(routesdw2))

        routesup0 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[3], nn.Parameter(routesup0))
  
        routesup1 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[4], nn.Parameter(routesup1))
        
        routesup2 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[5], nn.Parameter(routesup2))
        
        routessk01 = nn.Parameter(1e-3*torch.ones(4).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[6], nn.Parameter(routessk01))
        
        routessk12 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[7], nn.Parameter(routessk12))
        
        routessk23 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[8], nn.Parameter(routessk23))
        
        routessk34 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[9], nn.Parameter(routessk34))
        
        routessk02 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[10], nn.Parameter(routessk02))
        
        routessk13 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[11], nn.Parameter(routessk13))
        
        routessk24 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[12], nn.Parameter(routessk24))
        
        routessk03 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[13], nn.Parameter(routessk03))
        
        routessk14 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[14], nn.Parameter(routessk14))
        
        routessk04 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[15], nn.Parameter(routessk04))
        
        
    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer0.append(self.module[layer][stage](input))        
                    else :
                        layer0.append(self.module[layer][stage](self.pool(layer0[stage-1])))
                        
            elif layer == 1:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],torch.sigmoid(self.routesup0[stage]) * self.up(layer0[stage+1])], 1)))
                    if stage > 0 and stage < (len(mlayer)+1)//2-1: 
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],torch.sigmoid(self.routesup0[stage]) * self.up(layer0[stage+1]),torch.sigmoid(self.routesdw0[stage-1]) * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer1[stage-1]))], 1)))
                    if  stage == (len(mlayer)+1)//2-1: #最後一層不乘機率
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],self.up(layer0[stage+1]),torch.sigmoid(self.routesdw0[stage-1]) * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer1[stage-1]))], 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],torch.sigmoid(self.routesup1[stage]) * self.up(layer1[stage+1])], 1)))
                    if stage > 0  and stage < (len(mlayer)+1)//2-1: 
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],torch.sigmoid(self.routesup1[stage]) * self.up(layer1[stage+1]),torch.sigmoid(self.routesdw1[stage-1]) * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer2[stage-1]))], 1)))
                    if  stage == (len(mlayer)+1)//2-1: #最後一層不乘機率
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],self.up(layer1[stage+1]),torch.sigmoid(self.routesdw1[stage-1]) * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer2[stage-1]))], 1)))
                        
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],torch.sigmoid(self.routesup2[stage]) * self.up(layer2[stage+1])], 1)))
                    if stage > 0  and stage < (len(mlayer)+1)//2-1: 
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],torch.sigmoid(self.routesup2[stage]) * self.up(layer2[stage+1]),torch.sigmoid(self.routesdw2[stage-1]) * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer3[stage-1]))], 1)))
                    if  stage == (len(mlayer)+1)//2-1: #最後一層不乘機率
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],self.up(layer2[stage+1]),torch.sigmoid(self.routesdw2[stage-1])*self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer3[stage-1]))], 1)))
                        
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    layer4.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk04[stage]) * layer0[stage],torch.sigmoid(self.routessk14[stage]) * layer1[stage],torch.sigmoid(self.routessk24[stage]) * layer2[stage],torch.sigmoid(self.routessk34[stage]) * layer3[stage],self.up(layer3[stage+1])], 1)))
                        
        return self.final(layer4[0])        
        
        
    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]
    
    
       
class ALLSearch_Decode_V1(nn.Module):
    def __init__(self, input_channels,n_classes,weight):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.multiplier = 2
        self.layers = 5
        self.module = nn.ModuleList()
        nb_filter = [32, 64, 128, 256, 512]
        self.inifilter = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final = nn.Conv2d(self.inifilter,n_classes , kernel_size=1)
        
        self.weight = weight 
        self.weight_decode = self.roundsigmoid()
        self.routesdw0 = self.weight_decode[0]
        self.routesdw1 = self.weight_decode[1]
        self.routesdw2 = self.weight_decode[2]
        self.routesup0 = self.weight_decode[3]
        self.routesup1 = self.weight_decode[4]
        self.routesup2 = self.weight_decode[5]
        self.routessk01 = self.weight_decode[6]
        self.routessk12 = self.weight_decode[7]
        self.routessk23 = self.weight_decode[8]
        self.routessk34 = self.weight_decode[9]
        self.routessk02 = self.weight_decode[10]
        self.routessk13 = self.weight_decode[11]
        self.routessk24 = self.weight_decode[12]
        self.routessk03 = self.weight_decode[13]
        self.routessk14 = self.weight_decode[14]
        self.routessk04 = self.weight_decode[15]
        
        
        for layer in range(self.layers):
            if layer == 0:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: 
                        self.layermodule.append(DoubleBlock(input_channels, nb_filter[0]))                           
                    else :
                        self.layermodule.append(DoubleBlock(self.inifilter * np.power(self.multiplier,stage-1) , self.inifilter * np.power(self.multiplier,stage)))
                self.module.append(self.layermodule)
            else :
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: #第一層 (32+64,32),(32*2+64,32)
                        self.layermodule.append(ConcatBlock(self.inifilter * np.power(self.multiplier,stage) * layer +self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                    if stage > 0 : #(64*2+128,64),(64*3+128,64),(64*4+128,64)
                        self.layermodule.append(ConcatBlock(self.inifilter * np.power(self.multiplier,stage) * (layer) + self.inifilter * np.power(self.multiplier,stage) + self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                for stage in range(self.layers-layer-1): #ex layer1 有四個stage,只有三個downsampling
                    self.layermodule.append(ConcatBlock(self.inifilter * np.power(self.multiplier,stage),self.inifilter * np.power(self.multiplier,stage+1))) # for downsampling + maxpooling
                self.module.append(self.layermodule)
        
    def roundsigmoid(self):
        total = 0.
        lst = []
        lstnp = []
        result = []
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                tmp = torch.sigmoid(self.weight[i][j])
                total += tmp
                lstnp.append(tmp)
        lstnp.sort()
        #print(lstnp)
        #print(lstnp[15])
        medium = lstnp[15]
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                tmp = torch.sigmoid(self.weight[i][j])
                #if tmp >= total/32:
                #if tmp > medium :
                if tmp >= 0.5 :
                    lst.append(1)
                else :
                    lst.append(0)
            result.append(lst)
            lst = []
        print(result)
        return result
    def getitem(self,tensor):
        result = []
        for i in range(len(tensor)):
            result.append(nn.Parameter(torch.tensor(tensor[i], requires_grad=False).cuda(),requires_grad=False))
        return result

    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer0.append(self.module[layer][stage](input))        
                    else :
                        layer0.append(self.module[layer][stage](self.pool(layer0[stage-1])))
                        
            elif layer == 1:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer1.append(self.module[layer][stage](torch.cat([self.routessk01[stage] * layer0[stage],self.routesup0[stage] * self.up(layer0[stage+1])], 1)))
                    if stage > 0 and stage < (len(mlayer)+1)//2-1: 
                        layer1.append(self.module[layer][stage](torch.cat([self.routessk01[stage] * layer0[stage],self.routesup0[stage] * self.up(layer0[stage+1]), self.routesdw0[stage-1] * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer1[stage-1]))], 1)))
                    if  stage == (len(mlayer)+1)//2-1: #最後一層up不乘機率 
                        layer1.append(self.module[layer][stage](torch.cat([self.routessk01[stage] * layer0[stage],self.up(layer0[stage+1]),  self.routesdw0[stage-1] * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer1[stage-1]))], 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer2.append(self.module[layer][stage](torch.cat([self.routessk02[stage] * layer0[stage],self.routessk12[stage] * layer1[stage],self.routesup1[stage] * self.up(layer1[stage+1])], 1)))
                    if stage > 0  and stage < (len(mlayer)+1)//2-1: 
                        layer2.append(self.module[layer][stage](torch.cat([self.routessk02[stage] * layer0[stage],self.routessk12[stage] * layer1[stage],self.routesup1[stage] * self.up(layer1[stage+1]),self.routesdw1[stage-1] * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer2[stage-1]))], 1))) 
                    if  stage == (len(mlayer)+1)//2-1: #最後一層不乘機率
                        layer2.append(self.module[layer][stage](torch.cat([self.routessk02[stage] * layer0[stage],self.routessk12[stage] * layer1[stage],self.up(layer1[stage+1]),self.routesdw1[stage-1] * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer2[stage-1]))], 1)))
                   
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer3.append(self.module[layer][stage](torch.cat([self.routessk03[stage] * layer0[stage],self.routessk13[stage] * layer1[stage],self.routessk23[stage] * layer2[stage],self.routesup2[stage] * self.up(layer2[stage+1])], 1)))
                    if stage > 0  and stage < (len(mlayer)+1)//2-1: 
                        layer3.append(self.module[layer][stage](torch.cat([self.routessk03[stage] * layer0[stage],self.routessk13[stage] * layer1[stage],self.routessk23[stage] * layer2[stage],self.routesup2[stage] * self.up(layer2[stage+1]),self.routesdw2[stage-1] * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer3[stage-1]))], 1)))
                        
                    if  stage == (len(mlayer)+1)//2-1: #最後一層不乘機率
                        layer3.append(self.module[layer][stage](torch.cat([self.routessk03[stage] * layer0[stage],self.routessk13[stage] * layer1[stage],self.routessk23[stage] * layer2[stage],self.up(layer2[stage+1]),self.routesdw2[stage-1] * self.module[layer][stage+(len(mlayer)+1)//2-1](self.pool(layer3[stage-1]))], 1)))
                    
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    layer4.append(self.module[layer][stage](torch.cat([self.routessk04[stage] * layer0[stage],self.routessk14[stage] * layer1[stage],self.routessk24[stage] * layer2[stage],self.routessk34[stage] * layer3[stage],self.up(layer3[stage+1])], 1)))
                        
                        
            
        return self.final(layer4[0]) 
    
class ALLSearch_V2(nn.Module):
    def __init__(self, input_channels,n_classes):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.multiplier = 2
        self.layers = 5
        self.module = nn.ModuleList()
        nb_filter = [32, 64, 128, 256, 512]
        self.inifilter = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final = nn.Conv2d(self.inifilter,n_classes , kernel_size=1)
        #self.routes = [nn.Parameter(torch.randn(2*(self.layers-1-i)-1).cuda(), requires_grad=True) for i in range(self.layers-1)] #7 5 3 1
        self._arch_param_names = ["routesdw0","routesdw1","routesdw2",
                                  "routesup0","routesup1","routesup2",
                                  "routessk01","routessk12","routessk23","routessk34",
                                  "routessk02","routessk13","routessk24",
                                  "routessk03","routessk14","routessk04"]
        self._initialize_alphas ()

        for layer in range(self.layers):
            if layer == 0:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: 
                        self.layermodule.append(DoubleBlock(input_channels, self.inifilter))                           
                    else :
                        self.layermodule.append(DoubleBlock(self.inifilter * np.power(self.multiplier,stage-1) , self.inifilter * np.power(self.multiplier,stage)))
                self.module.append(self.layermodule)
            else :
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: #第一層 (32+64,32),(32*2+64,32)
                        self.layermodule.append(DoubleBlock(self.inifilter * np.power(self.multiplier,stage) * layer + self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                    if stage > 0 : #(64*1+32+128,64),(64*2+32+128,64),(64*3+128,64)
                        self.layermodule.append(DoubleBlock(self.inifilter * np.power(self.multiplier,stage) * layer + self.inifilter * np.power(self.multiplier,stage-1) + self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                
                self.module.append(self.layermodule)
                
    def _initialize_alphas(self):
        routesdw0 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[0], nn.Parameter(routesdw0))
  
        routesdw1 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[1], nn.Parameter(routesdw1))
        
        routesdw2 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[2], nn.Parameter(routesdw2))

        routesup0 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[3], nn.Parameter(routesup0))
  
        routesup1 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[4], nn.Parameter(routesup1))
        
        routesup2 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[5], nn.Parameter(routesup2))
        
        routessk01 = nn.Parameter(1e-3*torch.ones(4).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[6], nn.Parameter(routessk01))
        
        routessk12 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[7], nn.Parameter(routessk12))
        
        routessk23 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[8], nn.Parameter(routessk23))
        
        routessk34 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[9], nn.Parameter(routessk34))
        
        routessk02 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[10], nn.Parameter(routessk02))
        
        routessk13 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[11], nn.Parameter(routessk13))
        
        routessk24 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[12], nn.Parameter(routessk24))
        
        routessk03 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[13], nn.Parameter(routessk03))
        
        routessk14 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[14], nn.Parameter(routessk14))
        
        routessk04 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[15], nn.Parameter(routessk04))
        
        
    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer0.append(self.module[layer][stage](input))        
                    else :
                        layer0.append(self.module[layer][stage](self.pool(layer0[stage-1])))
                        
            elif layer == 1:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],torch.sigmoid(self.routesup0[stage]) * self.up(layer0[stage+1])], 1)))
                    if stage > 0 and stage < len(mlayer) - 1: 
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],torch.sigmoid(self.routesup0[stage]) * self.up(layer0[stage+1]),torch.sigmoid(self.routesdw0[stage-1]) * self.pool(layer1[stage-1])], 1)))
                    if  stage == len(mlayer) - 1: #最後一層不乘機率
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],self.up(layer0[stage+1]),torch.sigmoid(self.routesdw0[stage-1]) * self.pool(layer1[stage-1])], 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],torch.sigmoid(self.routesup1[stage]) * self.up(layer1[stage+1])], 1)))
                    if stage > 0  and stage < len(mlayer) - 1: 
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],torch.sigmoid(self.routesup1[stage]) * self.up(layer1[stage+1]),torch.sigmoid(self.routesdw1[stage-1]) * self.pool(layer2[stage-1])], 1)))
                    if  stage == len(mlayer) - 1: #最後一層不乘機率
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],self.up(layer1[stage+1]),torch.sigmoid(self.routesdw1[stage-1]) * self.pool(layer2[stage-1])], 1)))
                        
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],torch.sigmoid(self.routesup2[stage]) * self.up(layer2[stage+1])], 1)))
                    if stage > 0  and stage < len(mlayer) - 1: 
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],torch.sigmoid(self.routesup2[stage]) * self.up(layer2[stage+1]),torch.sigmoid(self.routesdw2[stage-1]) * self.pool(layer3[stage-1])], 1)))
                    if  stage == len(mlayer) - 1: #最後一層不乘機率
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],self.up(layer2[stage+1]),torch.sigmoid(self.routesdw2[stage-1]) * self.pool(layer3[stage-1])], 1)))
                        
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    layer4.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk04[stage]) * layer0[stage],torch.sigmoid(self.routessk14[stage]) * layer1[stage],torch.sigmoid(self.routessk24[stage]) * layer2[stage],torch.sigmoid(self.routessk34[stage]) * layer3[stage],self.up(layer3[stage+1])], 1)))
                        
        return self.final(layer4[0])        
        
        
    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]
    
    
class ALLSearch_Decode_V2(nn.Module):
    def __init__(self, input_channels,n_classes,weight):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.multiplier = 2
        self.layers = 5
        self.module = nn.ModuleList()
        nb_filter = [32, 64, 128, 256, 512]
        self.inifilter = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final = nn.Conv2d(self.inifilter,n_classes , kernel_size=1)
        
        self.weight = weight 
        self.weight_decode = self.roundsigmoid()
        self.routesdw0 = self.weight_decode[0]
        self.routesdw1 = self.weight_decode[1]
        self.routesdw2 = self.weight_decode[2]
        self.routesup0 = self.weight_decode[3]
        self.routesup1 = self.weight_decode[4]
        self.routesup2 = self.weight_decode[5]
        self.routessk01 = self.weight_decode[6]
        self.routessk12 = self.weight_decode[7]
        self.routessk23 = self.weight_decode[8]
        self.routessk34 = self.weight_decode[9]
        self.routessk02 = self.weight_decode[10]
        self.routessk13 = self.weight_decode[11]
        self.routessk24 = self.weight_decode[12]
        self.routessk03 = self.weight_decode[13]
        self.routessk14 = self.weight_decode[14]
        self.routessk04 = self.weight_decode[15]
        
        
        for layer in range(self.layers):
            if layer == 0:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: 
                        self.layermodule.append(DoubleBlock(input_channels, nb_filter[0]))                           
                    else :
                        self.layermodule.append(DoubleBlock(self.inifilter * np.power(self.multiplier,stage-1) , self.inifilter * np.power(self.multiplier,stage)))
                self.module.append(self.layermodule)
                
            elif layer == 1:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0:
                        if self.routessk01[stage] + self.routesup0[stage] == 0:
                            self.routessk01[stage] = 1
                        self.layermodule.append(CBBlock(self.routessk01[stage] * self.inifilter * np.power(self.multiplier,stage) + self.routesup0[stage] * self.inifilter * np.power(self.multiplier,stage+1), self.inifilter * np.power(self.multiplier,stage)))
                    
                    elif stage > 0 and stage < self.layers-layer-1:
                        if self.routessk01[stage] + self.routesup0[stage] + self.routesdw0[stage-1] == 0.:
                            self.routessk01[stage] = 1     
                        self.layermodule.append(CBBlock((self.routessk01[stage]) * self.inifilter * np.power(self.multiplier,stage) + self.routesdw0[stage-1] * self.inifilter * np.power(self.multiplier,stage-1)+ self.routesup0[stage] * self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                            
                    else :
                        self.layermodule.append(CBBlock(self.routessk01[stage] * self.inifilter * np.power(self.multiplier,stage) + self.routesdw0[stage-1] * self.inifilter * np.power(self.multiplier,stage-1) + self.inifilter * np.power(self.multiplier,stage+1), self.inifilter * np.power(self.multiplier,stage)))
                
                self.module.append(self.layermodule)
                
            elif layer == 2:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0:
                        if self.routessk02[stage] + self.routessk12[stage] + self.routesup1[stage] == 0:
                            self.routessk12[stage] = 1.   
                        self.layermodule.append(CBBlock((self.routessk02[stage] + self.routessk12[stage]) * self.inifilter * np.power(self.multiplier,stage) + self.routesup1[stage] * self.inifilter * np.power(self.multiplier,stage+1), self.inifilter * np.power(self.multiplier,stage)))
                    
                    elif stage > 0 and stage < self.layers-layer-1:
                        if self.routessk02[stage] + self.routessk12[stage] + self.routesup1[stage] + self.routesdw1[stage-1] == 0:
                            self.routessk12[stage] = 1     
                        self.layermodule.append(CBBlock((self.routessk02[stage] + self.routessk12[stage] ) * self.inifilter * np.power(self.multiplier,stage) + self.routesdw1[stage-1]* self.inifilter * np.power(self.multiplier,stage-1) + self.routesup1[stage] * self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                            
                    else :
                        self.layermodule.append(CBBlock((self.routessk02[stage] + self.routessk12[stage]) * self.inifilter * np.power(self.multiplier,stage) + self.routesdw1[stage-1] * self.inifilter * np.power(self.multiplier,stage-1) + self.inifilter * np.power(self.multiplier,stage+1), self.inifilter * np.power(self.multiplier,stage)))
                
                self.module.append(self.layermodule)
                
                
            elif layer == 3:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0:
                        if self.routessk03[stage] + self.routessk13[stage] + self.routessk23[stage] + self.routesup2[stage] == 0:
                            self.routessk23[stage] = 1   
                        self.layermodule.append(CBBlock((self.routessk03[stage] + self.routessk13[stage] + self.routessk23[stage]) * self.inifilter * np.power(self.multiplier,stage) + self.routesup2[stage] * self.inifilter * np.power(self.multiplier,stage+1), self.inifilter * np.power(self.multiplier,stage)))
                    
                    else:    
                        self.layermodule.append(CBBlock((self.routessk03[stage] + self.routessk13[stage] + self.routessk23[stage] ) * self.inifilter * np.power(self.multiplier,stage) + self.routesdw2[stage-1] * self.inifilter * np.power(self.multiplier,stage-1) + self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                            
                
                self.module.append(self.layermodule)
                
            elif layer == 4:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):  
                    self.layermodule.append(CBBlock((self.routessk04[stage] + self.routessk14[stage] + self.routessk24[stage] + self.routessk34[stage]) * self.inifilter * np.power(self.multiplier,stage) + self.inifilter * np.power(self.multiplier,stage+1), self.inifilter * np.power(self.multiplier,stage)))
                
                self.module.append(self.layermodule)           
            
        
    def roundsigmoid(self):
        total = 0.
        lst = []
        lstnp = []
        result = []
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                tmp = torch.sigmoid(self.weight[i][j])
                total += tmp
                lstnp.append(tmp)
        lstnp.sort()
        #print(lstnp)
        #print(lstnp[15])
        medium = lstnp[15]
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                tmp = torch.sigmoid(self.weight[i][j])
                #if tmp >= total/32:
                #if tmp > medium :
                if tmp >= 0.5 :
                    lst.append(1)
                else :
                    lst.append(0)
            result.append(lst)
            lst = []
        print(result)
        return result
    def getitem(self,tensor):
        result = []
        for i in range(len(tensor)):
            result.append(nn.Parameter(torch.tensor(tensor[i], requires_grad=False).cuda(),requires_grad=False))
        return result

    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer0.append(self.module[layer][stage](input))        
                    else :
                        layer0.append(self.module[layer][stage](self.pool(layer0[stage-1])))
                        
            elif layer == 1:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        concatlst = []
                        if self.routessk01[stage] == 1:
                            concatlst.append(layer0[stage])
                        if self.routesup0[stage] == 1:
                            concatlst.append(self.up(layer0[stage+1]))
                        layer1.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                    if stage > 0 and stage < len(mlayer) - 1: 
                        concatlst = []
                        if self.routessk01[stage] == 1:
                            concatlst.append(layer0[stage])
                        if self.routesup0[stage] == 1:
                            concatlst.append(self.up(layer0[stage+1]))
                        if self.routesdw0[stage-1] == 1:
                            concatlst.append(self.pool(layer1[stage-1]))
                        layer1.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                    if  stage == len(mlayer) - 1: #最後一層up不乘機率 
                        concatlst = []
                        if self.routessk01[stage] == 1:
                            concatlst.append(layer0[stage])
                        if self.routesdw0[stage-1] == 1:
                            concatlst.append(self.pool(layer1[stage-1]))
                        concatlst.append(self.up(layer0[stage+1]))
                        layer1.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        concatlst = []
                        if self.routessk02[stage] == 1:
                            concatlst.append(layer0[stage])
                        if self.routessk12[stage] == 1:
                            concatlst.append(layer1[stage])
                        if self.routesup1[stage] == 1:
                            concatlst.append(self.up(layer1[stage+1]))
                        layer2.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                    if stage > 0  and stage < len(mlayer) - 1: 
                        concatlst = []
                        if self.routessk02[stage] == 1:
                            concatlst.append(layer0[stage])
                        if self.routessk12[stage] == 1:
                            concatlst.append(layer1[stage])
                        if self.routesup1[stage] == 1:
                            concatlst.append(self.up(layer1[stage+1]))
                        if self.routesdw1[stage-1] == 1:
                            concatlst.append(self.pool(layer2[stage-1]))
                        layer2.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                    if  stage == len(mlayer) - 1: #最後一層不乘機率
                        concatlst = []
                        if self.routessk02[stage] == 1:
                            concatlst.append(layer0[stage])
                        if self.routessk12[stage] == 1:
                            concatlst.append(layer1[stage])
                        if self.routesdw1[stage-1] == 1:
                            concatlst.append(self.pool(layer2[stage-1]))
                        concatlst.append(self.up(layer1[stage+1]))
                        layer2.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                        
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        concatlst = []
                        if self.routessk03[stage] == 1:
                            concatlst.append(layer0[stage]) 
                        if self.routessk13[stage] == 1:
                            concatlst.append(layer1[stage]) 
                        if self.routessk23[stage] == 1:
                            concatlst.append(layer2[stage]) 
                        if self.routesup2[stage] == 1:
                            concatlst.append(self.up(layer2[stage+1])) 
                        layer3.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                    else : 
                        concatlst = []
                        if self.routessk03[stage] == 1:
                            concatlst.append(layer0[stage]) 
                        if self.routessk13[stage] == 1:
                            concatlst.append(layer1[stage]) 
                        if self.routessk23[stage] == 1:
                            concatlst.append(layer2[stage]) 
                        if self.routesdw2[stage-1] == 1:
                            concatlst.append(self.pool(layer3[stage-1])) 
                        concatlst.append(self.up(layer2[stage+1])) 
                        layer3.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                    
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    concatlst = []
                    if self.routessk04[stage] == 1:
                        concatlst.append(layer0[stage]) 
                    if self.routessk14[stage] == 1:
                        concatlst.append(layer1[stage]) 
                    if self.routessk24[stage] == 1:
                        concatlst.append(layer2[stage]) 
                    if self.routessk34[stage] == 1:
                        concatlst.append(layer3[stage]) 
                    concatlst.append(self.up(layer3[stage+1])) 
                    layer4.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                       
        return self.final(layer4[0]) 
    
if __name__ == '__main__':
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.rand(size=(1,3,256,256))
    y = torch.rand(size=(1,3,256,256))
    model = ALLSearch_V2(3,3).to(device)
    
    '''
    model_s = ALLSearch(3,3).to(device)
    
    model_s.load_state_dict(torch.load("./models/ALLSearch/ALLSearch_best.pth"
                                         ,map_location='cpu'))
    para = model_s.arch_parameters ()
    # load pretrained model
    partial = torch.load("./models/ALLSearch/ALLSearch_best.pth", map_location='cpu')
    model = ALLSearch_Decode_V2(3,3,para).to(device)
    
    
    
    
    #print(summary(model,(3,256,256)))
    flops, params = profile(model, inputs=(x.to(device), ))
    print(flops,params)
    ''' 
    #print (model.arch_parameters ())
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    epoch_loss = 0
    for i in range(10):
        inputs = x.to(device)
        labels = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #print(loss)
        print("epoch:",i)
        print (model.arch_parameters ())
        #print(parm0,parm1,parm2,parm3)
    #for param in model.parameters():
        #print(type(param.data), param.size())
        
       
        
        
    
    