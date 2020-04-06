# -*- coding: utf-8 -*-
import numpy as np
from torch import nn ,optim
from torch.nn import functional as F
from torch.autograd import Variable
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

class NasUNet(nn.Module):
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
        self._arch_param_names = ["routes0", "routes1", "routes2", "routes3"]
        self._initialize_alphas ()
        #self.routes0 = nn.Parameter(Variable(1e-3*torch.ones(7).cuda(), requires_grad=True))
        #self.routes1 = nn.Parameter(Variable(1e-3*torch.ones(5).cuda(), requires_grad=True))
        #self.routes2 = nn.Parameter(Variable(1e-3*torch.ones(3).cuda(), requires_grad=True))
        #self.routes3 = nn.Parameter(Variable(1e-3*torch.ones(1).cuda(), requires_grad=True))
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
                    self.layermodule.append(DoubleBlock(self.inifilter * np.power(self.multiplier,stage)+self.inifilter * np.power(self.multiplier,stage+1) , self.inifilter * np.power(self.multiplier,stage)))
                self.module.append(self.layermodule)
    def _initialize_alphas(self):
        routes0 = nn.Parameter(1e-3*torch.ones(7).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[0], nn.Parameter(routes0))
  
        routes1 = nn.Parameter(1e-3*torch.ones(5).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[1], nn.Parameter(routes1))
        
        routes2 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[2], nn.Parameter(routes2))
        
        routes3 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[3], nn.Parameter(routes3))
        
    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]
            
    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        weight0 = torch.sigmoid(self.routes0)
        weight1 = torch.sigmoid(self.routes1)
        weight2 = torch.sigmoid(self.routes2)
        weight3 = torch.sigmoid(self.routes3)
        
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer0.append(self.module[layer][stage](input))        
                    else :
                        layer0.append(self.module[layer][stage](self.pool(layer0[stage-1])))
                        
            elif layer == 1:
                for stage ,layers in enumerate(mlayer) :
                    if stage < len(mlayer)-1:
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes0[stage])*layer0[stage],torch.sigmoid(self.routes0[stage+1])*self.up(layer0[stage+1])], 1)))
                    else : 
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes0[stage])*layer0[stage],self.up(layer0[stage+1])], 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage < len(mlayer)-1:
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes1[stage])*layer1[stage],torch.sigmoid(self.routes1[stage+1])*self.up(layer1[stage+1])], 1)))
                    else : 
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes1[stage])*layer1[stage],self.up(layer1[stage+1])], 1)))
                    
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage < len(mlayer)-1:
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes2[stage])*layer2[stage],torch.sigmoid(self.routes2[stage+1])*self.up(layer2[stage+1])], 1)))
                    else : 
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes2[stage])*layer2[stage],self.up(layer2[stage+1])], 1)))
                    
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    if stage < len(mlayer)-1:
                        layer4.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes3[stage])*layer3[stage],torch.sigmoid(self.routes3[stage+1])*self.up(layer3[stage+1])], 1)))
                    else : 
                        layer4.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routes3[stage])*layer3[stage],self.up(layer3[stage+1])], 1)))
        #x = self.final(x)
        return self.final(layer4[0]) 
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(size=(1,1,256,256))
    y = torch.rand(size=(1,1,256,256))
    model = NasUNet(1,1).to(device)
    print (model.arch_parameters ())
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    epoch_loss = 0
    for i in range(10):
        inputs = x.to(device)
        labels = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs ,parm0, parm1 , parm2, parm3= model(inputs)
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
