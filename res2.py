#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import math


class Res2BottleNeckX(nn.Module):
    expansion = 4
    def __init__(self, inplanes, 
                 planes, 
                 cardinality, stride=1, 
                 baseWidth=4, 
                 downsample=None, scale = 4, 
                 stype='normal'):
        
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Res2BottleNeckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C*scale, 
                               kernel_size=1, stride=1, 
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale - 1
        
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
            
        convs = []
        bns = []
        
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, 
                                 stride = stride, padding=1, 
                                 groups=C, bias=False))
          bns.append(nn.BatchNorm2d(D*C))
            
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D*C*scale, planes, 
                               kernel_size=1, stride=1, 
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale
        
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        residual = self.relu(self.conv(x))
        
        print("bottlenext========residual.shape")
        print(residual.shape)

        out = self.conv1(x)
        print("bottlenext========out = self.conv1(x)")
        print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            
            if i == 0:
                out = sp
             else:
                 out = torch.cat((out, sp), 1)
            
        if self.scale != 1 and self.stype == 'normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype == 'stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
 

# 
class Res2NextBlock(nn.Module):
    
    def __init__(self, 
                 inplanes, planes,
                 block, 
                 cardinality, 
                 layers=[3, 4, 6, 3], 
                 layer_planes=[30, 24, 30, 24],
                 baseWidth=4, 
                 scale=4):
#         Res2NextBlock(inplanes=24,
#                                     planes=24,
#                                     block=Res2BottleNeckX,
#                                     cardinality=8,
#                                     scale=4,
#                                     layers=[3, 4, 6, 3],
#                                     layer_planes=[30, 24, 30, 24]
#                                    )
        
        super(Res2NextBlock, self).__init__()
        
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = inplanes # 24
        self.planes = planes
        self.layers = layers
        self.layer_planes = layer_planes
        self.scale = scale
        
        self.conv1 = nn.Conv2d(self.inplanes, self.planes, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
#         layers_modules = []
        
#         print("====layer_planes=======")
#         print(self.layer_planes)
#         print("====layer_planes=======")
        
#         for i in range(len(layer_planes)):
#             if i != 0:
#                 layers_modules.append(self._make_layer(block, self.layer_planes[i], self.layers[i]))
#             else:
#                 layers_modules.append(self._make_layer(block, self.layer_planes[i], self.layers[i], 1))
       
        
#         self.layers_modules = nn.Sequential(*layers_modules)
        
        self.layer1 = self._make_layer(block, 24, layers[0])
#         self.layer2 = self._make_layer(block, 24, layers[1])
#         self.layer3 = self._make_layer(block, 30, layers[2], 2)
#         self.layer4 = self._make_layer(block, 24, layers[3], 2)
        
    
    def forward(self, x):
        out = self.conv1(x)
        print("res2=====out = self.conv1(x)")
        print(out.shape)
        out = self.relu(out)
        
#         out = self.layers_modules(out)
        out = self.layer1(out)
        print("res2=====out = out = self.layer1(out)")
        print(out.shape)
#         out = self.layer2(out)
        
        out = self.relu(out)
        
        return out
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

        layers = []
 
        layers.append(block(inplanes=self.inplanes, 
                            planes=planes, 
                            baseWidth=self.baseWidth, 
                            cardinality=self.cardinality, 
                            stride=stride, 
                            downsample=downsample, 
                            scale=self.scale, stype='stage'))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes,
                                planes=planes, 
                                baseWidth=self.baseWidth, 
                                cardinality=self.cardinality, 
                                scale=self.scale))

        return nn.Sequential(*layers)