import torch
from torch import nn
from involution import Involution
from coordinate_attention import CoordAtt
from ca import CALayer
import gc

class Bottle2neckX(nn.Module):


    def __init__(self, inplanes, planes, stride=1, scale = 4, kernel_size=3, dim=1):

        super(Bottle2neckX, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale
        
        self.dim = dim
        convs = []
        
        if dim == 1:        
            D = planes // scale
            C = D
            self.width = D
            for i in range(self.nums):
              convs.append(nn.Conv2d(D, D, kernel_size=kernel_size, stride = stride, padding=1, groups=C, bias=False))
            
        else:
            self.width = 480 // scale
            for i in range(self.nums):
              convs.append(nn.Conv2d(planes, planes, kernel_size=kernel_size, stride = stride, padding=1, groups=planes, bias=False))

        self.convs = nn.ModuleList(convs)


        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv3 = Involution(channels=24, kernel_size=7, stride=1, group_channels=12, reduction_ratio=4)

        self.relu = nn.ReLU(inplace=True)

        self.scale = scale

    def forward(self, x):


        out = self.conv1(x)
        out = self.relu(out)

        spx = torch.split(out, self.width, self.dim)

        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)

            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), self.dim)

   

        out = self.conv3(out)

        out = self.relu(out)

        return out

    
    
class ZBlock(nn.Module):
    def __init__(self, inplanes, planes, dims=[[1], [2, 3]], scales=[[5], [5, 5]]):
        super(ZBlock, self).__init__()
        m = []
        for i in range(len(dims)):
            ms = []
            for j in range(len(dims[i])):
                if j == 0:
                    ms.append(Bottle2neckX(inplanes=inplanes, planes=planes, dim=dims[i][j], scale=scales[i][j]))
                else:
                    ms.append(Bottle2neckX(inplanes=planes, planes=planes, dim=dims[i][j], scale=scales[i][j]))
            m.append(nn.Sequential(*ms))
        
#         print(m)
        self.m = nn.ModuleList(m)
#         self.conv = nn.Conv2d(planes*scales[0][0], planes, 1)

#         self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        outs = []
        
        for mm in self.m:
            outs.append(mm(x))
        out = torch.add(*outs)
#         out = torch.cat(list(outs), 1)
#         out = self.conv(out)
        return out
    
# class ZModule(nn.Module):
#     def __init__(self, inplanes=[24, 24], scales=[[[6], [6, 6]]]):
#         super(ZModule, self).__init__()
        
#         self.zblocks1_1_23 = ZBlock(inplanes[0], inplanes[1], dims=[[1], [2,3]], scales=scales[0])
# #         self.zblocks1_2_13 = ZBlock(inplanes[0], inplanes[1], dims=[[2], [1,3]], scales=scales[0])
# #         self.zblocks1_3_12 = ZBlock(inplanes[0], inplanes[1], dims=[[3], [1,2]], scales=scales[0])
        
        
#         self.ca = CoordAtt(24, 24, reduction=4)
#         self.se = CALayer(480,480,24)
    
#     def forward(self, x):
#         x = self.ca(x)
#         s1_1_23 = self.zblocks1_1_23(x)
# #         s1_2_13 = self.zblocks1_2_13(x)
# #         s1_3_12 = self.zblocks1_3_12(x)
#         s1 = s1_1_23 # + s1_2_13 + s1_3_12
        
        
#         out = self.se(s1)
        
#         return out

class ZModule(nn.Module):
    def __init__(self, planes=[24, 30, 24], dims=[[1], [2, 3]], scales=[[5], [6, 6]]):
        super(ZModule, self).__init__()
        i,j = 0, 0
        self.resznet_c1 = Bottle2neckX(inplanes=planes[0], planes=planes[1], dim=dims[i][j], scale=scales[i][j])
        self.resznet_h1 = Bottle2neckX(inplanes=planes[0], planes=planes[1], dim=dims[i][j], scale=scales[i][j])
        self.resznet_w1 = Bottle2neckX(inplanes=planes[0], planes=planes[1], dim=dims[i][j], scale=scales[i][j])
        self.conv1 = nn.Conv2d(planes[1]*3, planes[1], 1)
        
        i = i + 1
        self.resznet_c2 = Bottle2neckX(inplanes=planes[1], planes=planes[2], dim=dims[i][j], scale=scales[i][j])
        self.resznet_h2 = Bottle2neckX(inplanes=planes[1], planes=planes[2], dim=dims[i][j], scale=scales[i][j])
        self.resznet_w2 = Bottle2neckX(inplanes=planes[1], planes=planes[2], dim=dims[i][j], scale=scales[i][j])
        self.conv2 = nn.Conv2d(planes[2]*3, planes[2], 1)
        
        self.ca = CoordAtt(24, 24, reduction=4)
        
    
    def forward(self, x):
        c1 = self.resznet_c1(x)
        h1 = self.resznet_h1(x)
        w1 = self.resznet_w1(x)
        
        resz1 = self.conv1(torch.cat((c1, h1, w1), 1))
        
        c2 = self.resznet_c2(resz1)
        h2 = self.resznet_h2(resz1)
        w2 = self.resznet_w2(resz1)
        
        resz2 = self.conv2(torch.cat((c2, h2, w2), 1))
        
        ca = self.ca(resz2)

        del c1, h1, w1, resz1, c2, h2, w2, resz2
        gc.collect()
        
        return ca + x
