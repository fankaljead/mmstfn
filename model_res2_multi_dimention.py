 # -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import REncoder, FEncoder, Decoder
# from multiscale_se import MultiscalFeature_Block

from coordinate_attention import CoordAtt
from ca import CALayer
from res2net_multi_dimention import ZModule
from involution import Involution


# 上采样
def interpolate(inputs, 
                size=None, 
                scale_factor=None):
    return F.interpolate(inputs, 
                         size=size, 
                         scale_factor=scale_factor,
                         mode='bilinear', 
                         align_corners=True)  # bicubic bilinear


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.rencoder = REncoder()  # 一层
        self.fencoder = FEncoder()
        
        #self.zd = ZModule(inplanes=[24, 12, 24], scales=[[[3], [3, 3]], [[6], [6, 6]]])
        self.zu = ZModule()
        self.se = CALayer(480,480,24)
        #self.ca = CoordAtt(24, 24, reduction=4)

        self.involution = Involution(channels=24, kernel_size=7, stride=1, group_channels=12, reduction_ratio=4)

        self.decoder = Decoder()

    def forward(self, inputs):
        inputs[0] = interpolate(inputs[0], scale_factor=16)   # 表示 c_k-1
        inputs[-1] = interpolate(inputs[-1], scale_factor=16) # 表示 c_k

        L1feature = self.fencoder(inputs[1]) # f_k-1 提取 f_k-1特征
        
        pairesfeature = self.rencoder(torch.cat((inputs[0], inputs[1], inputs[-1]), 1)) # c_k-1, f_k-1, c_k
        
        fusionFeature = L1feature + pairesfeature
        
        #d = self.zd(fusionFeature)
        zu = self.zu(fusionFeature)

        invo = self.involution(zu)
        
        se = self.se(invo)
        #ca = self.ca(zu)
        
        mu = se # + ca
        

        
        result = self.decoder(torch.cat((mu, L1feature), 1))
        return result

