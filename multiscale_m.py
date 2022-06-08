import torch
from torch import nn

from res2net import Res2Net
from coordinate_attention import CoordAtt
from ca import CALayer

class MultiscalFeature_Block(nn.Module):
    def __init__(self, ):
        super(MultiscalFeature_Block, self).__init__()
        self.scaleca1 = Res2Net(inplanes=24, planeslist=[30, 24], scales = [5, 4])
        
        self.scaleca2_1 = Res2Net(inplanes=24, planeslist=[30, 24], scales = [5, 4])
        self.scaleca2_2 = Res2Net(inplanes=24, planeslist=[30, 24], scales = [5, 4])
        
        self.scaleca3_1 = Res2Net(inplanes=24, planeslist=[30, 24], scales = [5, 4])
        self.scaleca3_2 = Res2Net(inplanes=24, planeslist=[30, 24], scales = [5, 4])
        self.scaleca3_3 = Res2Net(inplanes=24, planeslist=[30, 24], scales = [5, 4])
        
        self.ca = CoordAtt(24, 24, reduction=4)
        
#         self.scalese1 = Res2Net(inplanes=24)
        
#         self.scalese2_1 = Res2Net(inplanes=24)
#         self.scalese2_2 = Res2Net(inplanes=24)
        
#         self.scalese3_1 = Res2Net(inplanes=24)
#         self.scalese3_2 = Res2Net(inplanes=24)
#         self.scalese3_3 = Res2Net(inplanes=24)
        
#         self.se = CALayer(480, 480, 24)



    def forward(self, inputs):

        
        scaleca1 = self.scaleca1(inputs)
        
        scaleca2_1 = self.scaleca2_1(inputs+scaleca1)
        scaleca2 = self.scaleca2_2(inputs+scaleca1 + scaleca2_1)
        
        scaleca3_1 = self.scaleca3_1(inputs+scaleca1+scaleca2)
        scaleca3_2 = self.scaleca3_2(inputs+scaleca1 + scaleca2 + scaleca3_1)
        scaleca3 = self.scaleca3_3(inputs+scaleca1 + scaleca2 + scaleca3_1 + scaleca3_2)
        
        scaleca = self.ca(inputs + scaleca1 + scaleca2 + scaleca3)


#         scalese1 = self.scalese1(inputs)
        
#         scalese2_1 = self.scalese2_1(inputs+scalese1)
#         scalese2 = self.scalese2_2(inputs+scalese1 + scalese2_1)
        
#         scalese3_1 = self.scalese3_1(inputs+scalese1+scalese2)
#         scalese3_2 = self.scalese3_2(inputs+scalese1 + scalese2 + scalese3_1)
#         scalese3 = self.scalese3_3(inputs+scalese1 + scalese2 + scalese3_1 + scalese3_2)
        
#         scalese = self.se(inputs + scalese1 + scalese2 + scalese3)
        
        result = scaleca #+ scalese
        return result
