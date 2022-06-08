import torch
from torch import nn
from involution import Involution


NUM_BANDS = 4

def conv3x3(in_channels, 
            out_channels, 
            stride=1):
    
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        
        nn.Conv2d(in_channels, 
                  out_channels, 
                  3, 
                  stride=stride)
    )



class REncoder(nn.Sequential):
    
    def __init__(self):
        
        channels = [NUM_BANDS * 3, 24] # 12, 24
        
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            Involution(channels=channels[1], kernel_size=7, stride=1, group_channels=12, reduction_ratio=4),
            nn.ReLU(True),
        )



class FEncoder(nn.Sequential):
    
    def __init__(self):
        
        channels = [NUM_BANDS, 16, 24] # 4, 16, 24
        
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]), # 4 16
            nn.ReLU(True),
            
            conv3x3(channels[1], channels[2]), # 16 24
            nn.ReLU(True),
            
            conv3x3(channels[2], channels[2]), # 24 24
            Involution(channels=channels[2], kernel_size=7, stride=1, group_channels=12, reduction_ratio=4),
            nn.ReLU(True),
        )




class Decoder(nn.Sequential):
    
    def __init__(self):
        
        channels = [48, 24, NUM_BANDS] # 48 24 4
        
        super(Decoder, self).__init__(
            
            conv3x3(channels[0], channels[1]), # 48 24
            Involution(channels=channels[1], kernel_size=7, stride=1, group_channels=12, reduction_ratio=4),
            nn.ReLU(True),
            
            conv3x3(channels[1], channels[2]), # 24 4
        )