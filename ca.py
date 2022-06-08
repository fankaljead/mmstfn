import torch
from torch import nn


class CALayer(nn.Module):
    def __init__(self, channel_h, channel_w, 
                 channel_c, r_h=32, r_w=32, r_c=4): # 480 480 24
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # h*w 的平均 将 n*c*h*w --> n*c*1*1

        self.conv_h = nn.Sequential(
            nn.Conv2d(channel_h, channel_h // r_h, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_h // r_h, channel_h, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.conv_w = nn.Sequential(
            nn.Conv2d(channel_w, channel_w // r_w, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_w // r_w, channel_w, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.conv_c = nn.Sequential(
            nn.Conv2d(channel_c, channel_c // r_c, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_c // r_c, channel_c, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # n c h w

        # h
        x_h = x.permute(0, 2, 1, 3)
        y_h = self.avg_pool(x_h)  # n h c w ---> n h 1x1
        y_h = x_h * self.conv_h(y_h)

        # w
        x_w = y_h.permute(0, 3, 2, 1)
        y_w = self.avg_pool(x_w)  # n w c h  ---> n w 1x1
        y_w = x_w * self.conv_w(y_w)

        # c
        x_c = y_w.permute(0, 2, 3, 1)
        y_c = self.avg_pool(x_c)  # n c h w  ---> n c 1x1
        y = x_c * self.conv_c(y_c)
        return y