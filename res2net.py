import torch
from torch import nn

class Bottle2neckX(nn.Module):


    def __init__(self, inplanes, planes, stride=1, scale = 4, kernel_size=3):

        super(Bottle2neckX, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale
        
        D = planes // scale
        C = D
        self.width = D
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(D, D, kernel_size=kernel_size, stride = stride, padding=1, groups=C, bias=False))

        self.convs = nn.ModuleList(convs)


        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.scale = scale

    def forward(self, x):


        out = self.conv1(x)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)

          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)

   

        out = self.conv3(out)

        out = self.relu(out)

        return out
    

class Res2Net(nn.Module):
    
    def __init__(self, inplanes, planeslist=[30, 24, 30, 24], stride=1, scales = [5, 4 ,5, 4], kernel_size=3):
        super(Res2Net, self).__init__()
        
        bottles = []
        for i in range(len(planeslist)):
            if i == 0:
                bottles.append(Bottle2neckX(inplanes=inplanes, planes=planeslist[i], scale=scales[i]))
            else:
                bottles.append(Bottle2neckX(inplanes=planeslist[i-1], planes=planeslist[i], scale=scales[i]))
        
        self.bottles = nn.Sequential(*bottles)
        
        
    def forward(self, x):
        
        return self.bottles(x)