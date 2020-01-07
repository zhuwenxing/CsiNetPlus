
import torch
import torch.nn as nn
from collections import OrderedDict


class ConvLayer(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size,stride=1, activation="LeakyReLu"):
        padding = (kernel_size - 1) // 2
        dict_activation ={"LeakyReLu":nn.LeakyReLU(negative_slope=0.3,inplace=True),"Sigmoid":nn.Sigmoid(),"Tanh":nn.Tanh()}
        activation_layer = dict_activation[activation]
        super(ConvLayer, self).__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
            ("activation",activation_layer)
        ]))

class RefineNetBlock(nn.Module):
    def __init__(self):
        super(RefineNetBlock, self).__init__()
        #一个7*7的卷积层+ 5*5的卷积层 + 3*3的卷积层 再加上一个跳跃连接
        self.direct = nn.Sequential(OrderedDict([
            ("conv_7x7", ConvLayer(2, 8, 7, activation="LeakyReLu")),
            ("conv_5x5", ConvLayer(8, 16, 5, activation="LeakyReLu")),
            ("conv_3x3",ConvLayer(16,2,3,activation="Tanh"))
        ]))
        self.identity = nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = self.identity(x)
        out = self.direct(x)
        out = self.relu(out + identity)
        
        return out

class CsiNetPlus(nn.Module):
    def __init__(self,reduction=4):
        super(CsiNetPlus, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("conv1_7x7", ConvLayer(2, 2, 7, activation='LeakyReLu')),
            ("conv2_7x7",ConvLayer(2,2,7,activation='LeakyReLu'))
        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)
        
        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        self.decoder_conv = ConvLayer(2, 2, 7, activation="Sigmoid")
        self.decoder_refine = nn.Sequential(OrderedDict([
            (f"RefineNet{i+1}",RefineNetBlock()) for i in range(5)
        ]))
        # self.refinenet = RefineNetBlock()
    def forward(self, x):
        n,c,h,w = x.detach().size()
        out = self.encoder_conv(x)
        out = self.encoder_fc(out.view(n, -1))
        
        out = self.decoder_fc(out).view(n, c, h, w)
        out = self.decoder_conv(out)
        out = self.decoder_refine(out)

        return out


        


if __name__ == "__main__":
    x = torch.ones(10, 2, 32, 32)
    net = CsiNetPlus()
    print(net)
    out = net(x)
    print(x.shape)
