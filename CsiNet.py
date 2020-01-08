import torch
import torch.nn as nn
from collections import OrderedDict



# PyTorch版本的CsiNet


class ConvBN(nn.Sequential): # 包含卷积；批次归一化；激活函数
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2 # padding的设置是为了让输出的特征图的大小保持一致
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)), # 为什么bia设置为FALSE呢？
            ('bn', nn.BatchNorm2d(out_planes)), # 所以BatchNorm2d的输入参数是输出的特征图的通道数？
            ('relu',nn.LeakyReLU(negative_slope=0.3, inplace=True))
        ]))


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.direct_path = nn.Sequential(OrderedDict([
            ("conv_1", ConvBN(2, 8, kernel_size=3)),
            ("conv_2", ConvBN(8, 16, kernel_size=3)),
            ("conv_3", nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)),
            ("bn", nn.BatchNorm2d(2))
        ]))
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
    def forward(self, x):
        identity = self.identity(x)
        out = self.direct_path(x)
        out = self.relu(out + identity)
        
        return out

class CsiNet(nn.Module):
    def __init__(self,reduction=4):
        super(CsiNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        dim_out = total_size // reduction
        
        self.encoder_convbn = ConvBN(in_channel, 2, kernel_size=3)
        self.encoder_fc = nn.Linear(total_size, dim_out)

        self.decoder_fc = nn.Linear(dim_out, total_size)
        self.decoder_RefineNet1 = ResBlock()
        self.decoder_RefineNet2 = ResBlock()
        self.decoder_conv = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.decoder_bn = nn.BatchNorm2d(2)
        self.decoder_sigmoid = nn.Sigmoid()

    def forward(self, x):
        n,c, h, w = x.detach().size()
        x = self.encoder_convbn(x)
        x = x.view(n,-1) # 平坦化,reshape
        x = self.encoder_fc(x)
        # 此时x为编码后的输出，需要将x回传给发送端

        x = self.decoder_fc(x)
        x = x.view(n, c, h, w)
        x = self.decoder_RefineNet1(x)
        x = self.decoder_RefineNet2(x)
        x = self.decoder_conv(x)
        x = self.decoder_bn(x)
        x = self.decoder_sigmoid(x)

        return x


        
if __name__ == "__main__":
    x = torch.ones(10, 2, 32, 32)
    net = CsiNet()
    x = net(x)
    print(x.shape)
        
        
        
