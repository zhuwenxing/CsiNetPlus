
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

# PyTorch 现在加入了nn.Flatten()

class DeFlatten(nn.Module):
    def __init__(self,size):
        super(DeFlatten, self).__init__()
        self.size = size
    def forward(self, x):
        x = x.view(self.size)
        return x
class Decoder(nn.Module):
    pass



class SM_CsiNet(nn.Module):
    def __init__(self,reduction_base=4):
        super(SM_CsiNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("conv1_7x7", ConvLayer(2, 2, 7, activation='LeakyReLu')),
            ("conv2_7x7",ConvLayer(2,2,7,activation='LeakyReLu'))
        ]))
        self.fc_cr4 = nn.Linear(total_size, total_size // 4)
        self.fc_cr8 = nn.Linear(total_size//4,total_size//8)
        self.fc_cr16 = nn.Linear(total_size // 8, total_size // 16)
        self.fc_cr32 = nn.Linear(total_size // 16, total_size//32)
        
    def forward(self, x,idx):
        x = self.encoder(x)
        # 如何控制流程？
        # 在训练过程中，原文是end 2 end的形式
        # 但是如果是交替的形式呢？不同的维度需要冻结
        
        dim_cr4 = self.fc_cr4(x)
        dim_cr8 = self.fc_cr8(dim_cr4)
        dim_cr16 = self.fc_cr16(dim_cr8)
        dim_cr32 = self.fc_cr32(dim_cr16)

        out_cr4 = self.decoder_cr4(dim_cr4)
        out_cr8 = self.decoder_cr8(dim_cr8)
        out_cr16 = self.decoder_cr16(dim_cr16)
        out_cr32 = self.decoder_cr32(dim_cr32)
        
        out_list = [out_cr4, out_cr8, out_cr16, out_cr32]
        # idx 为索引列表 例如idx = [0,1,2,3,4],表示返回所有值
        # idx = [0,1]只返回部分结果
        out = out_list[idx]

        return out

class Encoder(nn.Module):
    pass
class Decoder(nn.Module):
    pass
class PM_CsiNet(nn.Module):
    def __init__(self):
        super(PM_CsiNet, self).__init__()
        self.encoder = Encoder()
        self.fc_list = nn.ModuleList(nn.Linear(512, 64) for i in range(8))
        self.decoder_list = nn.ModuleList(Decoder(reduction=i) for i in [32,16,8,4])
    def forward(self, x):
        dim_512 = self.encoder(x)
        dim_64_list = [self.fc_list[i](x) for i in range(8)]
        cr_out_list = [torch.cat(dim_64_list[:(i**2)],dim=1) for i in range(4)]
        out_list = [self.decoder_list[i](cr_out_list[i]) for i in range(4)]

        return out_list





if __name__ == "__main__":
    x = torch.ones(10, 2, 32, 32)
    net = CsiNetPlus()
    print(net)
    out = net(x)
    print(x.shape)
