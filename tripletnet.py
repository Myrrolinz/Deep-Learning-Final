import torch
import torch.nn as nn
bestrelu="relu"  #后续直接修改这里可使用最优激活函数
bestsigmoid="sigmoid" #后续直接在这里修改

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bn=True,
        bias=False,
        activation=bestrelu
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        #self.relu = nn.ReLU() if relu else None
        if activation=="relu":
            self.relu=nn.ReLU()
        elif activation=="sigmoid":
            self.relu=nn.Sigmoid()
        elif activation=="gelu":
            self.relu=nn.GELU()
        elif activation=="leaky-relu":
            self.relu=nn.LeakyReLU()
        else:
            self.relu=None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def __init__(self,pool_types):
        super(ChannelPool, self).__init__()
        self.pool_types=pool_types

    def forward(self, x):
        self.pool_list = []
        for pool in self.pool_types:
            if pool == "max":
                self.pool_list.append(torch.max(x,1)[0].unsqueeze(1))
            elif pool =="avg":
                self.pool_list.append(torch.mean(x, 1).unsqueeze(1))
            elif pool=="median":
                self.pool_list.append(torch.median(x,1)[0].unsqueeze(1))
            elif pool=="l1":
                self.pool_list.append(torch.norm(x,p=1,dim=1).unsqueeze(1))
            elif pool=="l2":
                self.pool_list.append(torch.norm(x,p=2,dim=1).unsqueeze(1))
        return torch.cat(
            self.pool_list, dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self,activation,pool_types,replace_sigmoid):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool(pool_types)
        in_planes=len(pool_types)
        self.spatial = BasicConv(
            in_planes, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, activation=activation
        )
        if replace_sigmoid=="sigmoid":
            self.sig=nn.Sigmoid()
        elif replace_sigmoid=="tanh":
            self.sig=nn.Tanh()
        elif replace_sigmoid=="softmax":
            self.sig=nn.Softmax(dim=0)
        else:
            self.sig=nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        #scale = torch.sigmoid_(x_out)
        scale=self.sig(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max","median"],
        no_spatial=False,
        activation=bestrelu,
        replace_sigmoid=bestsigmoid
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate(activation,pool_types,replace_sigmoid)
        self.ChannelGateW = SpatialGate(activation,pool_types,replace_sigmoid=replace_sigmoid)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(activation,pool_types,replace_sigmoid=replace_sigmoid)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out