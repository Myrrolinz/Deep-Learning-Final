import torch
import torch.nn as nn
import CA
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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()
        #self.pool_types=pool_types

    def forward(self, x):
        # self.pool_list = []
        # for pool in self.pool_types:
        #     if pool == "max":
        #         self.pool_list.append(torch.max(x,1)[0].unsqueeze(1))
        #     elif pool =="avg":
        #         self.pool_list.append(torch.mean(x, 1).unsqueeze(1))
        #     elif pool=="median":
        #         self.pool_list.append(torch.median(x,1)[0].unsqueeze(1))
        #     elif pool=="l1":
        #         self.pool_list.append(torch.norm(x,p=1,dim=1).unsqueeze(1))
        #     elif pool=="l2":
        #         self.pool_list.append(torch.norm(x,p=2,dim=1).unsqueeze(1))
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        #in_planes=len(pool_types)
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )

        self.sig=nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        #scale = torch.sigmoid_(x_out)
        scale=self.sig(x_out)
        return x * scale


class TripletCAAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max","median"],
        no_spatial=False,
        combine=True,

    ):
        super(TripletCAAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.CA_C=CA.CoordAtt(inp=gate_channels,oup=gate_channels,reduction=reduction_ratio)
        # self.CA_W = CA.CoordAtt(inp=W, oup=W, reduction=reduction_ratio)
        # self.CA_H = CA.CoordAtt(inp=H, oup=H, reduction=reduction_ratio)
        self.no_spatial = no_spatial
        self.combine=combine
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous() #HxCxW
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()  # CxHxW
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        # if self.combine:
        #     x_out1=self.CA_H(x_perm1)
        # else:
        #     x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0, 2, 1, 3).contiguous() #CxHxW
        # x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        # if self.combine:
        #     x_out2=self.CA_W(x_perm2)
        # else:
        #     x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if self.combine:
            x_out=self.CA_C(x)
        else:
            x_out = self.SpatialGate(x)

        x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        # if not self.no_spatial:
        #
        #     x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out