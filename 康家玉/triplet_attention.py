import torch
import torch.nn as nn


# z-pool池化层
class ZPool(nn.Module):
    def forward(self, x):
        # Z-pool=[MaxPool,AvgPool]
        # torch.cat的使用:https://blog.csdn.net/xinjieyuan/article/details/105208352
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# 注意力模块
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        # 先Z-Pool池化
        self.z_pool = ZPool()
        # 再卷积 k*k的卷积核卷积->归一化处理
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1)
        )
        # 通过sigmoid来生成的注意力权值
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z_pool_out = self.z_pool(x)
        out = self.conv(z_pool_out)
        return x * self.sigmoid(out)


class TripletAttention(nn.Module):
    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial

        # 对三个分支进行初始化
        # 通道C和空间W维度交互
        self.cw = SpatialGate()
        # 通道C和空间H维度交互
        self.ch = SpatialGate()
        # 空间H和空间W进行注意力计算
        if self.spatial:
            self.hw = SpatialGate()

    def forward(self, x):
        # 0 2 1 3    b w c h
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ch(x_perm1)
        # 0 2 1 3    b c w h
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        # 0 3 2 1    b h w c
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.cw(x_perm2)
        # 0 3 2 1    b c w h
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.hw(x)
            return (1 / 3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1 / 2) * (x_out1 + x_out2)


if __name__ == '__main__':
    model = TripletAttention()
    print(model)

    random_input = torch.randn(1, 16, 256, 256)
    out = model(random_input)
    print(out.shape)
