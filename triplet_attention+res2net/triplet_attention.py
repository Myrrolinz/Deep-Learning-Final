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





