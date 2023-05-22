import torch
import torch.nn as nn


# z-pool池化层
class ZPool(nn.Module):
    def forward(self, x):
        # Z-pool=[MaxPool,AvgPool]
        # torch.cat的使用:https://blog.csdn.net/xinjieyuan/article/details/105208352
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


