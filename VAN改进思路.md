# VAN改进思路

LKA部分的源代码：

```python
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
```

作者在论文中提到了以下改进方向：

1. 继续改进它的结构。在本文中，只展示了一个直观的结构，还存在很多潜在的改进点，例如：应用大核、引入多尺度结构和使用多分支结构。
2. 大规模的自监督学习和迁移学习。VAN 自然地结合了CNN和ViT的优点。一方面VAN利用了图像的2D结构。另一方面 VAN可以基于输入图片动态的调整输出，它很适合自监督学习和迁移学习。结合了这两点，作者认为VAN可以在这两个领域有更好的性能。
3. 更多的应用场景。由于资源有限，作者只展示了它在视觉任务中的优秀性能。作者期待VAN在各个领域都展示优秀性能并变成一个通用的模型。

我主要聚焦第一个改进建议，进行相关工作。

### 应用大核

针对LKA模型的`conv0`层进行相应的修改

```python
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1))
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
```

### 多尺度分支

引入了`MultiScaleBranch`类作为多尺度分支结构，用于处理不同尺度的特征。`MultiScaleBranch`类包含三个不同尺寸的卷积核（3x3、5x5、7x7），它们分别对输入进行卷积操作，然后将结果进行拼接。

在`LKA`类的`forward`方法中，使用了`MultiScaleBranch`作为`conv_spatial`的替代，这样就引入了多尺度结构。在模型最后一层的卷积操作中，将多尺度分支的输出与原始输入进行拼接，以增强模型的表示能力。

```python
class MultiScaleBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)

    def forward(self, x):
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_7x7 = self.conv7x7(x)
        return torch.cat([out_3x3, out_5x5, out_7x7], dim=1)


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = MultiScaleBranch(dim)
        self.conv1 = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = F.relu(attn)
        attn = self.conv_spatial(attn)
        attn = F.relu(attn)
        attn = self.conv1(attn)

        return u * attn
```

### 其他改进思路

1. 增加批归一化层（Batch Normalization）：在卷积层之后添加批归一化层可以帮助加速模型的收敛并提高模型的泛化能力。可以在每个卷积层后添加nn.BatchNorm2d()层，并在forward()函数中应用。

```python
import torch.nn as nn

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.bn0 = nn.BatchNorm2d(dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.bn_spatial = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.bn1 = nn.BatchNorm2d(dim)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.bn0(attn)
        attn = self.conv_spatial(attn)
        attn = self.bn_spatial(attn)
        attn = self.conv1(attn)
        attn = self.bn1(attn)

        return u * attn

```

2. 添加非线性激活函数：在卷积层之后添加非线性激活函数可以引入非线性特征映射，提高模型的表示能力。可以在每个卷积层之后使用激活函数，例如ReLU。

```python
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = F.relu(attn)
        attn = self.conv_spatial(attn)
        attn = F.relu(attn)
        attn = self.conv1(attn)

        return u * attn

```

