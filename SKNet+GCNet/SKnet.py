import torch.nn as nn
import torch
from functools import reduce


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        """
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        """

        super(SKConv, self).__init__()
        # 计算从向量C降维到 向量Z 的长度d
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels

        # 根据分支数量 添加 不同核的卷积操作
        self.conv = nn.ModuleList()
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)  # 升维

        # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # ****the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))  # [batch_size,out_channels,H,W]

        # ****the part of fusion
        U = reduce(lambda x, y: x + y, output)  # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        s = self.global_pool(U)  # [batch_size,channel,1,1]
        z = self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)  # 调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]

        # ****the part of selection
        # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        # [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b = list(a_b.chunk(self.M, dim=1))
        # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))
        # 权重与对应  不同卷积核输出的U 逐元素相乘
        # [batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V = list(map(lambda x, y: x * y, output, a_b))
        # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        V = reduce(lambda x, y: x + y, V)
        return V  # [batch_size,out_channels,H,W]


class SKBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(SKBlock, self).__init__()
        # First convolutional layer (1x1 kernel size)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # Second convolutional layer (3x3 kernel size)-----此处改为SKblock

        self.conv2 = SKConv(out_channel, out_channel, stride)

        # Third convolutional layer (1x1 kernel size)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample  # Downsample layer for skip connection

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsample layer to input for skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # Add the input (skip connection)
        out = self.relu(out)  # ReLU activation

        return out


class SKNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(SKNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Max pooling
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # Average pooling and fully connected layer for classification
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # Create downsample layer for skip connection
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # Create the first block with the downsample layer
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        # Create the rest of the blocks
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # Initial convolutional layer
        x = self.bn1(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation
        x = self.maxpool(x)  # Max pooling

        # Pass through the residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)  # Average pooling
            x = torch.flatten(x, 1)  # Flatten the tensor
            x = self.fc(x)  # Fully connected layer for classification

        return x


def sknet50(num_classes=1000, include_top=True):
    return SKNet(SKBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


if __name__ == '__main__':
    # input = torch.randn(1, 3, 224, 224)  # B C H W
    # print(input.shape)
    # ResNet50 = res2net50(1000)
    # output = ResNet50.forward(input)
    # print(ResNet50)
    # print(output.shape)
    model = sknet50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
