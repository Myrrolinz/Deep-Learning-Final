import torch
import torch.nn as nn

# # SE模块要添加吗
# class SEModule(nn.Module):


# 残差层
class Bottleneck(nn.Module):
    # 输出通道数为输入通道数的4倍
    expansion = 4
    # 后续需要为其添加新参数
    def __init__(self, inplanes, planes, stride=1, downsample=None, scales=4, groups=1):
        super(Bottleneck, self).__init__()

        outplanes = scales * groups
        # 第一个卷积层：卷积核尺寸： 1×1 ，填充值为 0， 步长为 1
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        # 第二个卷积结构：卷积核尺寸： 3×3 ，填充值为 1， 步长为 1
        self.conv2 = nn.ModuleList([nn.Conv2d(outplanes // scales, outplanes // scales,
                                              kernel_size=3, stride=stride, padding=1, groups=groups) for _ in
                                    range(scales - 1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(outplanes // scales) for _ in range(scales - 1)])

        # 第三个卷积层：卷积核尺寸： 1×1 ，填充值为 0， 步长为 1
        self.conv3 = nn.Conv2d(outplanes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.scales = scales
        self.groups = groups

    def forward(self, x):
        identity = x # 将原始输入暂存为shortcut的输出
        # 对下采样进行处理
        # 如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 1*1卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3*3卷积结构
        x_scale = torch.chunk(out, self.scales, 1)  # 将x分割成scales块
        y_scale = []
        for s in range(self.scales):
            if s == 0:
                y_scale.append(x_scale[s])
            elif s == 1:
                conv_out = self.conv2[s - 1](x_scale[s])
                bn_out = self.bn2[s - 1](conv_out)
                y_scale.append(self.relu(bn_out))
            else:
                conv_out = self.conv2[s - 1](x_scale[s] + y_scale[-1])
                bn_out = self.bn2[s - 1](conv_out)
                y_scale.append(self.relu(bn_out))
        out = torch.cat(y_scale, 1)


        # 1*1卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        # 是否加入SE模块


        # 残差连接 out=F(X)+X
        out += identity
        out = self.relu(out)




class Res2Net(nn.Module):
    def __init__(self):



    def forward(self):