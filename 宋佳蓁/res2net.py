import torch
import torch.nn as nn

#TA模块
import torch
import torch.nn as nn


# z-pool池化层
class ZPool(nn.Module):
    def forward(self, x):
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


#SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


# 残差层
class Bottleneck(nn.Module):
    # 输出通道数为输入通道数的4倍
    expansion = 4

    # 后续需要为其添加新参数
    def __init__(self, inplanes, planes, stride=1, downsample=None, scales=4, groups=1, is_first_block=0, activation = 'relu'):
        super(Bottleneck, self).__init__()

        self.downsample = downsample
        self.scales = scales
        self.groups = groups
        self.stride = stride
        self.is_first_block = is_first_block
        
        #激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'celu':
            self.activation = nn.CELU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'leaky-relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(f"Activation not implemented!")
        
        outplanes = groups * planes
        # 第一个卷积层：卷积核尺寸： 1×1 ，填充值为 0， 步长为 1
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        # 第二个卷积结构：卷积核尺寸： 3×3 ，填充值为 1， 步长为 1 cnv2d中的参数????
        self.conv2 = nn.ModuleList([nn.Conv2d(outplanes // scales, outplanes // scales,
                                              kernel_size=3, stride=stride, padding=1, groups=groups, bias=False) for _ in
                                    range(scales - 1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(outplanes // scales) for _ in range(scales - 1)])

        # 第三个卷积层：卷积核尺寸： 1×1 ，填充值为 0， 步长为 1
        self.conv3 = nn.Conv2d(outplanes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        #self.relu = nn.ReLU(inplace=True)
        #self.celu = nn.CELU(inplace=True)
        # 处理第一个块
        if is_first_block == 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        # SE模块
        self.se = SEModule(planes * self.expansion)
        self.semodule = SEModule(outplanes // scales)
        # TA模块
        self.ta = TripletAttention()

    def forward(self, x):
        identity = x # 将原始输入暂存为shortcut的输出
        # 对下采样进行处理
        # 如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 1*1卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)
        out = self.activation(out)
        # 3*3卷积结构
        # x_scale = torch.chunk(out, self.scales, 1)  # 将x分割成scales块
        # y_scale = []
        # for s in range(self.scales):
        #     if s == 0:
        #         y_scale.append(x_scale[s])
        #     elif s == 1:
        #         conv_out = self.conv2[s - 1](x_scale[s])
        #         bn_out = self.bn2[s - 1](conv_out)
        #         y_scale.append(self.relu(bn_out))
        #         # print(x_scale[s].shape, conv_out.shape, bn_out.shape, self.relu(bn_out).shape)
        #     else :
        #         # print(x_scale[s].shape, y_scale[-1].shape)
        #         conv_out = self.conv2[s - 1](x_scale[s] + y_scale[-1])
        #         bn_out = self.bn2[s - 1](conv_out)
        #         y_scale.append(self.relu(bn_out))

        # out = torch.cat(y_scale, 1)

        x_scales = torch.chunk(out, self.scales, 1)
        for i in range(self.scales-1):
            if i == 0 or self.is_first_block == 1:
                y_scale = x_scales[i]
            else:
                y_scale = y_scale + x_scales[i]
            y_scale = self.conv2[i](y_scale)
            # y_scale = self.celu(self.bn2[i](y_scale))
            # y_scale = self.activation(self.bn2[i](y_scale))
            # y_scale = self.semodule(y_scale)
            y_scale = self.ta(y_scale)
            if i == 0:
                out = y_scale
            else:
                out = torch.cat((out, y_scale), 1)
        if self.scales != 1 and self.is_first_block == 0:
            out = torch.cat((out, x_scales[self.scales-1]), 1)
        elif self.scales != 1 and self.is_first_block == 1:
            out = torch.cat((out, self.pool(x_scales[self.scales-1])), 1)

        # 1*1卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        # 是否加入SE模块--该结构利用系数scale来使网络自适应的减弱或增强该通道的特征，与注意力机制有异曲同工之妙。

        out = self.se(out)
        # out = self.ta(out)
        # 添加triplet_attention

        # 残差连接 out=F(X)+X
        out += identity
        # out = self.relu(out)
        self.activation(out)

        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, num_classes=1000, scales=4, groups=1):
        super(Res2Net, self).__init__()
        # 通道数初始化
        self.inplanes = 64

        # 起始：7*7的卷积层，3*3的最大池化层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True)
        #     )



        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差结构
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, scales=scales, groups=groups)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, scales=scales, groups=groups)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, scales=scales, groups=groups)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, scales=scales, groups=groups)

        # 平均池化+全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 为啥还初始化了bn????

    def _make_layer(self, block, planes, layer, stride=1, scales=4, groups=1, se=True):
        # 积步长不为1或深度扩张有变化，导致F(X)与X的shape不同的残差块，就要对X定义下采样函数，使之shape相同
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.AvgPool2d(kernel_size=2,stride=2)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个残差块需要下采样  def __init__(self, inplanes, planes, stride=1, downsample=None, scales=4, groups=1):
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            scales=scales, groups=groups, is_first_block=1))
        self.inplanes = planes * block.expansion

        # 通过循环堆叠其余残差块
        for i in range(1, layer):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # probas = nn.functional.softmax(x, dim=1) ??????

        return x


# def __init__(self, block, layers, num_classes=1000, scales=4, groups=1):
def res2net50(num_classes=1000, scales=4, groups=1):
    return Res2Net(Bottleneck, [3, 4, 6, 3], num_classes, scales, groups)


def res2net101(num_classes=1000, scales=4, groups=1):
    return Res2Net(Bottleneck, [3, 4, 23, 3], num_classes, scales, groups)


if __name__ == '__main__':
    # input = torch.randn(1, 3, 224, 224)  # B C H W
    # print(input.shape)
    # ResNet50 = res2net50(1000)
    # output = ResNet50.forward(input)
    # print(ResNet50)
    # print(output.shape)
    model = res2net50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)






