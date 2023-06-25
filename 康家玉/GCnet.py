import torch.nn as nn
import torch
import torch.nn.functional as F

class GCBlock(nn.Module):
    def __init__(self, in_channels, scale=16):
        super(GCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels//scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        out = self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous()
        key = self.SoftMax(out)
        query = x.view(b, c, h*w)
        # [b, c, h*w] * [b, H*W, 1]
        out = torch.matmul(query, key)
        out = out.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(out)
        out = x + value
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, gc=True):
        super(Bottleneck, self).__init__()
        # First convolutional layer (1x1 kernel size)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # Batch normalization

        # Second convolutional layer (3x3 kernel size)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)  # Batch normalization

        # Third convolutional layer (1x1 kernel size)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)  # Batch normalization

        # GCBlock
        self.gc = GCBlock(out_channel*self.expansion) if gc else None

        self.relu = nn.ReLU(inplace=True)  # ReLU activation function
        self.downsample = downsample  # Downsample layer for skip connection

    def forward(self, x):
        identity = x  # Store the input for the skip connection
        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsample layer to input for skip connection

        out = self.conv1(x)  # First convolutional layer
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv2(out)  # Second convolutional layer
        out = self.bn2(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv3(out)  # Third convolutional layer
        out = self.bn3(out)  # Batch normalization

        if self.gc is not None:
            out = self.gc(out)

        out += identity  # Add the input (skip connection)
        out = self.relu(out)  # ReLU activation

        return out


class GCNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(GCNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(block, 64, blocks_num[0], gc=False)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, gc=True)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, gc=True)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2, gc=True)

        if self.include_top:
            # Average pooling and fully connected layer for classification
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, gc=True):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # Create downsample layer for skip connection
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # Create the first block with the downsample layer
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, gc=gc))
        self.in_channel = channel * block.expansion

        # Create the rest of the blocks
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, gc=gc))

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


def gcnet50(num_classes=1000, include_top=True):
    return GCNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

if __name__ == '__main__':
    # input = torch.randn(1, 3, 224, 224)  # B C H W
    # print(input.shape)
    # ResNet50 = res2net50(1000)
    # output = ResNet50.forward(input)
    # print(ResNet50)
    # print(output.shape)
    model = gcnet50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)