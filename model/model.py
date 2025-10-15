import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet BasicBlock used in ResNet18/34.

    RF tracking is annotated in the ResNet forward path where shapes are known.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck used in ResNet50/101/152 (kept for completeness)."""
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: type[nn.Module], num_blocks: list[int], num_classes: int = 100) -> None:
        super().__init__()
        self.in_planes = 64

        # Input: (B, 3, 32, 32)
        # conv1: 3x3, s=1, p=1 -> (B, 64, 32, 32)
        # RF: 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # layer1 (stride=1): keeps spatial size 32x32; RF grows with two 3x3 convs per block
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        # layer2 (stride=2): downsample to 16x16
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        # layer3 (stride=2): downsample to 8x8
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        # layer4 (stride=2): downsample to 4x4
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Adaptive pool to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 512*expansion, 1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block: type[nn.Module], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 3, 32, 32), RF=1 at pixel level
        out = F.relu(self.bn1(self.conv1(x)))  # (B, 64, 32, 32), RF=3
        out = self.layer1(out)                 # (B, 64, 32, 32), RF increases within blocks
        out = self.layer2(out)                 # (B, 128, 16, 16)
        out = self.layer3(out)                 # (B, 256, 8, 8)
        out = self.layer4(out)                 # (B, 512, 4, 4)
        out = self.avgpool(out)                # (B, 512, 1, 1)
        out = torch.flatten(out, 1)            # (B, 512)
        out = self.fc(out)                     # (B, num_classes)
        return out


def ResNet18(num_classes: int = 100) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes: int = 100) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes: int = 100) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


