import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################################################
# ResNet-50 (ImageNet) implementation 
# - 7x7 conv stem (stride=2), 3x3 maxpool (stride=2)
# - 4 stages of Bottleneck blocks with counts [3, 4, 6, 3]
# - Global average pooling and fully-connected classifier
###################################################################################################


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
	# 1x1 convolution used to project channel dimensions and optionally downsample spatially
	return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
	# Bottleneck block expands channel dimensions by a factor of 4
	expansion = 4

	def __init__(
		self,
		inplanes: int,                 # input channels to the block
		planes: int,                   # internal base width before expansion
		stride: int = 1,               # stride applied at the 3x3 conv for downsampling
		downsample: nn.Module | None = None,  # optional projection for the shortcut path
		base_width: int = 64,          # width per group (when groups=1, classic ResNet)
		norm_layer: type[nn.Module] | None = None,  # normalization layer constructor
	) -> None:
		super().__init__()

		# Default normalization is BatchNorm2d
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d

		# Compute the intermediate width (supports width scaling though groups=1 here)
		width = int(planes * (base_width / 64.0))

		# First 1x1 convolution reduces channel dimension
		self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
		self.bn1 = norm_layer(width)

		# 3x3 convolution performs spatial processing; stride may downsample
		self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = norm_layer(width)

		# Final 1x1 convolution expands to planes * expansion channels
		self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = norm_layer(planes * self.expansion)

		# Optional downsampling/projection for the residual path to match shape
		self.downsample = downsample

		# Shared ReLU activation (inplace for memory efficiency)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Preserve input as identity for the residual connection
		identity = x

		# Main path: 1x1 conv -> BN -> ReLU
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		# 3x3 conv -> BN -> ReLU (may downsample spatially)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		# 1x1 conv -> BN (no activation yet)
		out = self.conv3(out)
		out = self.bn3(out)

		# If the spatial/channel shape changed, project the identity
		if self.downsample is not None:
			identity = self.downsample(x)

		# Residual addition followed by activation
		out += identity
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	# ImageNet-style ResNet using Bottleneck blocks
	def __init__(
		self,
		block: type[Bottleneck],        # block class to use (Bottleneck for ResNet-50)
		layers: list[int],              # number of blocks in each of the 4 stages
		num_classes: int = 1000,        # number of output classes
		width_per_group: int = 64,      # width scaling for the bottleneck
		norm_layer: type[nn.Module] | None = None,  # normalization layer constructor
	) -> None:
		super().__init__()

		# Default normalization is BatchNorm2d
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		# Initial channel width before entering residual layers
		self.inplanes = 64
		self.base_width = width_per_group

		# Stem: 7x7 conv (stride=2) to quickly reduce spatial resolution
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# Residual stages: the first block in each stage may downsample (stride=2)
		self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		# Global average pooling to 1x1 followed by a fully-connected classifier
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		# Kaiming initialization for convolutions; constant initialization for norms
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(
		self,
		block: type[Bottleneck],   # block type (Bottleneck)
		planes: int,               # base planes for this stage (before expansion)
		blocks: int,               # number of blocks in the stage
		stride: int = 1,           # stride applied in the first block of the stage
	) -> nn.Sequential:
		# Build a residual stage consisting of `blocks` Bottleneck modules
		norm_layer = self._norm_layer
		downsample = None

		# Determine if a projection is needed for the shortcut path
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers_list: list[nn.Module] = []

		# First block in the stage; may downsample and uses downsample projection if needed
		layers_list.append(
			block(
				inplanes=self.inplanes,
				planes=planes,
				stride=stride,
				downsample=downsample,
				base_width=self.base_width,
				norm_layer=norm_layer,
			)
		)

		# Update inplanes to the expanded channel dimension for subsequent blocks
		self.inplanes = planes * block.expansion

		# Remaining blocks keep stride=1 and share the same channel dimensions
		for _ in range(1, blocks):
			layers_list.append(
				block(
					inplanes=self.inplanes,
					planes=planes,
					base_width=self.base_width,
					norm_layer=norm_layer,
				)
			)

		# Package blocks into a sequential container
		return nn.Sequential(*layers_list)

	def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
		# Standard ResNet forward pass through stem, stages, and head
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
		return x

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Thin wrapper for potential TorchScript compatibility
		return self._forward_impl(x)


def ResNet50(num_classes: int = 1000) -> ResNet:
	# Factory function producing a standard ResNet-50 with 1000-way classifier by default
	return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
