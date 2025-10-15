### ResNet-50 (ImageNet) — Beginner-Friendly Guide

This document explains the `ResNet-50` architecture in this repository in simple terms, with pointers to the exact lines in `model/model.py` so you can follow along.

Think of the network as a factory assembly line for images:
- We take an image in, shrink it gradually while increasing the number of feature channels, and at the end we make a decision (which class it is).
- "Residual" means the network learns to make small improvements on top of what it already knows, like a helper adding fixes to a draft rather than rewriting it from scratch.

---

## File you are reading along with
- Source: `model/model.py`

---

## 1) Helper: 1x1 Convolution (channel changer)
A 1x1 convolution changes the number of channels without changing width/height. It’s like a smart color-mixer for feature maps. We use it often to match shapes before we add tensors.

```13:16:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
	# 1x1 convolution used to project channel dimensions and optionally downsample spatially
	return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
```

- "project" means convert from one number of channels to another.
- With `stride>1`, it can also shrink width/height (downsample) while changing channels.

---

## 2) The Bottleneck Block (the core building unit)
A Bottleneck block is like a mini-assembly line with three steps: 1x1, 3x3, 1x1 convs. It first shrinks channels (cheap), processes features (3x3), then expands channels back (1x1). It also has a "shortcut" path to add the input back — this is the "residual" part.

```18:56:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
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

		# Final 1x1 convolution expands to planes * self.expansion channels
		self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = norm_layer(planes * self.expansion)

		# Optional downsampling/projection for the residual path to match shape
		self.downsample = downsample

		# Shared ReLU activation (inplace for memory efficiency)
		self.relu = nn.ReLU(inplace=True)
```

Key ideas:
- **Reduce → Process → Expand:** 1x1 reduces channels (cheap), 3x3 learns spatial patterns, 1x1 expands back.
- **Residual addition:** If shapes differ, we "project" the shortcut with a 1x1 conv so addition is possible.

Forward pass adds the shortcut:
```58:83:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Preserve input as identity for the residual connection
		identity = x
		# 1x1 -> BN -> ReLU
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		# 3x3 -> BN -> ReLU
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		# 1x1 -> BN
		out = self.conv3(out)
		out = self.bn3(out)
		# Match shapes if needed
		if self.downsample is not None:
			identity = self.downsample(x)
		# Residual add + ReLU
		out += identity
		out = self.relu(out)
		return out
```

---

## 3) The ResNet class (putting blocks together)
This is the full network: a "stem" to start, then 4 stages of blocks, then pooling and a final linear layer.

### 3.1 Constructor and Stem
The stem quickly shrinks the image and increases channels — like a big entrance door.
```86:117:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
class ResNet(nn.Module):
	# ImageNet-style ResNet using Bottleneck blocks
	def __init__(
		self,
		block: type[Bottleneck],
		layers: list[int],
		num_classes: int = 1000,
		width_per_group: int = 64,
		norm_layer: type[nn.Module] | None = None,
	) -> None:
		# choose BN if none was passed
		...
		self.inplanes = 64
		self.base_width = width_per_group
		# Stem: 7x7 conv (stride=2) + BN + ReLU + 3x3 maxpool (stride=2)
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```
- 7x7 conv with stride 2 reduces size a lot initially (ImageNet standard).
- MaxPool reduces size again; we go from large images to manageable feature maps.

### 3.2 Four Residual Stages
Each stage stacks several Bottleneck blocks. The first block of a stage may downsample (stride=2).
```113:121:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
		self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
```
- ResNet-50 uses `layers = [3, 4, 6, 3]` blocks per stage.
- Channels grow 64 → 128 → 256 → 512 (then expanded by 4x inside bottlenecks).

### 3.3 Global Average Pooling and Classifier
```119:121:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)
```
- AdaptiveAvgPool2d makes every feature map become 1×1, no matter the input size.
- Final `Linear` maps the features to class scores.

---

## 4) Building a Stage: `_make_layer`
This function builds one stage (e.g., layer2) by stacking Bottleneck blocks.
```131:178:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
	def _make_layer(
		self,
		block: type[Bottleneck],
		planes: int,
		blocks: int,
		stride: int = 1,
	) -> nn.Sequential:
		# Choose norm layer and decide if we need a shortcut projection
		norm_layer = self._norm_layer
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		# First block (maybe downsampling)
		layers_list: list[nn.Module] = []
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
		# Update inplanes to expanded channels for the rest of the blocks
		self.inplanes = planes * block.expansion
		# Remaining blocks
		for _ in range(1, blocks):
			layers_list.append(
				block(
					inplanes=self.inplanes,
					planes=planes,
					base_width=self.base_width,
					norm_layer=norm_layer,
				)
			)
		return nn.Sequential(*layers_list)
```
- If shapes don’t match, we build a `downsample` (using `conv1x1`) to fix the shortcut.
- The first block may shrink spatial size with `stride=2`.

---

## 5) Forward Pass (how data flows)
This is the full journey of the image through the network.
```180:199:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
	def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		# Four stages of residual blocks
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		# Global pooling and classification
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x
```
- Stem shrinks quickly, residual stages learn increasingly complex patterns, pool to 1×1, then classify.

---

## 6) Factory Function (easy constructor)
```202:204:/Users/namratan/TSAI/ERA V4/Session 9 Capstone/resnet50-imagenet-1k/model/model.py
def ResNet50(num_classes: int = 1000) -> ResNet:
	# Factory function producing a standard ResNet-50 with 1000-way classifier by default
	return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
```
- Use this to get a ready-to-train ResNet-50.

Example:
```python
from model.model import ResNet50
model = ResNet50(num_classes=1000)
```

---

## Visual intuition (very simplified)
```
Image → [7x7 conv, stride 2] → MaxPool →
  Stage1: [Bottleneck ×3] →
  Stage2: [Bottleneck ×4] (first block stride 2) →
  Stage3: [Bottleneck ×6] (first block stride 2) →
  Stage4: [Bottleneck ×3] (first block stride 2) →
  GlobalAvgPool → FC → class scores
```

---

## Why residual connections?
Without shortcuts, very deep nets can forget earlier information and are hard to train. Residuals give the network an "escape route" to pass information forward, making training stable and letting us build deeper models like ResNet-50.
