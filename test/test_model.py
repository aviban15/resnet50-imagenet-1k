## Initial Setup ##

# Add parent directory to Python path to import src module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Other packages
from tqdm import tqdm

# Import model
from src.model import get_resnet50

## Test Model ##

if __name__ == "__main__":
	model = get_resnet50()
	print(model)