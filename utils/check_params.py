# Add parent directory to Python path to import src module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import packages
from torchsummary import summary

# Import model
from src.model import get_resnet50 # absolute import

## Test Model ##
if __name__ == "__main__":

    print("ResNet-50 model:")
    model = get_resnet50()
    summary(model, (3, 224, 224)) # Printing model summary with parameters
