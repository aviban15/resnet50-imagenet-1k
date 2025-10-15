## Initial Setup ##

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

# Import modules
from ..src.dataloader import ImageNetDataLoader
from ..src.model import ResNetModel

## Data Transformations ##

# ImageNet statistics
mean = [0.485, 0.485, 0.406]
std = [0.229, 0.224, 0.225]

# Train data transformations
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(
       224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
    ),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Normalize(mean=mean, std=std)
])

# Validation data transformations
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=256, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=mean, std=std)
])

## Train & Validation Datasets ##

# Load ImageNet data
train_dataset = datasets.ImageFolder(
    root="imagenet_sample_4x12",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root="imagenet_sample_4x4",
    transform=val_transforms
)

## Train/Validation Dataloaders ##

# Set seed
SEED = 1

# For reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Train dataloader
train_sampler = torch.utils.data.RandomSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
)

# Val dataloader
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=False,
    pin_memory=True
)

## Base ResNet Model ##

# ResNet base model
from torchvision.models import resnet50
model = resnet50()

# Model summary
device = (
   "cuda" if torch.cuda.is_available()
    else "cpu"
)
print('Device =', device)
model = model.to(device)
summary(model, input_size=(3, 224, 224))

## Train and Test Functions ##

# For visualization
train_losses = []
test_losses = []
train_acc = []
test_acc = []

# Train function
def train(model, device, train_loader, optimizer, epoch):
# def train(model, device, train_loader, optimizer, scheduler, epoch):
  model.train()  # Set to train mode
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    # Get samples
    data, target = data.to(device), target.to(device)

    # Initialize gradients to zero
    optimizer.zero_grad()

    # Predict using model
    y_pred = model(data)

    # Calculate loss
    # loss = F.nll_loss(y_pred, target)
    loss = F.cross_entropy(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    # scheduler.step()

    # Calculate accuracy
    pred = y_pred.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    # Update tqdm progress bar
    pbar.set_description(desc= f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:.2f}')
    train_acc.append(100*correct/processed)

  return train_losses[-1], train_acc[-1]

# Test workflow
def test(model, device, test_loader):
    model.eval()  # Set to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # Average loss
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
        ))
    test_acc.append(100. * correct / len(test_loader.dataset))

    return test_losses[-1], test_acc[-1]

## Training and Testing the Model ##

# Training parameters
EPOCHS = 4
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

# Training loop
acc_best = 0
for epoch in range(EPOCHS):
    print(f'EPOCH: {epoch} | LR: {scheduler.get_last_lr()[0]:.6f}')
    loss_train, acc_train = train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    loss_test, acc_test = test(model, device, val_loader)

    # Save training state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, 'checkpoint.pth')

    # Save best weights
    if acc_test > acc_best:
      acc_best = acc_test
      torch.save(model.state_dict(), "best_model_weights.pth")