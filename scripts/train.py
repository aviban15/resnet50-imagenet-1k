## Initial Setup ##

# Import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.amp import GradScaler, autocast

# Other packages
import os
import logging
from tqdm import tqdm
from datetime import datetime

# Import modules and configurations
from src.dataloader import get_dataloaders
from src.model import get_resnet50
from scripts.config import *

## Logging Setup ##

# Create logs and checkpoints directory
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Logging configuration
start_time = datetime.now().strftime(r'%Y%m%d_%H%M%S_%Z')
logging.basicConfig(
    filename=f'logs/training_{start_time}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Initialize logging
logger = logging.getLogger(__name__)
logger.info("=== ResNet50 ImageNet-1K Training Started ===")

## Dataloaders setup ##
logger.info("Loading ImageNet dataloaders...")
train_loader, val_loader = get_dataloaders(
   TRAIN_DIR, VAL_DIR,
   batch_size=BATCH_SIZE,
   num_workers=NUM_WORKERS
)
logger.info(f"Train loader: {len(train_loader)} batches")
logger.info(f"Validation loader: {len(val_loader)} batches")

## Model setup ##
logger.info("Loading ResNet model...")
model = get_resnet50(num_classes=NUM_CLASSES)
logger.info("ResNet model loaded successfully")

## Device setup ##
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device = {device}")

## Train and Test Functions ##

# For visualization
train_losses = []
test_losses = []
train_acc = []
test_acc = []

# Train function
def train(model, device, train_loader, optimizer, scaler, epoch):
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

    # Predict using model with mixed precision
    with autocast():
      y_pred = model(data)
      # Calculate loss
      # loss = F.nll_loss(y_pred, target)
      loss = F.cross_entropy(y_pred, target)
    train_losses.append(loss)

    # Backpropagation with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # scheduler.step()

    # Calculate accuracy
    pred = y_pred.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    # Update tqdm progress bar
    pbar.set_description(desc= f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:.2f}')
    train_acc.append(100*correct/processed)

  return train_losses[-1], train_acc[-1]

# Test function
def test(model, device, test_loader):
    model.eval()  # Set to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast():
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
logger.info(f"Training configuration: {NUM_EPOCHS} epochs")
model = model.to(device)  # Move model to device
scaler = GradScaler()  # Initialize GradScaler for mixed precision training
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-4)

logger.info(f"Optimizer: {optimizer}")
logger.info(f"Scheduler: {scheduler}")
logger.info("Mixed precision training enabled")

# Load training state if resuming from checkpoint
checkpoint_path = 'checkpoints/checkpoint.pth'
if RESUME_TRAINING:
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Resumed from epoch {start_epoch}")
else:
    logger.info("Starting fresh training...")
    start_epoch = 0

# Training loop
logger.info("Starting training loop...")
acc_best = 0
for epoch in range(start_epoch, NUM_EPOCHS):
    logger.info(f"=" * 60)
    logger.info(f"EPOCH: {epoch+1}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f}")
    print(f'EPOCH: {epoch} | LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # Training phase
    loss_train, acc_train = train(model, device, train_loader, optimizer, scaler, epoch)
    scheduler.step()
    
    # Validation phase
    loss_test, acc_test = test(model, device, val_loader)

    # Log epoch summary
    logger.info(f"Epoch {epoch+1} Summary:")
    logger.info(f"  Train Loss: {loss_train:.4f}, Train Accuracy: {acc_train:.2f}%")
    logger.info(f"  Validation Loss: {loss_test:.4f}, Validation Accuracy: {acc_test:.2f}%")

    # Save training state as checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, checkpoint_path)
    logger.info("Checkpoint saved")

    # Save best weights
    if acc_test > acc_best:
      acc_best = acc_test
      torch.save(model.state_dict(), "checkpoints/best_model_weights.pth")
      logger.info(f"New best accuracy: {acc_best:.2f}% - Best model weights saved")
    
    # Save log file periodically
    logger.info(f"Epoch {epoch+1} completed and logged.")

logger.info("=" * 60)
logger.info("Training completed!")
logger.info(f"Best validation accuracy achieved: {acc_best:.2f}%")