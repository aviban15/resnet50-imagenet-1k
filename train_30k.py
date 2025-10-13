import boto3
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Initialize S3 client
s3_client = boto3.client('s3')

# S3 bucket details
bucket_name = 'imagenet-30sample'
prefix = 'imagenet_sampled30/'

# Custom S3ImageFolder class with stratified split
class S3ImageFolder(Dataset):
    """Custom PyTorch Dataset that reads ImageNet data directly from S3"""
    
    def __init__(self, bucket_name, prefix, s3_client, transform=None, limit_classes=10, split='train', train_samples=24):
        self.bucket = bucket_name
        self.prefix = prefix
        self.s3 = s3_client
        self.transform = transform
        self.split = split
        self.train_samples = train_samples
        
        print(f"Loading class folders from S3 for {split} split...")
        
        paginator = self.s3.get_paginator('list_objects_v2')
        
        # Get class folders
        class_folders = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for folder in page['CommonPrefixes']:
                    folder_name = folder['Prefix'].replace(self.prefix, '').rstrip('/')
                    if folder_name:
                        class_folders.add(folder_name)
        
        self.classes = sorted(list(class_folders))[:limit_classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} classes")
        
        # Build list of images with stratified split
        self.samples = []
        print(f"Loading image paths from S3 for {split} split...")
        
        for class_name in self.classes:
            class_prefix = f"{self.prefix}{class_name}/"
            class_idx = self.class_to_idx[class_name]
            
            # Collect all images for this class
            class_images = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=class_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith(('.jpg', '.jpeg', '.JPEG', '.png', '.PNG')):
                            class_images.append(key)
            
            # Sort for consistent splitting
            class_images.sort()
            
            # Stratified split: first N for train, rest for test
            if self.split == 'train':
                selected_images = class_images[:self.train_samples]
            else:  # test
                selected_images = class_images[self.train_samples:]
            
            # Add to samples
            for key in selected_images:
                self.samples.append((key, class_idx))
        
        print(f"Found {len(self.samples)} images for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_key, label = self.samples[idx]
        
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=img_key)
            img_data = response['Body'].read()
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        except Exception as e:
            print(f"\nError loading {img_key}: {e}")
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Minimal transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create datasets with stratified split
print("\n--- Creating Datasets with Stratified Split ---")
train_dataset = S3ImageFolder(bucket_name, prefix, s3_client, transform=train_transform, 
                              limit_classes=1000, split='train', train_samples=24)
test_dataset = S3ImageFolder(bucket_name, prefix, s3_client, transform=test_transform, 
                             limit_classes=1000, split='test', train_samples=24)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- Using device: {device} ---")

# Load ResNet50
print("\n--- Loading ResNet50 ---")
model = models.resnet50(pretrained=True)#change this

# Modify final layer for number of classes
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training function with tqdm
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=True)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Testing function with tqdm
def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Testing', leave=True)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Training loop
print("\n--- Starting Training ---")
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"{'='*60}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    # Test
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Step scheduler
    scheduler.step()
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

print("\n--- Training Complete ---")

# Save model
# torch.save(model.state_dict(), 'resnet50_imagenet_s3.pth')
# print("Model saved to resnet50_imagenet_s3.pth")