import boto3
from PIL import Image
import io
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Initialize S3 client
s3_client = boto3.client('s3')

# Custom S3ImageFolder class for training data (class-based structure)
class S3ImageFolderTrain(Dataset):
    """Custom PyTorch Dataset that reads ImageNet training data from S3 (class folders)"""
    
    def __init__(self, bucket_name, prefix, s3_client, transform=None, limit_classes=None):
        self.bucket = bucket_name
        self.prefix = prefix
        self.s3 = s3_client
        self.transform = transform
        
        print(f"Loading class folders from S3 bucket: {bucket_name}/{prefix}")
        
        paginator = self.s3.get_paginator('list_objects_v2')
        
        # Get class folders
        class_folders = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for folder in page['CommonPrefixes']:
                    folder_name = folder['Prefix'].replace(self.prefix, '').rstrip('/')
                    if folder_name:
                        class_folders.add(folder_name)
        
        self.classes = sorted(list(class_folders))
        if limit_classes:
            self.classes = self.classes[:limit_classes]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} classes")
        
        # Build list of images
        self.samples = []
        print(f"Loading image paths from S3...")
        
        for class_name in tqdm(self.classes, desc="Loading classes"):
            class_prefix = f"{self.prefix}{class_name}/"
            class_idx = self.class_to_idx[class_name]
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=class_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith(('.jpg', '.jpeg', '.JPEG', '.png', '.PNG')):
                            self.samples.append((key, class_idx))
        
        print(f"Found {len(self.samples)} training images")
    
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


# Custom S3 Dataset for validation data (flat structure with class subfolders)
class S3ImageFolderVal(Dataset):
    """Custom PyTorch Dataset that reads ImageNet validation data from S3 (organized class folders)"""
    
    def __init__(self, bucket_name, prefix, s3_client, transform=None, class_to_idx=None):
        self.bucket = bucket_name
        self.prefix = prefix
        self.s3 = s3_client
        self.transform = transform
        
        print(f"Loading validation data from S3 bucket: {bucket_name}/{prefix}")
        
        paginator = self.s3.get_paginator('list_objects_v2')
        
        # Get class folders
        class_folders = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for folder in page['CommonPrefixes']:
                    folder_name = folder['Prefix'].replace(self.prefix, '').rstrip('/')
                    if folder_name:
                        class_folders.add(folder_name)
        
        self.classes = sorted(list(class_folders))
        
        # Use provided class_to_idx mapping if available (to match train classes)
        if class_to_idx:
            self.class_to_idx = class_to_idx
            # Filter to only classes that exist in training
            self.classes = [cls for cls in self.classes if cls in class_to_idx]
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} validation classes")
        
        # Build list of images
        self.samples = []
        print(f"Loading validation image paths from S3...")
        
        for class_name in tqdm(self.classes, desc="Loading validation classes"):
            class_prefix = f"{self.prefix}{class_name}/"
            class_idx = self.class_to_idx[class_name]
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=class_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith(('.jpg', '.jpeg', '.JPEG', '.png', '.PNG')):
                            self.samples.append((key, class_idx))
        
        print(f"Found {len(self.samples)} validation images")
    
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


# Training data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Validation data transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create datasets from separate S3 sources
print("\n--- Creating Datasets from Separate S3 Sources ---")

# Training dataset
train_bucket = 'imagenet-30sample'
train_prefix = 'imagenet_sampled30/'
train_dataset = S3ImageFolderTrain(
    bucket_name=train_bucket,
    prefix=train_prefix,
    s3_client=s3_client,
    transform=train_transform,
    limit_classes=None  # Set to None for all classes, or a number to limit
)

# Validation dataset
val_bucket = 'imagenet-val-5'
val_prefix = 'val/'
val_dataset = S3ImageFolderVal(
    bucket_name=val_bucket,
    prefix=val_prefix,
    s3_client=s3_client,
    transform=val_transform,
    class_to_idx=train_dataset.class_to_idx  # Use same class mapping as training
)

print(f"\nTrain size: {len(train_dataset)}, Val size: {len(val_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- Using device: {device} ---")

# Load ResNet50
print("\n--- Loading ResNet50 ---")
model = models.resnet50()#pretrained=True

# Modify final layer for number of classes
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function with tqdm
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Validating', leave=True)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Training loop
print("\n--- Starting Training ---")
num_epochs = 2

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"{'='*60}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    # Validate
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Step scheduler
    scheduler.step()
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

print("\n--- Training Complete ---")

# Save model locally and to S3
print("\n--- Saving Model ---")
local_model_path = 'resnet50_imagenet_s3.pth'
torch.save(model.state_dict(), local_model_path)
print(f"✓ Model saved locally to {local_model_path}")

# Upload to S3 for persistence (CRITICAL for EC2!)
try:
    model_bucket = 'imagenet-30sample'  # Using existing bucket
    model_key = f'trained_models/resnet50_{num_classes}classes_{num_epochs}epochs.pth'
    
    print(f"Uploading to S3...")
    s3_client.upload_file(local_model_path, model_bucket, model_key)
    print(f"✓ Model uploaded to s3://{model_bucket}/{model_key}")
    print(f"\nIMPORTANT: Model is now safely stored in S3!")
    print(f"To load later, download from: s3://{model_bucket}/{model_key}")
except Exception as e:
    print(f"❌ ERROR: Could not upload to S3: {e}")
    print(f"⚠️  WARNING: Model only exists locally and will be LOST if EC2 stops!")
    print(f"Local path: {os.path.abspath(local_model_path)}")