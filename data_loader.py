import boto3
from PIL import Image
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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


def get_dataloaders(batch_size=32, num_workers=4, limit_classes=None):
    """
    Returns DataLoaders for training and validation datasets from S3.
    S3 bucket names and prefixes are hardcoded for fixed datasets.
    """
    # Hardcoded S3 paths
    train_bucket = 'imagenet-30sample'
    train_prefix = 'imagenet_sampled30/'
    val_bucket = 'imagenet-val-5'
    val_prefix = 'val/'

    train_dataset = S3ImageFolderTrain(
        bucket_name=train_bucket,
        prefix=train_prefix,
        s3_client=s3_client,
        transform=train_transform,
        limit_classes=limit_classes
    )

    val_dataset = S3ImageFolderVal(
        bucket_name=val_bucket,
        prefix=val_prefix,
        s3_client=s3_client,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes
