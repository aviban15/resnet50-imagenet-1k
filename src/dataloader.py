import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


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


def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4, limit_classes=None):
    """
    Returns DataLoaders for training and validation datasets from local directories.
    
    Parameters:
    - train_dir (str): Path to the training dataset directory.
    - val_dir (str): Path to the validation dataset directory.
    - batch_size (int): Batch size for DataLoader.
    - num_workers (int): Number of workers for data loading.
    - limit_classes (int): If specified, limits the number of classes to this number.
    
    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - val_loader (DataLoader): DataLoader for validation.
    """
    
    # Load training dataset using ImageFolder
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    # Optionally limit the number of classes for training
    if limit_classes:
        # Get the indices of the classes to include
        class_indices = list(range(limit_classes))
        train_dataset.samples = [
            (path, target) for path, target in train_dataset.samples 
            if target in class_indices
        ]
        train_dataset.targets = [target for _, target in train_dataset.samples]
    
    print(f"Found {len(train_dataset.classes)} classes in training data")
    
    # Load validation dataset using ImageFolder
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )
    
    print(f"Found {len(val_dataset.classes)} classes in validation data")
    
    # DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # DataLoader for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Example usage:
# Replace these with the actual paths on your EC2 instance
# train_dir = "/path/to/your/training_data"
# val_dir = "/path/to/your/validation_data"

# train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size=64, num_workers=8)
