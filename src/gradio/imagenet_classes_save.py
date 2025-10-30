from torchvision.models import resnet50, ResNet50_Weights

# Load default ImageNet weights
weights = ResNet50_Weights.DEFAULT

# Extract human-readable class labels
class_names = weights.meta["categories"]

# Optional: save to a text file
with open("imagenet_classes.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"✅ Saved {len(class_names)} ImageNet classes to imagenet_classes.txt")
print("Example classes:", class_names[:10])
