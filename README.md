# ResNet-50 ImageNet-1K Implementation

A complete implementation of ResNet-50 architecture for ImageNet-1K classification, including training pipeline, data augmentation, and AWS deployment setup.

## 🛠️ Install and Quick Start

### Dependencies
```bash
uv sync
```

### Quick Start
1. **Clone repository** and navigate to project directory
2. **Install deps**: `uv sync` or install from the toml file
3. **Prepare data**: Arrange ImageNet as `train/` and `val/` per class
4. **Configure paths**: Update `src/config.py`
5. **Train**: `python src/train.py`
6. **Launch Gradio App** : `python src/gradio/app.py`

### Configuration (src/config.py)
```python
TRAIN_DIR = "/data/ILSVRC/Data/CLS-LOC/train"
VAL_DIR = "/data/ILSVRC/Data/CLS-LOC/val"
BATCH_SIZE = 128
NUM_WORKERS = 6
NUM_EPOCHS = 90
```

## 📁 Project Structure

```
resnet50-imagenet-1k/
├── docs/                         # Documentation and training logs
│   ├── ec2-complete.png
│   ├── ec2-training.png
│   └── training_90_epochs.md
├── src/                          # Core model and training code
│   ├── config.py                 # Configuration parameters
│   ├── dataloader.py             # Data loading and preprocessing
│   ├── model.py                  # ResNet-50 model implementation
│   ├── train.py                  # Main training script
│   └── src/gradio/               # Gradio demo app
│       ├── app.py
│       ├── imagenet_classes.txt
│       └── requirements.txt
├── utils/                        # Utilities and helpers
│   ├── check_gpu.py              
│   ├── check_params.py           
│   ├── convert_val.py            
│   └── mount_ebs.sh
├── .python-version               # Python version
├── pyproject.toml                # Project dependencies
└── README.md                     # This file
```

## 🧰 Data and Dataloaders

### Training Augmentations (src/dataloader.py)
- **RandomResizedCrop (224)**: scale=(0.08, 1.0), ratio=(3/4, 4/3)
  - Improves invariance to scale/aspect ratio; bilinear + antialias
- **RandomHorizontalFlip (p=0.5)**: mirror invariance
- **Normalize**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Validation Transform
- **Resize 256 → CenterCrop 224**
- **Normalize**: same ImageNet stats

### DataLoader Configuration
- **Batch size**: 128
- **Number of Workers**: 6; **pin_memory**: True
- **Shuffle**: train=True, val=False

## 🏗️ Model Architecture (src/model.py)

### ResNet-50 Overview
- **Parameters**: 25,557,032
- **Input**: 224×224×3
- **Head**: GlobalAvgPool → Linear(2048→1000)

### Stem
- **7×7 Conv (stride=2, pad=3)**: channels 3→64, size 224→112
- **3×3 MaxPool (stride=2, pad=1)**: size 112→56

### Residual Stages (Bottleneck blocks, expansion×4)
- **Stage 1 (C=64, blocks=3)**: output 256 channels, 56×56, stride=1
- **Stage 2 (C=128, blocks=4)**: output 512 channels, 28×28, stride=2 on first block
- **Stage 3 (C=256, blocks=6)**: output 1024 channels, 14×14, stride=2 on first block
- **Stage 4 (C=512, blocks=3)**: output 2048 channels, 7×7, stride=2 on first block

### Bottleneck Block (per block)
- **1×1 Conv**: reduce channels to base width
- **3×3 Conv**: spatial processing; stride=2 when downsampling
- **1×1 Conv**: expand to `planes×4`
- **Skip path**: identity or 1×1 projection when shape/stride changes

## 🧪 Training Setup (src/train.py)

### Optimization
- **Optimizer**: SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss
- **Epochs**: from `src/config.py` (`NUM_EPOCHS`)

### Scheduler: Cosine Annealing
- **T_max**: `NUM_EPOCHS`, **eta_min**: 1e-4
- **Step timing**: `scheduler.step()` once per epoch, after training
- **Schedule**: η_t = η_min + (η_max−η_min)·(1+cos(π·t/T_max))/2

### Mixed Precision
- Enabled via `torch.amp` with `GradScaler` and `autocast`

### Logging and Artifacts
- **Logs**: `logs/training_{YYYYMMDD_%H%M%S_%Z}.log`
- **Latest checkpoint**: `checkpoints/checkpoint.pth` saved every epoch
- **Periodic checkpoints**: additional save every 10 epochs (suffix `_epoch.pth`)
- **Best weights**: `checkpoints/best_model_weights.pth` on improved val accuracy
- **Progress**: tqdm bar and per-epoch summary (LR, loss, accuracy)

### Resume Training
- **Resume**: set `RESUME_TRAINING=True` to load `checkpoints/checkpoint.pth`
- **Load previous weights**: `LOAD_PREV_WEIGHTS=True` loads best weights before training

## 🔍 Utilities

- **Parameter check**: `utils/check_params.py` confirms ~25.6M parameters
- **GPU check**: `utils/check_gpu.py` prints device details
- **Validation set converter**: `utils/convert_val.py` organizes ImageNet val set

## 🚀 AWS Setup and Deployment

### Data Preparation
1. **Download**: Acquire ImageNet-1K (e.g., from Kaggle)
2. **Attach EBS**: Create and mount extra volume (e.g., 500GB+ gp3)
3. **Place data**: `/mnt/data/ILSVRC/Data/CLS-LOC/{train,val}`

### Instance and Storage
- **Instance**: GPU - g5.2xlarge
- **Root volume**: 60GB gp3
- **Dataset volume**: 420GB gp3

### Connect and Prepare
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```
1. `python utils/check_gpu.py` — verify CUDA and GPU specs
2. `python utils/convert_val.py -d <val_dir> -l LOC_val_solution.csv` — organize val set
3. `python src/train.py` — start training

### Utilities
- **check_gpu.py**: prints device count, name, mem, compute capability
- **convert_val.py**: moves val images into class folders (ImageFolder ready)

---

This repository delivers a clean ResNet-50 implementation, a robust training loop with cosine annealing, reproducible dataloaders and augmentations, and pragmatic AWS setup notes to train at ImageNet scale.

## 🎛️ Gradio App

- **Launch**: `python src/gradio/app.py` (starts a local web UI for image classification)
- **Classes**: reads ImageNet-1K labels from `src/gradio/imagenet_classes.txt`
- **Weights**: uses trained weights if `checkpoints/best_model_weights.pth` exists; else initializes randomly
- **Deps**: optional extras listed in `src/gradio/requirements.txt`

## 📈 Results

- **Top-1 validation accuracy**: 76.15% after 90 epochs on ImageNet-1K
- **Setup**: batch size 128, SGD + cosine schedule, mixed precision on a g5.2xlarge

## 📒 Training Log Summary (epochs 10→90)

| Epoch | LR       | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|------:|----------|-----------:|---------------|---------:|------------:|
| 10    | 0.097555 | 3.4439     | 42.55         | 2.3681   | 45.81       |
| 20    | 0.089411 | 2.9090     | 45.25         | 2.2823   | 47.29       |
| 30    | 0.076519 | 3.4572     | 47.40         | 2.1269   | 50.71       |
| 40    | 0.060435 | 2.2385     | 49.96         | 1.9999   | 53.19       |
| 50    | 0.043098 | 1.9958     | 53.22         | 1.7543   | 58.11       |
| 60    | 0.026600 | 2.7037     | 57.50         | 1.5575   | 61.98       |
| 70    | 0.012930 | 2.1797     | 63.12         | 1.3720   | 66.28       |
| 80    | 0.003737 | 2.0871     | 70.68         | 1.0846   | 72.60       |
| 90    | 0.000130 | 1.5460     | 77.23         | 0.9459   | 76.15       |
