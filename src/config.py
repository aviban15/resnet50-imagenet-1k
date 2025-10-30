# Dataloader configuration
TRAIN_DIR = "/mnt/data/ILSVRC/Data/CLS-LOC/train"
VAL_DIR = "/mnt/data/ILSVRC/Data/CLS-LOC/val"
BATCH_SIZE = 128
NUM_WORKERS = 6

# Model configuration
NUM_CLASSES = 1000

# Training configuration
NUM_EPOCHS = 90
LOAD_PREV_WEIGHTS = False
RESUME_TRAINING = False