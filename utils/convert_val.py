"""
Description:
Convert validation images into class-specific folders based on a mapping file

Usage:
python3 convert_val.py \
  -d ILSVRC/Data/CLS-LOC/val \
  -l LOC_val_solution.csv

"""

import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="Directory containing validation images", required=True)
parser.add_argument("-l", "--labels", help="File with image name to class label mapping", required=True)
args = parser.parse_args()

# Keep track of created class directories
processed_classes = set()

with open(args.labels, "r") as file:
    # Skip header line if present
    header = next(file).strip()
    if not header or " " not in header and "," not in header:
        # If the first line doesn’t look like a header, rewind
        file.seek(0)

    for line in file:
        line = line.strip()
        if not line:
            continue

        # Support both comma- or space-separated mappings
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue

        img_name, class_name = parts[0], parts[1]
        src = os.path.join(args.dir, img_name + ".JPEG")
        dst_dir = os.path.join(args.dir, class_name)
        dst = os.path.join(dst_dir, img_name + ".JPEG")

        # Create target class directory if needed
        if class_name not in processed_classes:
            os.makedirs(dst_dir, exist_ok=True)
            processed_classes.add(class_name)

        # Move image if it exists
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"Warning: {src} not found.")