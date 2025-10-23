# This script mounts an attached EBS volume to /mnt/data
# Enable execution permissions with: chmod +x utils/mount_ebs.sh
# Run this script with: sudo ./mount_ebs.sh

#!/bin/bash

# Find the device name
DEVICE_NAME=$(lsblk -o NAME,MOUNTPOINT | grep -v 'MOUNTPOINT' | grep -v '/' | grep -E 'nvme|xvd' | tail -n1 | awk '{print "/dev/"$1}')

# Create mount point
mkdir -p /mnt/data

# Mount the device
mount $DEVICE_NAME /mnt/data

# Change ownership
chown -R ubuntu:ubuntu /mnt/data

# Print success message
echo "Mounted successfully to /mnt/data"