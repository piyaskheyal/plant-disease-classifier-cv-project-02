import os
import shutil
import random
from pathlib import Path

# Set paths
SOURCE_DIR = 'PlantVillage'  # your original dataset folder
DEST_DIR = 'dataset_split'  # output folder for train/val
SPLIT_RATIO = 0.8  # 80% train, 20% val

# Set random seed for reproducibility
random.seed(42)

# Get all class folders
class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for class_name in class_names:
    class_dir = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Create target class folders
    train_class_dir = os.path.join(DEST_DIR, 'train', class_name)
    val_class_dir = os.path.join(DEST_DIR, 'val', class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy train images
    for img_name in train_images:
        src_path = os.path.join(class_dir, img_name)
        dst_path = os.path.join(train_class_dir, img_name)
        shutil.copy2(src_path, dst_path)

    # Copy val images
    for img_name in val_images:
        src_path = os.path.join(class_dir, img_name)
        dst_path = os.path.join(val_class_dir, img_name)
        shutil.copy2(src_path, dst_path)

print("✅ Dataset split completed: 80% train / 20% val")
