import os
import shutil
import random
from PIL import Image
import numpy as np

# ── Path Settings ──────────────────────────────────────────
RAW_DATA_PATH = "data/raw/plantvillage dataset/color"
PROCESSED_PATH = "data/processed"

TRAIN_PATH = os.path.join(PROCESSED_PATH, "train")
VAL_PATH   = os.path.join(PROCESSED_PATH, "val")
TEST_PATH  = os.path.join(PROCESSED_PATH, "test")

# Image size our model will use
IMAGE_SIZE = (224, 224)

# Split ratio - 80% train, 10% val, 10% test
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

def create_folders():
    """Create train, val, test folders"""
    print("Creating folders...")
    for folder in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        os.makedirs(folder, exist_ok=True)
    print("Folders created successfully!")

def process_and_split_data():
    """Read images, resize them and split into train/val/test"""
    
    # Get all disease class folders
    classes = os.listdir(RAW_DATA_PATH)
    print(f"Total classes found: {len(classes)}")
    
    for class_name in classes:
        class_path = os.path.join(RAW_DATA_PATH, class_name)
        
        if not os.path.isdir(class_path):
            continue
            
        print(f"Processing: {class_name}")
        
        # Get all images in this class
        images = os.listdir(class_path)
        random.shuffle(images)  # Shuffle randomly
        
        # Calculate split numbers
        total     = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end   = int(total * (TRAIN_RATIO + VAL_RATIO))
        
        train_images = images[:train_end]
        val_images   = images[train_end:val_end]
        test_images  = images[val_end:]
        
        # Copy images to correct folders
        for split_name, split_images, split_path in [
            ("train", train_images, TRAIN_PATH),
            ("val",   val_images,   VAL_PATH),
            ("test",  test_images,  TEST_PATH)
        ]:
            # Create class folder inside split folder
            dest_folder = os.path.join(split_path, class_name)
            os.makedirs(dest_folder, exist_ok=True)
            
            for img_name in split_images:
                src  = os.path.join(class_path, img_name)
                dest = os.path.join(dest_folder, img_name)
                
                try:
                    # Open, resize and save image
                    img = Image.open(src).convert("RGB")
                    img = img.resize(IMAGE_SIZE)
                    img.save(dest)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
        
    print("All images processed and split successfully!")

def show_summary():
    """Show how many images are in each split"""
    print("\n── Dataset Summary ──")
    for split_name, split_path in [
        ("Train", TRAIN_PATH),
        ("Val",   VAL_PATH),
        ("Test",  TEST_PATH)
    ]:
        total = 0
        if os.path.exists(split_path):
            for class_folder in os.listdir(split_path):
                class_full = os.path.join(split_path, class_folder)
                if os.path.isdir(class_full):
                    total += len(os.listdir(class_full))
        print(f"{split_name}: {total} images")

if __name__ == "__main__":
    print("Starting preprocessing...")
    create_folders()
    process_and_split_data()
    show_summary()
    print("\nPreprocessing Complete!")