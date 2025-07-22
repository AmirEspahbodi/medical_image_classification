import os
import shutil
import random
import logging
from collections import defaultdict

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure(root_path, splits, classes):
    """
    Creates the necessary directory structure for the new dataset.
    e.g., .../target_dir/train/NORMAL, .../target_dir/validation/PNEUMONIA
    """
    for split in splits:
        for class_name in classes:
            folder_path = os.path.join(root_path, split, class_name)
            os.makedirs(folder_path, exist_ok=True)
    logger.info("Created all necessary directories in the target path.")

def rebalance_and_split_dataset(source_dir, target_dir, train_ratio, val_ratio, random_seed=42):
    """
    Pools all images, shuffles, and splits them into train, validation, and test sets.

    Args:
        source_dir (str): Path to the original dataset (e.g., '/content/dataset/chest_xray').
        target_dir (str): Path to store the newly split dataset.
        train_ratio (float): The proportion of data to allocate to the training set.
        val_ratio (float): The proportion of data to allocate to the validation set.
        random_seed (int): Seed for reproducibility.
    """
    # Test ratio is implicitly calculated
    test_ratio = 1.0 - train_ratio - val_ratio
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        logger.error("Ratios must sum to 1.0")
        return False

    # Set random seed for reproducibility
    random.seed(random_seed)

    # 1. Discover classes from the source 'train' directory
    try:
        source_train_dir = os.path.join(source_dir, 'train')
        classes = [d for d in os.listdir(source_train_dir) if os.path.isdir(os.path.join(source_train_dir, d))]
        if not classes:
            logger.error("No class directories found in source 'train' folder.")
            return False
        logger.info(f"Discovered classes: {classes}")
    except FileNotFoundError:
        logger.error(f"Source directory '{source_train_dir}' not found.")
        return False

    # 2. Create the target directory structure
    splits = ['train', 'validation', 'test']
    create_directory_structure(target_dir, splits, classes)

    # 3. Pool, shuffle, and split files for each class
    for class_name in classes:
        logger.info(f"--- Processing class: {class_name} ---")
        
        all_image_paths = []
        # Gather files from both original train and test folders
        for split in ['train', 'test']:
            class_folder = os.path.join(source_dir, split, class_name)
            if os.path.exists(class_folder):
                files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
                all_image_paths.extend(files)
        
        logger.info(f"Found {len(all_image_paths)} total images for class '{class_name}'.")

        # Shuffle the pooled list of images
        random.shuffle(all_image_paths)

        # Calculate split indices
        total_files = len(all_image_paths)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # Slice the list into three sets
        train_set = all_image_paths[:train_end]
        val_set = all_image_paths[train_end:val_end]
        test_set = all_image_paths[val_end:]
        
        # Create a dictionary to easily iterate and copy files
        sets_to_copy = {
            'train': train_set,
            'validation': val_set,
            'test': test_set
        }

        # 4. Copy files to their new destination
        for split_name, file_list in sets_to_copy.items():
            destination_folder = os.path.join(target_dir, split_name, class_name)
            for src_path in file_list:
                try:
                    shutil.copy2(src_path, destination_folder)
                except Exception as e:
                    logger.error(f"Failed to copy {src_path} to {destination_folder}: {e}")
            logger.info(f"Copied {len(file_list)} files to {destination_folder}")

    return True

def count_files_in_dataset(dataset_path):
    """Prints statistics about the dataset splits and classes."""
    overall_total = 0
    try:
        splits = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        for split in sorted(splits):
            split_path = os.path.join(dataset_path, split)
            split_total = 0
            print(f"\n{split.upper()} SET:")
            
            classes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
            for class_name in classes:
                class_dir = os.path.join(split_path, class_name)
                file_count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
                split_total += file_count
                print(f"  {class_name}: {file_count} files")
            
            print(f"  TOTAL: {split_total} files")
            overall_total += split_total
        
        print(f"\n----------------------------------")
        print(f"OVERALL DATASET TOTAL: {overall_total} files")

    except FileNotFoundError:
        print(f"Error: Directory not found at {dataset_path}")
    except Exception as e:
        print(f"An error occurred while counting files: {str(e)}")

def main():
    # --- User-defined Parameters ---
    
    # Original dataset with 'train' and 'test' folders
    original_dataset = '/content/dataset/chest_xray'
    
    # Directory where the new 80/10/10 split will be saved
    target_dataset = '/content/dataset/splited_chest_xray_80_10_10'

    # Define the desired split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    # TEST_RATIO is automatically calculated as 1.0 - TRAIN_RATIO - VAL_RATIO
    
    # Random seed for shuffling, ensuring the split is the same every time
    RANDOM_SEED = 42
    
    # --- Execution ---
    
    # Clean up target directory if it exists, to ensure a fresh start
    if os.path.exists(target_dataset):
        logger.warning(f"Target directory '{target_dataset}' already exists. Removing it for a fresh split.")
        shutil.rmtree(target_dataset)
        
    print(f"Starting dataset split from '{original_dataset}' to '{target_dataset}'...")
    
    success = rebalance_and_split_dataset(
        source_dir=original_dataset,
        target_dir=target_dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED
    )
    
    if success:
        print("\n✅ Dataset split and reorganization complete.")
        print("Final file counts in the new directory:")
        count_files_in_dataset(target_dataset)
    else:
        print("\n❌ Dataset setup failed. Please check the log messages for errors.")

if __name__ == "__main__":
    main()
