import os
import shutil
import random
import logging
from collections import defaultdict

# --- Configuration ---
# Configure logging to show progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_and_organize_dataset(original_path, target_path, train_ratio=0.85, val_ratio=0.05, random_seed=42):
    """
    Combines images from original train/test folders, shuffles them, and creates a new
    dataset split into train, validation, and test sets while maintaining class balance.

    Args:
        original_path (str): Path to the original dataset (e.g., '/content/dataset/chest_xray').
        target_path (str): Path to save the new split dataset.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        random_seed (int): Seed for reproducibility.
    """
    random.seed(random_seed)

    # 1. Clean up target directory and create new structure
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
        logger.info(f"Removed existing directory: {target_path}")

    try:
        # Assumes class names are the folders inside the original 'train' directory
        classes = os.listdir(os.path.join(original_path, 'train'))
        logger.info(f"Found classes: {classes}")
    except FileNotFoundError:
        logger.error(f"Error: Could not find 'train' directory in '{original_path}'. Please check the path.")
        return

    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(target_path, split, class_name), exist_ok=True)
    logger.info(f"Created new directory structure at {target_path}")

    # 2. Gather all image file paths from both 'train' and 'test' folders
    all_files = defaultdict(list)
    for split in ['train', 'test']:
        split_dir = os.path.join(original_path, split)
        if not os.path.isdir(split_dir):
            logger.warning(f"Skipping non-existent directory: {split_dir}")
            continue
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning(f"Skipping non-existent class directory: {class_dir}")
                continue
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path):
                    all_files[class_name].append(file_path)

    # 3. Perform a stratified split for each class and copy files
    for class_name, files in all_files.items():
        random.shuffle(files)

        # Calculate the number of files for each set
        total_count = len(files)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)

        # Slice the list of files for each set
        train_files = files[:train_count]
        val_files = files[train_count : train_count + val_count]
        test_files = files[train_count + val_count:] # The remainder goes to the test set

        # Helper function to copy files
        def copy_files(file_list, split_name):
            dest_dir = os.path.join(target_path, split_name, class_name)
            for file_path in file_list:
                shutil.copy(file_path, dest_dir)

        # Copy the files to their new destination
        copy_files(train_files, 'train')
        copy_files(val_files, 'validation')
        copy_files(test_files, 'test')

        logger.info(f"Class '{class_name}': {len(train_files)} train | {len(val_files)} val | {len(test_files)} test")

    logger.info("✅ Dataset splitting and copying complete!")

def count_files_in_dataset(dataset_path):
    """Prints a summary of the file counts in the newly created dataset."""
    if not os.path.exists(dataset_path):
        logger.error(f"Verification failed: Dataset path does not exist: {dataset_path}")
        return

    try:
        classes = os.listdir(os.path.join(dataset_path, 'train'))
        grand_total = 0
        print("\n" + "="*40)
        print("🔍 Final Dataset Verification")
        print("="*40)

        for split in ['train', 'validation', 'test']:
            print(f"\n--- {split.upper()} SET ---")
            split_total = 0
            for class_name in classes:
                class_dir = os.path.join(dataset_path, split, class_name)
                num_files = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
                print(f"  {class_name}: {num_files} files")
                split_total += num_files
            print(f"  TOTAL: {split_total} files")
            grand_total += split_total

        print("\n" + "="*40)
        print(f"🎉 GRAND TOTAL: {grand_total} files")
        print("="*40)
    except Exception as e:
        logger.error(f"An error occurred while counting files: {e}")

def main():
    """Main function to define paths and run the script."""
    # --- Paths ---
    # IMPORTANT: Make sure this path points to your original 'chest_xray' dataset directory
    original_dataset = '/content/dataset/chest_xray'
    # This is where the new, split dataset will be saved
    target_dataset = '/content/dataset/splited_chest_xray'

    # --- Ratios ---
    train_ratio = 0.85
    val_ratio = 0.05
    # The remaining will be the test set (1.0 - 0.85 - 0.05 = 0.10)

    # Run the splitting process
    split_and_organize_dataset(original_dataset, target_dataset, train_ratio, val_ratio)

    # Verify the results by counting the files in the new structure
    count_files_in_dataset(target_dataset)

if __name__ == "__main__":
    main()
