import os
import shutil
import random
from pathlib import Path
import logging

# Configure logging to provide progress updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure(destination_root, classes):
    """
    Creates the necessary directory structure (train, validation, test) for each class.
    """
    logger.info("Creating directory structure...")
    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            folder_path = os.path.join(destination_root, split, class_name)
            os.makedirs(folder_path, exist_ok=True)
    logger.info("Directory structure created successfully.")

def copy_files_to_validation(source_root, dest_root, source_split, ratio, classes, random_seed):
    """
    Copies a specified ratio of files from a source split (train or test) to the validation set.
    """
    random.seed(random_seed)
    total_copied = 0
    
    logger.info(f"Copying {ratio:.0%} of files from '{source_split}' set to 'validation' set.")
    
    for class_name in classes:
        source_dir = os.path.join(source_root, source_split, class_name)
        target_dir = os.path.join(dest_root, 'validation', class_name)
        
        try:
            files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
            if not files:
                logger.warning(f"No files found in {source_dir}")
                continue
                
            # Calculate the number of files to copy
            num_to_copy = int(len(files) * ratio)
            files_to_copy = random.sample(files, num_to_copy)
            
            logger.info(f"Copying {len(files_to_copy)} files from {source_dir} to {target_dir}")
            
            # Copy the selected files
            for file_name in files_to_copy:
                source_path = os.path.join(source_dir, file_name)
                dest_path = os.path.join(target_dir, file_name)
                shutil.copy2(source_path, dest_path) # Use copy2 to preserve metadata
                total_copied += 1

        except FileNotFoundError:
            logger.error(f"Source directory not found: {source_dir}")
        except Exception as e:
            logger.error(f"Error processing directory {source_dir}: {e}")
            
    return total_copied

def setup_dataset(original_path, target_path, val_ratio=0.1, random_seed=42):
    """
    Sets up the new dataset structure by copying files.
    
    Args:
        original_path (str): Path to the original dataset.
        target_path (str): Path where the new dataset will be created.
        val_ratio (float): Ratio of images from train/test to copy to the validation set.
        random_seed (int): Seed for reproducibility.
    """
    try:
        # Get class names from the original training directory
        classes = [d for d in os.listdir(os.path.join(original_path, 'train')) if os.path.isdir(os.path.join(original_path, 'train', d))]
        if not classes:
            logger.error(f"No class subdirectories found in {os.path.join(original_path, 'train')}")
            return False
        logger.info(f"Detected classes: {classes}")
    except FileNotFoundError:
        logger.error(f"Original dataset path not found: {original_path}")
        return False

    # 1. Create the full directory structure in the target location
    create_directory_structure(target_path, classes)

    # 2. Copy all original train and test files to the new train and test directories
    logger.info("--- Step 1: Copying original train and test sets to new location ---")
    for split in ['train', 'test']:
        for class_name in classes:
            source_class_dir = os.path.join(original_path, split, class_name)
            target_class_dir = os.path.join(target_path, split, class_name)
            try:
                shutil.copytree(source_class_dir, target_class_dir, dirs_exist_ok=True)
                logger.info(f"Successfully copied all files from {source_class_dir} to {target_class_dir}")
            except Exception as e:
                logger.error(f"Failed to copy tree from {source_class_dir}: {e}")
    
    # 3. Populate the validation set by COPYING from the ORIGINAL train and test sets
    logger.info("--- Step 2: Creating validation set by copying from original sources ---")
    
    # Copy 10% from the original training set to validation
    copied_from_train = copy_files_to_validation(original_path, target_path, 'train', val_ratio, classes, random_seed)
    
    # Copy 10% from the original test set to validation
    # Use a different seed for sampling from the test set to ensure different random choices
    copied_from_test = copy_files_to_validation(original_path, target_path, 'test', val_ratio, classes, random_seed + 1)
    
    logger.info(f"Total files copied to validation: {copied_from_train} from train, {copied_from_test} from test.")
    return True

def count_files_in_dataset(dataset_path):
    """Prints the number of files in each split and class of the dataset."""
    print("\n" + "="*30)
    print("📊 FINAL DATASET FILE COUNT")
    print("="*30)
    try:
        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                print(f"\n{split.upper()} SET: Not found")
                continue
            
            total_split_files = 0
            print(f"\n📁 {split.upper()} SET:")
            
            classes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
            for class_name in classes:
                class_dir = os.path.join(split_path, class_name)
                file_count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
                print(f"  - {class_name}: {file_count} files")
                total_split_files += file_count
            print(f"  --------------------")
            print(f"  ✅ TOTAL: {total_split_files} files")
            
    except Exception as e:
        print(f"Error counting files: {e}")

def main():
    # --- Parameters ---
    # NOTE: Ensure these paths exist in your environment.
    original_dataset = '/content/dataset/chest_xray' 
    target_dataset = '/content/dataset/splited_chest_xray'
    validation_ratio = 0.10  # 10% for validation set from both train and test
    random_seed = 42
    
    # Clean up target directory if it exists to ensure a fresh start
    if os.path.exists(target_dataset):
        logger.warning(f"Target directory {target_dataset} already exists. Removing it for a fresh start.")
        shutil.rmtree(target_dataset)

    print(f"🚀 Starting dataset setup...")
    print(f"Source: {original_dataset}")
    print(f"Destination: {target_dataset}")
    
    success = setup_dataset(original_dataset, target_dataset, validation_ratio, random_seed)
    
    if success:
        print("\n🎉 Dataset setup completed successfully!")
        count_files_in_dataset(target_dataset)
    else:
        print("\n❌ Dataset setup failed. Please check the log messages for errors.")

if __name__ == "__main__":
    main()
