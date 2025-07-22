import os
import shutil
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure(destination_root, classes):
    """
    Create the necessary directory structure for the dataset.
    """
    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            folder_path = os.path.join(destination_root, split, class_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Created directory: {folder_path}")

# --- MODIFIED FUNCTION ---
# Renamed from 'move_files_between_splits' to 'copy_files_between_splits'
def copy_files_between_splits(source_root, destination_root, source_split, target_split, ratio, classes, random_seed=42):
    """
    Copy a specific ratio of files from source_split to target_split.
    """
    random.seed(random_seed)
    total_copied = 0
    
    for class_name in classes:
        source_dir = os.path.join(source_root, source_split, class_name)
        target_dir = os.path.join(destination_root, target_split, class_name)
        
        # Ensure directories exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Get all files in the source directory
        try:
            files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
            num_files = len(files)
            
            if num_files == 0:
                logger.warning(f"No files found in {source_dir}")
                continue
                
            # Calculate number of files to copy
            num_to_copy = int(num_files * ratio)
            files_to_copy = random.sample(files, num_to_copy)
            
            logger.info(f"Copying {num_to_copy} files from {source_dir} to {target_dir}")
            
            # Copy the files
            for file in files_to_copy:
                source_path = os.path.join(source_dir, file)
                dest_path = os.path.join(target_dir, file)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    # --- KEY CHANGE ---
                    # The os.remove(source_path) line was deleted to prevent moving the file.
                    total_copied += 1
                except Exception as e:
                    logger.error(f"Failed to copy {file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing directory {source_dir}: {str(e)}")
    
    return total_copied

def setup_dataset(original_path, target_path, val_from_test_ratio=1.0, random_seed=42):
    """
    Set up the dataset with train, test, and validation splits.
    
    Args:
        original_path: Path to the original dataset
        target_path: Path where the reorganized dataset will be stored
        val_from_test_ratio: Ratio of test files to copy to validation
        random_seed: Random seed for reproducibility
    """
    # Get class names
    try:
        classes = os.listdir(os.path.join(original_path, 'train'))
        logger.info(f"Found classes: {classes}")
    except Exception as e:
        logger.error(f"Failed to get classes: {str(e)}")
        return False
    
    # Create directory structure
    create_directory_structure(target_path, classes)
    
    # First, copy all files to maintain the original structure
    for split in ['train', 'test']:
        for class_name in classes:
            source_class_dir = os.path.join(original_path, split, class_name)
            target_class_dir = os.path.join(target_path, split, class_name)
            
            try:
                # Get all files
                files = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]
                
                logger.info(f"Copying {len(files)} files from {source_class_dir} to {target_class_dir}")
                
                # Copy files
                for file in files:
                    source_file = os.path.join(source_class_dir, file)
                    target_file = os.path.join(target_class_dir, file)
                    shutil.copy2(source_file, target_file)
            except Exception as e:
                logger.error(f"Error copying files for {class_name} in {split}: {str(e)}")
    
    # Create validation set by copying files from test
    copied_to_val = copy_files_between_splits( # Using the modified function
        target_path, target_path, 
        'test', 'validation', 
        val_from_test_ratio, classes, random_seed
    )
    logger.info(f"Copied {copied_to_val} files from test to validation")
    
    return True

def count_files_in_dataset(dataset_path):
    """Print statistics about the dataset."""
    try:
        # Check for classes in the train directory first
        train_path = os.path.join(dataset_path, 'train')
        if not os.path.exists(train_path):
            print(f"Error: 'train' directory not found in {dataset_path}")
            return
            
        classes = os.listdir(train_path)
        
        for split in ['train', 'validation', 'test']:
            total = 0
            print(f"\n{split.upper()} SET:")
            
            for class_name in classes:
                class_dir = os.path.join(dataset_path, split, class_name)
                if os.path.exists(class_dir):
                    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
                    file_count = len(files)
                    total += file_count
                    print(f"  {class_name}: {file_count} files")
                else:
                    print(f"  {class_name}: Directory not found")
            
            print(f"  TOTAL: {total} files")
    except Exception as e:
        print(f"Error counting files: {str(e)}")

def main():
    # Define paths
    original_dataset = '/content/dataset/chest_xray'
    target_dataset = '/content/dataset/splited_chest_xray'
    
    # --- KEY CHANGE ---
    # Set ratio to 1.0 to copy ALL test files to validation
    val_from_test_ratio = 1.0 
    
    random_seed = 42
    
    print(f"Setting up dataset from {original_dataset} to {target_dataset}")
    success = setup_dataset(original_dataset, target_dataset, val_from_test_ratio, random_seed)
    
    if success:
        print("\nDataset setup complete. File counts:")
        count_files_in_dataset(target_dataset)
    else:
        print("Dataset setup failed. Check logs for details.")

if __name__ == "__main__":
    main()
