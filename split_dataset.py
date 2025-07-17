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

def move_files_between_splits(source_root, destination_root, source_split, target_split, ratio, classes, random_seed=42):
    """
    Move a specific ratio of files from source_split to target_split.
    """
    random.seed(random_seed)
    total_moved = 0
    
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
                
            # Calculate number of files to move
            num_to_move = max(1, int(num_files * ratio))
            files_to_move = random.sample(files, num_to_move)
            
            logger.info(f"Moving {num_to_move} files from {source_dir} to {target_dir}")
            
            # Move the files
            for file in files_to_move:
                source_path = os.path.join(source_dir, file)
                dest_path = os.path.join(target_dir, file)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    os.remove(source_path)
                    total_moved += 1
                except Exception as e:
                    logger.error(f"Failed to move {file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing directory {source_dir}: {str(e)}")
    
    return total_moved

def setup_dataset(original_path, target_path, val_from_test_ratio=0.1, random_seed=42):
    """
    Set up the dataset with train, test, and validation splits.
    
    Args:
        original_path: Path to the original dataset
        target_path: Path where the reorganized dataset will be stored
        val_from_test_ratio: Ratio of test files to move to validation
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
    
    # Create validation set by moving files from test
    moved_to_val = move_files_between_splits(
        target_path, target_path, 
        'test', 'validation', 
        val_from_test_ratio, classes, random_seed
    )
    logger.info(f"Moved {moved_to_val} files from test to validation")
    
    return True

def count_files_in_dataset(dataset_path):
    """Print statistics about the dataset."""
    try:
        classes = os.listdir(os.path.join(dataset_path, 'train'))
        
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
    
    # Parameters
    # Change this to modify where validation data comes from
    val_from_test_ratio = 0.2  # Portion of test data to move to validation
    
    # To create validation from train instead, set:
    # val_from_test_ratio = 0
    # val_from_train_ratio = 0.1 (and add this parameter to the setup_dataset function call)
    
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