import os
import shutil
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure(destination_root, classes):
    """
    Create the necessary directory structure (train, validation, test) for the dataset.
    """
    logger.info("Creating directory structure...")
    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            folder_path = Path(destination_root) / split / class_name
            folder_path.mkdir(parents=True, exist_ok=True)
            # No need to log every single directory creation, one message is enough.
    logger.info("Directory structure created successfully.")

def setup_dataset_split(original_path, target_path, splits_ratio=(0.8, 0.1, 0.1), random_seed=42):
    """
    Pools all data, shuffles it, and splits it into train, validation, and test sets.
    
    Args:
        original_path (str): Path to the original dataset with 'train' and 'test' folders.
        target_path (str): Path where the reorganized dataset will be stored.
        splits_ratio (tuple): A tuple containing the ratios for (train, validation, test).
        random_seed (int): Random seed for reproducibility.
    """
    random.seed(random_seed)
    
    try:
        # Assume classes are the same in both train and test directories
        train_path = Path(original_path) / 'train'
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
        if not classes:
            logger.error(f"No class directories found in {train_path}")
            return False
        logger.info(f"Found classes: {classes}")
    except FileNotFoundError:
        logger.error(f"Original training path not found at {original_path}/train")
        return False

    # Create the target directory structure
    create_directory_structure(target_path, classes)

    # Process each class individually to maintain class balance
    for class_name in classes:
        logger.info(f"--- Processing class: {class_name} ---")
        
        all_files = []
        # Gather all files from original train and test sets
        for split_name in ['train', 'test']:
            source_dir = Path(original_path) / split_name / class_name
            if source_dir.exists():
                files_in_dir = [p for p in source_dir.iterdir() if p.is_file()]
                all_files.extend(files_in_dir)
                logger.info(f"Found {len(files_in_dir)} files in {source_dir}")

        if not all_files:
            logger.warning(f"No files found for class {class_name}. Skipping.")
            continue
            
        # Shuffle the pooled files
        random.shuffle(all_files)
        total_files = len(all_files)
        logger.info(f"Total files for class '{class_name}': {total_files}. Shuffling and splitting.")

        # Calculate split indices
        train_ratio, val_ratio, _ = splits_ratio
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # Define file lists for each new split
        train_files = all_files[:train_end]
        validation_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]

        # Define target directories
        dest_train = Path(target_path) / 'train' / class_name
        dest_val = Path(target_path) / 'validation' / class_name
        dest_test = Path(target_path) / 'test' / class_name

        # Copy files to new directories
        for f_list, dest in [(train_files, dest_train), (validation_files, dest_val), (test_files, dest_test)]:
            logger.info(f"Copying {len(f_list)} files to {dest}...")
            for file_path in f_list:
                try:
                    shutil.copy2(file_path, dest)
                except Exception as e:
                    logger.error(f"Failed to copy {file_path} to {dest}: {e}")
    
    return True


def count_files_in_dataset(dataset_path):
    """Prints statistics about the dataset."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return
        
    try:
        # Infer classes from the 'train' directory
        classes = [d.name for d in (dataset_path / 'train').iterdir() if d.is_dir()]
        grand_total = 0
        
        for split in ['train', 'validation', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
                
            total = 0
            print(f"\n{split.upper()} SET:")
            
            for class_name in classes:
                class_dir = split_path / class_name
                if class_dir.exists():
                    file_count = len([f for f in class_dir.iterdir() if f.is_file()])
                    total += file_count
                    print(f"  {class_name}: {file_count} files")
            
            print(f"  TOTAL: {total} files")
            grand_total += total
        
        print(f"\nGRAND TOTAL IN ALL SPLITS: {grand_total} files")
    except Exception as e:
        print(f"Error counting files: {e}")

def main():
    """Main function to execute the script."""
    # Define paths
    original_dataset = '/content/dataset/chest_xray'
    target_dataset = '/content/dataset/splited_chest_xray'
    
    # Parameters for the 80/10/10 split
    split_ratios = (0.8, 0.1, 0.1) # Train, Validation, Test
    random_seed = 42
    
    # Clean up target directory if it exists to ensure a fresh split
    if os.path.exists(target_dataset):
        logger.warning(f"Target directory {target_dataset} already exists. It will be removed.")
        shutil.rmtree(target_dataset)
    
    print(f"Setting up dataset from {original_dataset} to {target_dataset}")
    print(f"Split ratio (Train/Val/Test): {split_ratios[0]*100}% / {split_ratios[1]*100}% / {split_ratios[2]*100}%")
    
    success = setup_dataset_split(original_dataset, target_dataset, split_ratios, random_seed)
    
    if success:
        print("\nDataset setup complete. Final file counts:")
        count_files_in_dataset(target_dataset)
    else:
        print("\nDataset setup failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
