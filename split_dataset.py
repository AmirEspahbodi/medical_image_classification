import os
import shutil
import random
import logging
from pathlib import Path

# Configure logging to show informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure(destination_root, classes):
    """
    Creates the necessary directory structure (train, validation, test) for each class.
    """
    logger.info("Creating new directory structure...")
    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            folder_path = Path(destination_root) / split / class_name
            # Create directories, including any necessary parent folders
            folder_path.mkdir(parents=True, exist_ok=True)

def setup_dataset(original_path, target_path, split_ratios, random_seed=42):
    """
    Pools all images, shuffles them, and splits them into new train, validation,
    and test sets based on the specified ratios.

    Args:
        original_path (str): Path to the original dataset with 'train' and 'test' folders.
        target_path (str): Path where the reorganized dataset will be stored.
        split_ratios (dict): Dictionary with ratios for 'train', 'validation', and 'test'.
        random_seed (int): Seed for the random number generator for reproducibility.
    """
    random.seed(random_seed)
    original_path = Path(original_path)
    target_path = Path(target_path)

    # 1. Get class names from the original 'train' directory
    try:
        # Assumes the same class subdirectories exist in both 'train' and 'test'
        classes = [d.name for d in (original_path / 'train').iterdir() if d.is_dir()]
        logger.info(f"Detected classes: {classes}")
    except FileNotFoundError:
        logger.error(f"Error: Could not find 'train' directory in {original_path}")
        return False
        
    if not classes:
        logger.error(f"Error: No class subdirectories found in {original_path / 'train'}")
        return False

    # 2. Create the new directory structure in the target path
    create_directory_structure(target_path, classes)

    # 3. Process each class separately to maintain class balance
    for class_name in classes:
        logger.info(f"--- Processing class: {class_name} ---")
        
        # --- Pool all image files for the current class from original train/test ---
        all_files = []
        train_dir = original_path / 'train' / class_name
        test_dir = original_path / 'test' / class_name
        
        if train_dir.exists():
            all_files.extend(list(train_dir.glob('*.*'))) # Using glob to get file paths
        if test_dir.exists():
            all_files.extend(list(test_dir.glob('*.*')))

        if not all_files:
            logger.warning(f"No images found for class '{class_name}'. Skipping.")
            continue
            
        logger.info(f"Found {len(all_files)} total images for class '{class_name}'.")

        # --- Shuffle the pooled list of files randomly ---
        random.shuffle(all_files)
        
        # --- Calculate split points based on ratios ---
        total_count = len(all_files)
        train_count = int(total_count * split_ratios['train'])
        val_count = int(total_count * split_ratios['validation'])
        
        # Define the slices for each set
        train_files = all_files[:train_count]
        validation_files = all_files[train_count : train_count + val_count]
        test_files = all_files[train_count + val_count:]
        
        # --- Create a map for easy file copying ---
        split_map = {
            'train': train_files,
            'validation': validation_files,
            'test': test_files
        }

        # --- Copy files to their new destination folders ---
        for split_name, files_to_copy in split_map.items():
            destination_dir = target_path / split_name / class_name
            logger.info(f"Copying {len(files_to_copy)} files to {destination_dir}")
            for src_path in files_to_copy:
                try:
                    # copy2 preserves metadata
                    shutil.copy2(src_path, destination_dir / src_path.name)
                except Exception as e:
                    logger.error(f"Failed to copy {src_path} to {destination_dir}: {e}")
    
    logger.info("✅ Dataset splitting and copying completed successfully.")
    return True

def count_files_in_dataset(dataset_path):
    """
    Counts and prints the number of files in each subdirectory of the dataset.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return
        
    try:
        classes = sorted([d.name for d in (dataset_path / 'train').iterdir() if d.is_dir()])
    except FileNotFoundError:
        print(f"Error: Could not find 'train' directory in {dataset_path}.")
        return

    grand_total = 0
    print("\n" + "="*40)
    print("      FINAL DATASET FILE COUNT")
    print("="*40)
    for split in ['train', 'validation', 'test']:
        split_total = 0
        print(f"\n📁 {split.upper()} SET:")
        split_path = dataset_path / split
        
        for class_name in classes:
            class_dir = split_path / class_name
            file_count = len([f for f in class_dir.iterdir() if f.is_file()]) if class_dir.exists() else 0
            split_total += file_count
            print(f"  - {class_name}: {file_count} files")
        
        print(f"  ------------------\n  TOTAL: {split_total} files")
        grand_total += split_total
    
    print("\n" + "="*40)
    print(f"🎉 GRAND TOTAL IN ALL SPLITS: {grand_total} files")
    print("="*40)


def main():
    """Main function to configure and run the dataset splitting process."""
    # --- 1. DEFINE YOUR PATHS HERE ---
    # Path to your original dataset containing 'train' and 'test' folders
    original_dataset = '/content/dataset/chest_xray' 
    # Path where the new, split dataset will be created
    target_dataset = '/content/dataset/splited_chest_xray'
    
    # --- 2. DEFINE YOUR SPLIT RATIOS HERE ---
    # Ratios must sum to 1.0
    split_ratios = {
        'train': 0.10,      # 10%
        'validation': 0.10, # 10%
        'test': 0.80        # 80%
    }
    
    # --- 3. DEFINE YOUR RANDOM SEED ---
    random_seed = 42
    
    # --- SCRIPT EXECUTION ---
    print("--- Starting Dataset Re-splitting ---")
    print(f"➡️  Original Location: {original_dataset}")
    print(f"⬅️  Target Location: {target_dataset}")
    print(f"📊 Desired Split -> Train: {split_ratios['train']:.0%}, Validation: {split_ratios['validation']:.0%}, Test: {split_ratios['test']:.0%}")
    
    # Safety check to ensure ratios are correct
    if not sum(split_ratios.values()) == 1.0:
        logger.error(f"Split ratios must sum to 1.0, but they sum to {sum(split_ratios.values())}")
        return
        
    # Optional: Remove the target directory if it exists for a clean run
    if os.path.exists(target_dataset):
        print(f"⚠️ Target directory '{target_dataset}' already exists. Removing it now.")
        shutil.rmtree(target_dataset)
        
    # Run the main setup function
    success = setup_dataset(original_dataset, target_dataset, split_ratios, random_seed)
    
    if success:
        # Print the final counts to verify the split
        count_files_in_dataset(target_dataset)
    else:
        print("\n--- ❌ Dataset Splitting Failed ---")
        print("Please check the log messages above for specific errors.")

if __name__ == "__main__":
    main()
