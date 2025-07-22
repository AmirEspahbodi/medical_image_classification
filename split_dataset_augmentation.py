import os
import shutil
import random
from pathlib import Path
import logging
import numpy as np
from PIL import Image, ImageEnhance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NEW AUGMENTATION FUNCTION ---
def apply_augmentation(image):
    """
    Applies a randomly selected augmentation to a PIL image.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The augmented image.
    """
    # List of augmentation techniques
    techniques = ['rotate', 'scale', 'contrast'] #, 'noise'
    chosen_technique = random.choice(techniques)

    if chosen_technique == 'rotate':
        # Rotation (e.g. random ± 15°)
        angle = random.uniform(-15, 15)
        return image.rotate(angle, resample=Image.BICUBIC, expand=True)

    elif chosen_technique == 'scale':
        # Scaling/zooming (e.g. random zoom in/out by ~20%)
        scale_factor = random.uniform(0.8, 1.2)
        original_size = image.size
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        
        # Resize and then crop/paste to maintain original dimensions
        scaled_image = image.resize(new_size, resample=Image.BICUBIC)
        
        if scale_factor > 1.0: # Zoom in, crop center
            left = (new_size[0] - original_size[0]) / 2
            top = (new_size[1] - original_size[1]) / 2
            right = (new_size[0] + original_size[0]) / 2
            bottom = (new_size[1] + original_size[1]) / 2
            return scaled_image.crop((left, top, right, bottom))
        else: # Zoom out, paste on black background
            new_img = Image.new(image.mode, original_size, (0, 0, 0))
            paste_position = (int((original_size[0] - new_size[0]) / 2), int((original_size[1] - new_size[1]) / 2))
            new_img.paste(scaled_image, paste_position)
            return new_img

    elif chosen_technique == 'contrast':
        # Contrast adjustment
        factor = random.uniform(0.6, 1.4) # 1.0 is original contrast
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    elif chosen_technique == 'noise':
        # Gaussian noise
        img_array = np.array(image)
        # Add noise with a random standard deviation
        noise = np.random.normal(0, random.uniform(10, 25), img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    return image # Fallback

def create_directory_structure(destination_root, classes):
    """
    Create the necessary directory structure for the dataset.
    """
    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            folder_path = os.path.join(destination_root, split, class_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Created directory: {folder_path}")

def copy_files_between_splits(source_root, destination_root, source_split, target_split, ratio, classes, random_seed=42):
    """
    Copy a specific ratio of files from source_split to target_split.
    """
    random.seed(random_seed)
    total_copied = 0
    
    for class_name in classes:
        source_dir = os.path.join(source_root, source_split, class_name)
        target_dir = os.path.join(destination_root, target_split, class_name)
        
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
            num_files = len(files)
            
            if num_files == 0:
                logger.warning(f"No files found in {source_dir}")
                continue
                
            num_to_copy = int(num_files * ratio)
            files_to_copy = random.sample(files, num_to_copy)
            
            logger.info(f"Copying {num_to_copy} files from {source_dir} to {target_dir}")
            
            for file in files_to_copy:
                source_path = os.path.join(source_dir, file)
                dest_path = os.path.join(target_dir, file)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    total_copied += 1
                except Exception as e:
                    logger.error(f"Failed to copy {file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing directory {source_dir}: {str(e)}")
    
    return total_copied

def setup_dataset(original_path, target_path, val_from_test_ratio=1.0, random_seed=42):
    """
    Set up the dataset with train, test, and validation splits,
    and apply augmentation to the training set.
    """
    try:
        classes = os.listdir(os.path.join(original_path, 'train'))
        logger.info(f"Found classes: {classes}")
    except Exception as e:
        logger.error(f"Failed to get classes: {str(e)}")
        return False
    
    create_directory_structure(target_path, classes)
    
    # Copy files and apply augmentation where needed
    for split in ['train', 'test']:
        for class_name in classes:
            source_class_dir = os.path.join(original_path, split, class_name)
            target_class_dir = os.path.join(target_path, split, class_name)
            
            try:
                files = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]
                
                logger.info(f"Copying {len(files)} files from {source_class_dir} to {target_class_dir}")
                
                for file in files:
                    source_file = os.path.join(source_class_dir, file)
                    target_file = os.path.join(target_class_dir, file)
                    shutil.copy2(source_file, target_file)
                    
                    # --- DATA AUGMENTATION LOGIC ---
                    # If the image is for the training set, create and save an augmented version.
                    if split == 'train':
                        try:
                            if class_name=="PNEUMONIA":
                                if random.random()>0.5:
                                    pass
                            # Use .convert('RGB') to handle grayscale or other modes consistently
                            with Image.open(source_file).convert('RGB') as img:
                                augmented_img = apply_augmentation(img)
                                
                                # Define a new name for the augmented file
                                name, ext = os.path.splitext(file)
                                aug_filename = f"{name}_aug{ext}"
                                aug_filepath = os.path.join(target_class_dir, aug_filename)
                                
                                augmented_img.save(aug_filepath)
                        except Exception as e:
                            logger.error(f"Could not augment and save {file}: {e}")
                            
            except Exception as e:
                logger.error(f"Error copying files for {class_name} in {split}: {str(e)}")
    
    # Create validation set by copying files from the new test set
    copied_to_val = copy_files_between_splits(
        target_path, target_path, 
        'test', 'validation', 
        val_from_test_ratio, classes, random_seed
    )
    logger.info(f"Copied {copied_to_val} files from test to validation")
    
    return True

def count_files_in_dataset(dataset_path):
    """Print statistics about the dataset."""
    try:
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
    
    # Set ratio to 1.0 to copy ALL test files to validation
    val_from_test_ratio = 1.0 
    
    random_seed = 42
    
    print(f"Setting up dataset from {original_dataset} to {target_dataset}")
    # Remove target directory if it exists to ensure a fresh start
    if os.path.exists(target_dataset):
        shutil.rmtree(target_dataset)
        logger.info(f"Removed existing target directory: {target_dataset}")

    success = setup_dataset(original_dataset, target_dataset, val_from_test_ratio, random_seed)
    
    if success:
        print("\nDataset setup complete. File counts:")
        count_files_in_dataset(target_dataset)
    else:
        print("Dataset setup failed. Check logs for details.")

if __name__ == "__main__":
    main()
