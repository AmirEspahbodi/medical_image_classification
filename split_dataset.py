import os
import shutil
import random
import argparse
from pathlib import Path
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def apply_medical_augmentation(image):
    """
    Apply 1 to 4 random augmentation techniques sequentially on the input image.
    Techniques include: rotate, scale, contrast adjustment, and Poisson noise.
    """
    techniques = ['rotate', 'scale', 'contrast', 'noise']
    # Choose between 1 and 4 techniques (allowing repeats)
    selected = random.choices(techniques, k=random.randint(1, 4))

    augmented = image
    for tech in selected:
        if tech == 'rotate':
            angle = random.uniform(-10, 10)
            rotated = augmented.rotate(angle, resample=Image.BICUBIC, expand=True)
            # Fit back to original size
            augmented = ImageOps.fit(rotated, image.size, method=Image.BICUBIC, centering=(0.5, 0.5))

        elif tech == 'scale':
            scale_factor = random.uniform(0.9, 1.1)
            w, h = augmented.size
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            scaled = augmented.resize((new_w, new_h), resample=Image.BICUBIC)
            # Pad or crop back to original
            pad_w = max(0, w - new_w)
            pad_h = max(0, h - new_h)
            padded = ImageOps.expand(
                scaled,
                border=(pad_w // 2, pad_h // 2),
                fill=None
            )
            augmented = padded.crop((0, 0, w, h))

        elif tech == 'contrast':
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(factor)

        elif tech == 'noise':
            arr = np.array(augmented).astype(np.float32) / 255.0
            noisy = np.random.poisson(arr * 255.0) / 255.0
            noisy_img = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
            augmented = Image.fromarray(noisy_img)

        else:
            # Should not happen
            raise RuntimeError(f"Unknown augmentation: {tech}")

    return augmented



def create_directory_structure(destination_root, classes):
    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            path = os.path.join(destination_root, split, class_name)
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")


def copy_files_between_splits(source_root, split_from, split_to, classes, random_seed=42):
    random.seed(random_seed)
    total = 0
    for class_name in classes:
        src_dir = os.path.join(source_root, split_from, class_name)
        tgt_dir = os.path.join(source_root, split_to, class_name)
        os.makedirs(tgt_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        for f in files:
            try:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(tgt_dir, f))
                total += 1
            except Exception as e:
                logger.error(f"Failed to copy {f}: {e}")
    return total


def setup_dataset(original_path, target_path, augment_flag):
    try:
        classes = os.listdir(os.path.join(original_path, 'train'))
        logger.info(f"Found classes: {classes}")
    except Exception as e:
        logger.error(f"Unable to read classes: {e}")
        return False

    create_directory_structure(target_path, classes)

    for split in ['train', 'test']:
        for class_name in classes:
            src = os.path.join(original_path, split, class_name)
            dst = os.path.join(target_path, split, class_name)
            os.makedirs(dst, exist_ok=True)
            files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
            logger.info(f"Copying {'and augmenting' if split=='train' and augment_flag == 1 else ''}{len(files)} from {src} to {dst}")
            for f in files:
                src_file = os.path.join(src, f)
                dst_file = os.path.join(dst, f)
                shutil.copy2(src_file, dst_file)

                if split == 'train' and augment_flag == 1:
                    if class_name == 'PNEUMONIA' and random.random() >= 0.5:
                        continue
                    try:
                        with Image.open(src_file).convert('L') as img_gray:
                            # work in grayscale to preserve medical detail
                            aug = apply_medical_augmentation(img_gray)
                            name, ext = os.path.splitext(f)
                            aug.save(os.path.join(dst, f"{name}_aug{ext}"))
                    except Exception as e:
                        logger.error(f"Augmentation failed for {f}: {e}")

    # Copy entire test set to validation
    copied = copy_files_between_splits(target_path, 'test', 'validation', classes)
    logger.info(f"Copied {copied} files from test to validation")

    return True


def count_files_in_dataset(path):
    base = os.path.join(path, 'train')
    if not os.path.exists(base):
        print(f"Error: 'train' not found in {path}")
        return
    classes = os.listdir(base)
    for split in ['train', 'validation', 'test']:
        total = 0
        print(f"\n{split.upper()}:")
        for c in classes:
            d = os.path.join(path, split, c)
            cnt = len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]) if os.path.exists(d) else 0
            print(f"  {c}: {cnt}")
            total += cnt
        print(f"  TOTAL: {total}")


def main():
    parser = argparse.ArgumentParser("Chest X-ray dataset setup (optional augmentation)")
    parser.add_argument('--augment', type=int, choices=[0,1], required=True,
                        help='0 = no augmentation, 1 = apply augmentation')
    args = parser.parse_args()

    original_dataset = '/content/dataset/chest_xray'
    target_dataset = '/content/dataset/splited_chest_xray'

    if os.path.exists(target_dataset):
        shutil.rmtree(target_dataset)
        logger.info(f"Removed existing: {target_dataset}")

    success = setup_dataset(original_dataset, target_dataset, args.augment)
    if success:
        print("\nDataset setup complete. File counts:")
        count_files_in_dataset(target_dataset)
    else:
        print("Dataset setup failed. See logs.")

if __name__ == '__main__':
    main()

