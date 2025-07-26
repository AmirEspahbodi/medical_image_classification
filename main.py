import os
import random
import torch
import sys
import argparse

import hydra
import numpy as np
from omegaconf import OmegaConf, ListConfig

from src.utils.func import *
from trainA import train as train_a
from trainB import train as train_b
from trainC import train as train_c, evaluate
from src.utils.metrics import Estimator
from data.builder import generate_dataset
from src.builder import generate_model, load_weights
from src.coatnet_models import CoAtNetSideViTClassifier_1, CoAtNetSideViTClassifier_2, CoAtNetSideViTClassifier_3, CoAtNetSideViTClassifier_3_reg, CoAtNetSideViTClassifier_4, CoAtNetSideViTClassifier_5
from src.resnet_models import ResNetSideViTClassifier_1, ResNetSideViTClassifier_2, ResnetSideViTClassifier_3


@hydra.main(config_path="configs", config_name="config")
def main(cfg, backbone_trainable_layers, vit1_feature_strame, vit2_feature_strame):
    # print configuration
    print_msg('LOADING CONFIG FILE')
    print(OmegaConf.to_yaml(cfg))

    # --- Start of new validation block ---
    # This block validates the command-line arguments after they've been loaded by Hydra.
    print_msg('VALIDATING RUNTIME ARGUMENTS')
    validation_passed = True

    # Validate backbone_trainable_layers (btl)
    # Ensure it's a list and all values are within the allowed range
    if not all(x in [1, 2, 3, 4] for x in backbone_trainable_layers):
        print(f"ERROR: Invalid 'backbone_trainable_layers': {backbone_trainable_layers}. All values must be in [1, 2, 3, 4].")
        validation_passed = False
    else:
        print(f"  - OK: Backbone trainable layers set to: {backbone_trainable_layers}")

    # Validate vit1_feature_strame (v1fs)
    # Ensure it's a list of two numbers, each between 1 and 4
    if len(vit1_feature_strame) != 2 or not all(1 <= x <= 4 for x in vit1_feature_strame):
        print(f"ERROR: Invalid 'vit1_feature_strame': {vit1_feature_strame}. Must be two numbers, each between 1 and 4.")
        validation_passed = False
    else:
        print(f"  - OK: ViT1 Feature Strame set to: {vit1_feature_strame}")

    # Validate vit2_feature_strame (v2fs)

    if len(vit2_feature_strame) != 2 or not all(1 <= x <= 4 for x in vit2_feature_strame):
        print(f"ERROR: Invalid 'vit2_feature_strame': {vit2_feature_strame}. Must be two numbers, each between 1 and 4.")
        validation_passed = False
    else:
        print(f"  - OK: ViT2 Feature Strame set to: {vit2_feature_strame}")

    if not validation_passed:
        print_msg("Argument validation failed. Exiting.", warning=True)
        sys.exit(1)
    # --- End of new validation block ---
    
    print(backbone_trainable_layers)
    print(vit1_feature_strame)
    print(vit2_feature_strame)
    print(type(backbone_trainable_layers))
    print(type(vit1_feature_strame))
    print(type(vit2_feature_strame))


    # create folder
    save_path = cfg.dataset.save_path
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            print_msg(f'Save path {save_path} exists and will be overwritten.', warning=True)
        else:
            new_save_path = add_path_suffix(save_path)
            cfg.dataset.save_path = new_save_path
            warning = f'Save path {save_path} exists. New save path is set to {new_save_path}.'
            print_msg(warning, warning=True)

    os.makedirs(cfg.dataset.save_path, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.dataset.save_path, 'cfg.yaml'))

    # check preloading
    if cfg.dataset.preload_path:
        print(f"cfg.dataset.preload_path = {cfg.dataset.preload_path}")
        assert os.path.exists(cfg.dataset.preload_path), 'Preload path does not exist.'
        print_msg(f'Preloading is enabled using {cfg.dataset.preload_path}')

    if cfg.base.random_seed >= 0:
        set_seed(cfg.base.random_seed, cfg.base.cudnn_deterministic)

    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    frozen_encoder, side_vit_model1 = generate_model(cfg)
    frozen_encoder2, side_vit_model2 = generate_model(cfg)
    del frozen_encoder2

    print(f"type cfg = {type(cfg)}")
    if cfg.network.model in ["coatnet_3", "coatnet_3_reg", "coatnet_4", "coatnet_5"]:
        match cfg.network.model:
            case "coatnet_3":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_3
            case "coatnet_3_reg":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_3_reg
            case "coatnet_4":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_4
            case "coatnet_5":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_5
            case _:
                raise RuntimeError()
        classifier_with_side_vits = EnhancedSideViTClassifier(
            side_vit1=side_vit_model1,
            side_vit2=side_vit_model2,
            cfg=cfg,
        ).to(cfg.base.device)
    else:
        frozen_encoder3, side_vit_model_cnn = generate_model(cfg,use_cnn=True)
        del frozen_encoder3

        match cfg.network.model:
            case "coatnet_1":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_1
            case "coatnet_2":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_2
            case "resnet_1":
                EnhancedSideViTClassifier = ResNetSideViTClassifier_1
            case "resnet_2":
                EnhancedSideViTClassifier = ResNetSideViTClassifier_2
            case "resnet_3":
                EnhancedSideViTClassifier = ResnetSideViTClassifier_3
            case _:
                raise RuntimeError()
        classifier_with_side_vits = EnhancedSideViTClassifier(
            side_vit1=side_vit_model1,
            side_vit2=side_vit_model2,
            side_vit_cnn=side_vit_model_cnn,
            cfg=cfg,
        ).to(cfg.base.device)

    estimator = Estimator(cfg.train.metrics, cfg.dataset.num_classes, cfg.train.criterion)
    train_pipeline = None
    if cfg.base.training_plan=="A":
        train_pipeline = train_a
    elif cfg.base.training_plan=="B":
        train_pipeline = train_b
    elif cfg.base.training_plan=="C":
        train_pipeline = train_c
    else:
        raise RuntimeError()
    train_pipeline(
        cfg=cfg,
        frozen_encoder=frozen_encoder,
        model=classifier_with_side_vits,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
    )

    print('This is the performance of the final model:')
    checkpoint = os.path.join(cfg.dataset.save_path, 'final_weights.pt')
    load_weights(classifier_with_side_vits, checkpoint)
    evaluate(cfg, frozen_encoder, classifier_with_side_vits, test_dataset, estimator)

    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(cfg.dataset.save_path, 'best_validation_weights.pt')
    load_weights(classifier_with_side_vits, checkpoint)
    evaluate(cfg, frozen_encoder, classifier_with_side_vits, test_dataset, estimator)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training with custom arguments for Hydra.")
    parser.add_argument('-btl', '--backbone_trainable_layers', type=int, nargs='+',
                        help='List of backbone layers to train (e.g., 1 2 3 4).')
    parser.add_argument('-v1fs', '--vit1_feature_strame', type=int, nargs=2,
                        help='Two numbers for ViT1 feature stride, each between 1 and 4 (e.g., 1 4).')
    parser.add_argument('-v2fs', '--vit2_feature_strame', type=int, nargs=2,
                        help='Two numbers for ViT2 feature stride, each between 1 and 4 (e.g., 2 3).')
    args, unknown_args = parser.parse_known_args()
    print(args, unknown_args)

    main(args.btl, args.v1fs, args.v2fs)
