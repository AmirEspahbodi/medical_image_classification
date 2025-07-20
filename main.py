import os
import random
import torch

import hydra
import numpy as np
from omegaconf import OmegaConf

from src.utils.func import *
from train import train, evaluate
from src.utils.metrics import Estimator
from data.builder import generate_dataset
from src.builder import generate_model, load_weights
from src.side_resnet_vit import (
    ResNetSideViTClassifier_MLP,
    ResNetSideViTClassifier_FC,
    ResNetSideViTClassifier_FFN_MLP,
    ResNetSideViTClassifier_FFN_FC,
)


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    # print configuration
    print_msg('LOADING CONFIG FILE')
    print(OmegaConf.to_yaml(cfg))

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
    _, side_vit_model2 = generate_model(cfg)

    print(f"type cfg = {type(cfg)}")
    match cfg.network.model:
        case "resnet_sidevit_ffn_fc":
            ResNetSideViTClassifier = ResNetSideViTClassifier_FFN_FC
        case "resnet_sidevit_ffn_mlp":
            ResNetSideViTClassifier = ResNetSideViTClassifier_FFN_MLP
        case "resnet_sidevit_fc":
            ResNetSideViTClassifier = ResNetSideViTClassifier_FC
        case "resnet_sidevit_mlp":
            ResNetSideViTClassifier = ResNetSideViTClassifier_ML_
