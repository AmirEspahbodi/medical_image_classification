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
from src.coatnet_models import CoAtNetSideViTClassifier_1, CoAtNetSideViTClassifier_2, CoAtNetSideViTClassifier_3, CoAtNetSideViTClassifier_4, CoAtNetSideViTClassifier_5
from src.resnet_models import ResNetSideViTClassifier_1, ResNetSideViTClassifier_2, ResnetSideViTClassifier_3


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
    frozen_encoder2, side_vit_model2 = generate_model(cfg)
    del frozen_encoder2

    print(f"type cfg = {type(cfg)}")
    if cfg.network.model in ["coatnet_3"]:
        match cfg.network.model:
            case "coatnet_3":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_3
            case _:
                raise RuntimeError()
        classifier_with_side_vits = EnhancedSideViTClassifier(
            side_vit1=side_vit_model1,
            side_vit2=side_vit_model2,
            cfg=cfg,
        ).to(cfg.base.device)
    elif cfg.network.model in ["coatnet_4", "coatnet_5"]:
        frozen_encoder3, side_vit_model_3 = generate_model(cfg)
        del frozen_encoder3

        match cfg.network.model:
            case "coatnet_4":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_4
            case "coatnet_5":
                EnhancedSideViTClassifier = CoAtNetSideViTClassifier_5
            case _:
                raise RuntimeError()
        classifier_with_side_vits = EnhancedSideViTClassifier(
            side_vit1=side_vit_model1,
            side_vit2=side_vit_model2,
            side_vit3=side_vit_model_3,
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
    train(
        cfg=cfg,
        frozen_encoder=frozen_encoder,
        model=classifier_with_side_vits,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator
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
    main()
