import os
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
# ✨ Added SWA and standard schedulers
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import OneCycleLR

from src.utils.func import *
from src.loss import *
# Removed custom scheduler import as we now use torch.optim.lr_scheduler
# from src.scheduler import *

def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    """
    Refactored training loop implementing Plan B:
    1. Main Training Phase: Uses AdamW, OneCycleLR, and Early Stopping.
    2. SWA Phase: Averages model weights for the final epochs to improve generalization.
    """
    device = cfg.base.device
    
    # ✨ Setup for Differential Learning Rates
    # Assumes your model has a 'cnn_backbone' attribute. Adjust if named differently.
    backbone_params = [p for n, p in model.named_parameters() if 'cnn_backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'cnn_backbone' not in n and p.requires_grad]
    
    param_groups = [
        {'params': backbone_params, 'lr': cfg.solver.backbone_lr}, # e.g., 1e-5
        {'params': head_params, 'lr': cfg.solver.head_lr}         # e.g., 1e-4
    ]

    optimizer = initialize_optimizer(cfg, param_groups)
    # ✨ Loss function now supports label smoothing from config
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

    # ✨ Schedulers: OneCycleLR for main training, SWALR for SWA phase
    main_scheduler = OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in param_groups],
        total_steps=len(train_loader) * cfg.train.epochs,
        pct_start=0.1, # Warmup for 10% of steps
        anneal_strategy='cos'
    )

    # ✨ SWA Setup
    swa_start_epoch = cfg.train.swa_start_epoch # e.g., start SWA at 75% of total epochs
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg.solver.swa_lr) # e.g., 1e-3

    # ✨ Early Stopping Setup
    max_indicator = 0
    epochs_no_improve = 0
    patience = cfg.train.early_stopping_patience # e.g., 10 epochs

    # --- Training Initialization ---
    start_epoch = 0
    if cfg.base.checkpoint:
        # Note: Resuming SWA requires more complex state saving. 
        # This implementation resumes standard training.
        start_epoch = resume(cfg, model, optimizer)

    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        # --- Update Dynamic Loss Weights (if any) ---
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        epoch_loss = 0
        estimator.reset()
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.train.epochs}', leave=False) if cfg.base.progress else train_loader
        
        for step, train_data in enumerate(progress):
            # --- Data Preparation ---
            if cfg.dataset.preload_path:
                X_side, key_states, value_states, y = train_data
                key_states, value_states = key_states.to(device), value_states.to(device)
                key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
            else:
                X_lpm, X_side, y = train_data
                X_lpm = X_lpm.to(device)
                with torch.no_grad():
                    _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

            X_side, y = X_side.to(device), y.to(device)
            y = select_target_type(y, cfg.train.criterion)

            # --- Forward & Backward Pass ---
            y_pred = model(X_side, key_states, value_states)
            loss = loss_function(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # ✨ Scheduler step (OneCycleLR steps per batch)
            if epoch < swa_start_epoch:
                main_scheduler.step()

            # --- Logging ---
            epoch_loss += loss.item()
            estimator.update(y_pred, y)
            if cfg.base.progress:
                progress.set_postfix(Loss=f'{epoch_loss/(step+1):.4f}', LR=f"{optimizer.param_groups[0]['lr']:.2e}")

        # --- SWA Scheduler Step (steps per epoch) ---
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # --- Validation & Early Stopping ---
        # Evaluate using the standard model during main phase, and SWA model during SWA phase
        eval_model = swa_model if epoch >= swa_start_epoch else model
        eval(cfg, frozen_encoder, eval_model, val_loader, estimator, device)
        val_scores = estimator.get_scores(6)
        
        print(f"\nEpoch {epoch+1} Validation Metrics: {val_scores}")
        
        # --- Model Checkpointing & Saving Best Model ---
        indicator = val_scores[cfg.train.indicator]
        if indicator > max_indicator:
            max_indicator = indicator
            epochs_no_improve = 0
            # Always save the best single model found before or during SWA
            save_weights(cfg, model, 'best_validation_weights.pt')
            print(f"🚀 New best model saved with {cfg.train.indicator}: {max_indicator:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"--- Early stopping triggered after {patience} epochs with no improvement. ---")
            break

    # --- Finalize and Save SWA Model ---
    print("\n--- Training finished. Finalizing SWA model. ---")
    # Update SWA batch norm stats
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    # Save the final SWA model
    save_weights(cfg, swa_model, 'swa_model_final_weights.pt')

    # Return the SWA model for final evaluation
    return swa_model

###
### Helper Function Modifications
###

def initialize_optimizer(cfg, params): # Now accepts params directly
    solver = cfg.solver.optimizer
    if solver == 'SGD':
        optimizer = torch.optim.SGD(params, lr=cfg.solver.head_lr, momentum=cfg.solver.momentum, weight_decay=cfg.solver.weight_decay)
    elif solver == 'ADAM':
        optimizer = torch.optim.Adam(params, lr=cfg.solver.head_lr, betas=cfg.solver.betas, weight_decay=cfg.solver.weight_decay)
    elif solver == 'ADAMW':
        optimizer = torch.optim.AdamW(params, lr=cfg.solver.head_lr, betas=cfg.solver.betas, weight_decay=cfg.solver.weight_decay)
    else:
        raise NotImplementedError('Not implemented optimizer.')
    return optimizer


def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion
    weight = None
    loss_weight_scheduler = None
    loss_weight = cfg.train.loss_weight
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.train.loss_weight_decay_rate)
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
        loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.train.label_smoothing)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss()
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss()
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss()
    elif criterion == 'kappa_loss':
        loss = KappaLoss()
    elif criterion == 'focal_loss':
        loss = FocalLoss()
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


def adjust_learning_rate(cfg, optimizer, epoch):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.dataset.learning_rate * epoch / cfg.train.warmup_epochs
    else:
        lr = cfg.dataset.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - cfg.train.warmup_epochs) / (cfg.train.epochs - cfg.train.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(cfg, model, epoch, optimizer, save_name):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    checkpoint_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(checkpoint, checkpoint_path)


def save_weights(cfg, model, save_name):
    save_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(model.state_dict(), save_path)
    print_msg('Model saved at {}'.format(save_path))


def resume(cfg, model, optimizer):
    checkpoint = cfg.base.checkpoint
    if os.path.exists(checkpoint):
        print_msg('Loading checkpoint {}'.format(checkpoint))

        checkpoint = torch.load(checkpoint, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print_msg('Loaded checkpoint {} from epoch {}'.format(checkpoint, checkpoint['epoch']))
        return start_epoch
    else:
        print_msg('No checkpoint found at {}'.format(checkpoint))
        raise FileNotFoundError('No checkpoint found at {}'.format(checkpoint))
