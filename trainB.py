import os
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import OneCycleLR

# Assuming these are your custom utility/loss imports
# from src.utils.func import *
# from src.loss import *

# --- 1. Main Training Function (Plan B with Plotting) ---
def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    """
    Refactored training loop implementing Plan B with performance plotting:
    1. Main Training Phase: Uses AdamW, OneCycleLR, and Early Stopping.
    2. SWA Phase: Averages model weights for the final epochs.
    3. Plotting: Tracks and saves loss/accuracy graphs at the end.
    """
    device = cfg.base.device
    
    # Differential Learning Rates Setup
    backbone_params = [p for n, p in model.named_parameters() if 'cnn_backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'cnn_backbone' not in n and p.requires_grad]
    param_groups = [
        {'params': backbone_params, 'lr': cfg.solver.backbone_lr},
        {'params': head_params, 'lr': cfg.solver.head_lr}
    ]

    optimizer = initialize_optimizer(cfg, param_groups)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

    # Schedulers: OneCycleLR for main training
    main_scheduler = OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in param_groups],
        total_steps=len(train_loader) * cfg.train.epochs,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # SWA Setup
    swa_start_epoch = cfg.train.swa_start_epoch
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg.solver.swa_lr)

    # Early Stopping & History Tracking
    max_indicator = 0
    epochs_no_improve = 0
    patience = cfg.train.early_stopping_patience
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_epoch = 0
    if cfg.base.checkpoint:
        start_epoch = resume(cfg, model, optimizer)

    print("--- Starting Training with Plan B (SWA + Plotting) ---")
    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        # --- Training Phase ---
        epoch_loss = 0
        # ✨ FIX: Reset estimator state before starting training for the epoch
        estimator.reset()
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.train.epochs}', leave=False) if cfg.base.progress else train_loader
        
        for step, train_data in enumerate(progress):
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

            y_pred = model(X_side, key_states, value_states)
            loss = loss_function(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch < swa_start_epoch:
                main_scheduler.step()

            epoch_loss += loss.item()
            estimator.update(y_pred, y)
            if cfg.base.progress:
                progress.set_postfix(Loss=f'{epoch_loss/(step+1):.4f}', LR=f"{optimizer.param_groups[0]['lr']:.2e}")
        
        # Record training metrics
        avg_train_loss = epoch_loss / len(train_loader)
        train_scores = estimator.get_scores(4)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_scores.get(cfg.train.indicator, 0))

        # --- SWA Update Step ---
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # --- Validation Phase ---
        # ✨ FIX: Reset estimator state again before validation to ensure a clean calculation
        estimator.reset()
        eval_model = swa_model if epoch >= swa_start_epoch else model
        val_loss, val_scores = eval(cfg, frozen_encoder, eval_model, val_loader, estimator, device, loss_function)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_scores.get(cfg.train.indicator, 0))

        print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_scores.get(cfg.train.indicator, 0):.4f}")
        print(f"Epoch {epoch+1} Val Loss:   {val_loss:.4f}, Val Acc:   {val_scores.get(cfg.train.indicator, 0):.4f}")
        
        indicator = val_scores[cfg.train.indicator]
        if indicator > max_indicator:
            max_indicator = indicator
            epochs_no_improve = 0
            # Always save the best single model, not the SWA model during training
            save_weights(cfg, model, 'best_validation_weights.pt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"--- Early stopping triggered after {patience} epochs with no improvement. ---")
            break

    # Finalize SWA Model
    print("\n--- Training finished. Updating SWA model batch norm statistics. ---")
    swa_model.train()
    with torch.no_grad():
        for i, train_data in enumerate(train_loader):
            if i >= 100: break
            if cfg.dataset.preload_path:
                X_side, key_states, value_states, y = train_data
                key_states, value_states = key_states.to(device), value_states.to(device)
                key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
            else:
                X_lpm, X_side, y = train_data
                X_lpm = X_lpm.to(device)
                _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)
            X_side = X_side.to(device)
            swa_model(X_side, key_states, value_states)
    
    save_weights(cfg, swa_model, 'swa_model_final_weights.pt')
    
    # Save plots
    save_plots(history, cfg.dataset.save_path)
    
    return swa_model

# --- 2. Helper Function Modifications ---

def initialize_optimizer(cfg, params):
    solver = cfg.solver.optimizer
    if solver == 'ADAMW':
        optimizer = torch.optim.AdamW(params, lr=cfg.solver.head_lr, betas=cfg.solver.betas, weight_decay=cfg.solver.weight_decay)
    else:
        raise NotImplementedError('Not implemented optimizer.')
    return optimizer

def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion
    weight = None
    loss_weight_scheduler = None
    if criterion == 'cross_entropy':
        loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.train.label_smoothing)
    else:
        raise NotImplementedError('Not implemented loss function.')
    return loss, loss_weight_scheduler

def save_plots(history, save_path):
    """Saves plots for training/validation loss and accuracy."""
    print("--- Generating and saving performance plots... ---")
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    plt.close()
    print(f"Performance plots saved to {save_path}")

# --- 3. Unchanged & Modified Helper Functions ---

def initialize_dataloader(cfg, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    return train_loader, val_loader

def eval(cfg, frozen_encoder, model, dataloader, estimator, device, loss_function):
    """
    Evaluates the model on a given dataloader.
    ✨ FIX: This function no longer resets the estimator. The caller is responsible
    for managing the estimator's state.
    """
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    
    for test_data in dataloader:
        if cfg.dataset.preload_path:
            X_side, key_states, value_states, y = test_data
            key_states, value_states = key_states.to(device), value_states.to(device)
            key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
        else:
            X_lpm, X_side, y = test_data
            X_lpm = X_lpm.to(device)
            with torch.no_grad():
                _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)
        
        X_side, y = X_side.to(device), y.to(device)
        y_true = select_target_type(y, cfg.train.criterion)
        y_pred = model(X_side, key_states, value_states)
        
        loss = loss_function(y_pred, y_true)
        total_loss += loss.item()
        estimator.update(y_pred, y_true)
        
    model.train()
    torch.set_grad_enabled(True)
    avg_loss = total_loss / len(dataloader)
    scores = estimator.get_scores(6)
    return avg_loss, scores

def save_weights(cfg, model, save_name):
    save_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(model.state_dict(), save_path)
    print(f'Model weights saved at: {save_path}')

def select_target_type(y, criterion): return y
