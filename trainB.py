import os
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# Assuming these are your custom utility/loss imports
from src.utils.func import *
from src.loss import *
from src.scheduler import *

# --- 1. Sharpness-Aware Minimization (SAM) Optimizer Implementation ---
# This is a standard, self-contained implementation of SAM.
class SAM(torch.optim.Optimizer):
    """
    Implements the Sharpness-Aware Minimization (SAM) optimizer.
    SAM seeks parameters in flat loss regions for better generalization.
    It wraps a base optimizer (e.g., AdamW) and performs a two-step update.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # Initialize the base optimizer (e.g., AdamW) with the model parameters
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first step of SAM: ascends the loss landscape to find a point
        with high sharpness and stores the original parameter values.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                # Calculate the perturbation (e_w)
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Performs the second step: moves back to the original parameters and then
        takes a gradient step using the gradients from the perturbed point.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # Restore original parameters before the base optimizer step
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        """Calculates the gradient norm."""
        # A small trick to put all norms on the same device
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2
        )
        return norm

    # Forward other methods to the base optimizer
    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't support closure for step.")

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# --- 2. Main Training Function (Plan A) ---
def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    """
    Main training loop implementing Plan A:
    - Optimizer: SAM (Sharpness-Aware Minimization) wrapping AdamW.
    - Scheduler: Cosine Annealing with Linear Warmup.
    - Regularization: Label Smoothing and Differential Learning Rates.
    """
    device = cfg.base.device
    
    # Setup for Differential Learning Rates
    backbone_params = [p for n, p in model.named_parameters() if 'cnn_backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'cnn_backbone' not in n and p.requires_grad]
    param_groups = [
        {'params': backbone_params, 'lr': cfg.solver.backbone_lr},
        {'params': head_params, 'lr': cfg.solver.head_lr}
    ]

    optimizer = initialize_optimizer(cfg, param_groups)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)
    
    start_epoch = 0
    if cfg.base.checkpoint:
        start_epoch = resume(cfg, model, optimizer)
    
    sam_start_epoch = cfg.train.sam_start_epoch
    max_indicator = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("--- Starting Training with Plan A (SAM Optimizer) ---")
    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        lr = adjust_learning_rate(cfg, optimizer, epoch)
        
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        epoch_loss = 0
        estimator.reset()
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.train.epochs}', leave=False) if cfg.base.progress else train_loader
        
        for step, train_data in enumerate(progress):
            # --- Data Preparation (matches your original code) ---
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

            # --- Forward & Backward Pass with SAM logic ---
            y_pred = model(X_side, key_states, value_states)
            loss = loss_function(y_pred, y)

            if epoch >= sam_start_epoch:
                # Enable gradient calculation for the first step
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second forward/backward pass on the perturbed weights
                loss_function(model(X_side, key_states, value_states), y).backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard optimizer step before SAM is active
                optimizer.zero_grad()
                loss.backward()
                optimizer.base_optimizer.step()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            epoch_loss += loss.item()
            estimator.update(y_pred, y)
            if cfg.base.progress:
                progress.set_postfix(Loss=f'{epoch_loss/(step+1):.4f}', LR=f"{lr:.2e}")
        
        # --- Validation & Saving Best Model ---
        eval(cfg, frozen_encoder, model, val_loader, estimator, device)
        val_scores = estimator.get_scores(6)

        avg_train_loss = epoch_loss / len(train_loader)
        train_scores = estimator.get_scores(4)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_scores.get(cfg.train.indicator, 0)) # Use the main indicator metric
        
        val_loss, val_scores = eval(cfg, frozen_encoder, model, val_loader, estimator, device, loss_function)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_scores.get(cfg.train.indicator, 0))
        
        print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_scores.get(cfg.train.indicator, 0):.4f}")
        print(f"Epoch {epoch+1} Val Loss:   {val_loss:.4f}, Val Acc:   {val_scores.get(cfg.train.indicator, 0):.4f}")

        indicator = val_scores[cfg.train.indicator]
        if indicator > max_indicator:
            max_indicator = indicator
            save_weights(cfg, model, 'best_validation_weights.pt')

    save_plots(history, cfg.dataset.save_path)
    
    save_weights(cfg, model, 'final_weights.pt')
    print("--- Training finished. Final model saved. ---")
    return model

# ✨ New plotting function
def save_plots(history, save_path):
    """Saves plots for training/validation loss and accuracy."""
    print("--- Generating and saving performance plots... ---")
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

    # Plot 2: Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(save_path, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Accuracy plot saved to {acc_plot_path}")

# --- 3. Helper Function Modifications for Plan A ---

def initialize_optimizer(cfg, params):
    """Initializes the SAM optimizer wrapping a base optimizer like AdamW."""
    base_optimizer_choice = torch.optim.AdamW
    print(f"--- Initializing SAM with base optimizer: {base_optimizer_choice.__name__} ---")
    
    optimizer = SAM(
        params, 
        base_optimizer_choice, 
        rho=cfg.solver.rho, 
        adaptive=True, 
        betas=cfg.solver.betas, 
        weight_decay=cfg.solver.weight_decay
    )
    return optimizer

def initialize_loss(cfg, train_dataset):
    """Initializes the loss function, adding label smoothing from config."""
    criterion = cfg.train.criterion
    weight = None
    loss_weight_scheduler = None # Assuming this is your custom scheduler
    loss_weight = cfg.train.loss_weight
    
    if criterion == 'cross_entropy':
        print(f"--- Using CrossEntropyLoss with label smoothing: {cfg.train.label_smoothing} ---")
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
    
    # Assuming WarpedLoss is a custom wrapper you have
    # loss_function = WarpedLoss(loss, criterion)
    loss_function = loss
    return loss_function, loss_weight_scheduler

def adjust_learning_rate(cfg, optimizer, epoch):
    """
    Handles a linear warmup phase followed by a cosine decay schedule.
    This is the recommended scheduler for Plan A.
    """
    main_lr = cfg.solver.head_lr
    warmup_epochs = cfg.train.warmup_epochs
    total_epochs = cfg.train.epochs

    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
    else:
        # Cosine decay phase
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr_scale = 0.5 * (1. + math.cos(math.pi * progress))

    # Apply the scaled learning rate to each parameter group
    for i, param_group in enumerate(optimizer.param_groups):
        base_lr = cfg.solver.head_lr if i == 1 else cfg.solver.backbone_lr
        param_group['lr'] = base_lr * lr_scale
        
    return optimizer.param_groups[1]['lr'] # Return head lr for logging

# --- 4. Placeholder/Unchanged Functions ---
# These functions are required for the script to be complete but are
# assumed to be defined correctly elsewhere in your project.

def initialize_dataloader(cfg, train_dataset, val_dataset):
    # This function remains unchanged from your original code
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    return train_loader, val_loader

# ✨ Modified eval function to return loss
def eval(cfg, frozen_encoder, model, dataloader, estimator, device, loss_function):
    model.eval()
    torch.set_grad_enabled(False)
    
    estimator.reset()
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
        
        # Calculate loss for the batch
        loss = loss_function(y_pred, y_true)
        total_loss += loss.item()
        
        estimator.update(y_pred, y_true)
        
    model.train()
    torch.set_grad_enabled(True)
    
    avg_loss = total_loss / len(dataloader)
    scores = estimator.get_scores(6)
    
    return avg_loss, scores

def save_weights(cfg, model, save_name):
    # This function remains unchanged
    save_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(model.state_dict(), save_path)
    print(f'Model weights saved at: {save_path}')
