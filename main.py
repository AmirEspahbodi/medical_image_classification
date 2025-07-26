import os
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# --- Import your existing utility functions ---
# Make sure these files are accessible in your project structure.
# from src.utils.func import *
# from src.loss import *
# from src.scheduler import *

# --- Helper Class: Early Stopping ---
# Stops training when validation loss stops improving and saves the best model.
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving best model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- Helper Function for Logging and Visualization ---
def log_and_visualize_results(cfg, history, class_names, all_preds, all_labels):
    """
    Generates and saves plots for loss, accuracy, and a confusion matrix.
    """
    print("--- Generating and saving result visualizations ---")
    output_dir = cfg.base.working_dir
    
    # --- 1. Plot Loss Curve ---
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved to {loss_curve_path}")

    # --- 2. Plot Accuracy Curve ---
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    accuracy_curve_path = os.path.join(output_dir, 'accuracy_curve.png')
    plt.savefig(accuracy_curve_path)
    plt.close()
    print(f"Accuracy curve saved to {accuracy_curve_path}")

    # --- 3. Generate and Plot Confusion Matrix from final validation results ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Validation Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

# --- REFACTORED Main Training Function ---
def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    """
    Refactored training loop with modern techniques to combat overfitting.
    """
    device = cfg.base.device
    num_classes = cfg.dataset.num_classes
    epochs = cfg.train.epochs
    
    # --- 1. Optimizer with Differential Learning Rates ---
    print("--- Setting up AdamW optimizer with differential learning rates ---")
    backbone_params = [p for n, p in model.named_parameters() if "cnn_backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "cnn_backbone" not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg.train.backbone_lr},
        {'params': head_params, 'lr': cfg.train.head_lr}
    ], weight_decay=cfg.train.weight_decay)

    # --- 2. Advanced LR Scheduler with Warmup ---
    warmup_epochs = cfg.train.warmup_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    # --- 3. Loss Function with Label Smoothing ---
    # This replaces the CrossEntropy part of your old `initialize_loss`
    loss_function = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    # --- 4. MixUp/CutMix Augmentation ---
    mixup_or_cutmix = transforms.RandomChoice([
        transforms.MixUp(num_classes=num_classes, alpha=0.2),
        transforms.CutMix(num_classes=num_classes, alpha=1.0)
    ])
    use_mixup_cutmix = cfg.train.use_mixup_cutmix

    # --- 5. Gradient Accumulation & Mixed Precision ---
    accumulation_steps = cfg.train.accumulation_steps
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # --- 6. Early Stopping & Checkpoint Handling ---
    checkpoint_path = os.path.join(cfg.base.working_dir, 'best_model.pt')
    early_stopper = EarlyStopping(patience=cfg.train.patience, verbose=True, path=checkpoint_path)
    
    start_epoch = 0
    if cfg.base.checkpoint:
        start_epoch = resume(cfg, model, optimizer) # Use your resume function

    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("--- Starting Advanced Training ---")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        estimator.reset()
        
        # Manual Warmup Logic
        if epoch < warmup_epochs:
            for i, param_group in enumerate(optimizer.param_groups):
                base_lr = cfg.train.backbone_lr if i == 0 else cfg.train.head_lr
                param_group['lr'] = base_lr * (epoch + 1) / warmup_epochs

        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')
        
        for step, train_data in enumerate(progress):
            # Your data loading logic
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
            y_hard = select_target_type(y, cfg.train.criterion)
            
            y_soft = y_hard
            if use_mixup_cutmix:
                X_side, y_soft = mixup_or_cutmix(X_side, y)

            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                y_pred = model(X_side, key_states, value_states)
                loss = loss_function(y_pred, y_soft)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            estimator.update(y_pred, y_hard)
            epoch_loss += loss.item() * accumulation_steps
            avg_loss = epoch_loss / (step + 1)
            current_lr = optimizer.param_groups[1]['lr']
            progress.set_postfix({'Loss': f'{avg_loss:.4f}', 'LR': f'{current_lr:.6f}'})

        train_scores = estimator.get_scores()
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_scores.get('accuracy', 0.0))

        if epoch >= warmup_epochs:
            scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        estimator.reset()
        with torch.no_grad():
            for val_data in val_loader:
                # Your validation data loading logic
                if cfg.dataset.preload_path:
                    X_side_val, key_states_val, value_states_val, y_val = val_data
                    key_states_val, value_states_val = key_states_val.to(device), value_states_val.to(device)
                    key_states_val, value_states_val = key_states_val.transpose(0, 1), value_states_val.transpose(0, 1)
                else:
                    X_lpm_val, X_side_val, y_val = val_data
                    X_lpm_val = X_lpm_val.to(device)
                    _, key_states_val, value_states_val = frozen_encoder(X_lpm_val, interpolate_pos_encoding=True)
                
                X_side_val, y_val = X_side_val.to(device), y_val.to(device)
                y_val_hard = select_target_type(y_val, cfg.train.criterion)
                
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    y_pred_val = model(X_side_val, key_states_val, value_states_val)
                
                val_loss += loss_function(y_pred_val, y_val_hard).item()
                estimator.update(y_pred_val, y_val_hard)

        avg_val_loss = val_loss / len(val_loader)
        val_scores = estimator.get_scores()
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_scores.get('accuracy', 0.0))
        
        print(f"\nEpoch {epoch+1} Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # --- End of Training ---
    print(f"Training finished. Loading best model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    
    # --- Final Visualization ---
    # Get final predictions and labels from the validation set for the confusion matrix
    final_preds, final_labels = [], []
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            if cfg.dataset.preload_path:
                X_side_val, key_states_val, value_states_val, y_val = val_data
                key_states_val, value_states_val = key_states_val.to(device), value_states_val.to(device)
                key_states_val, value_states_val = key_states_val.transpose(0, 1), value_states_val.transpose(0, 1)
            else:
                X_lpm_val, X_side_val, y_val = val_data
                X_lpm_val = X_lpm_val.to(device)
                _, key_states_val, value_states_val = frozen_encoder(X_lpm_val, interpolate_pos_encoding=True)
            X_side_val, y_val = X_side_val.to(device), y_val.to(device)
            y_pred_val = model(X_side_val, key_states_val, value_states_val)
            final_preds.extend(torch.argmax(y_pred_val, 1).cpu().numpy())
            final_labels.extend(y_val.cpu().numpy())

    log_and_visualize_results(cfg, history, cfg.dataset.class_names, final_preds, final_labels)
    
    # Save the final model weights (best performing one)
    save_weights(cfg, model, 'final_best_weights.pt')
    
    return model

# --- YOUR HELPER FUNCTIONS (Slightly modified or kept as is) ---

def evaluate(cfg, frozen_encoder, model, test_dataset, estimator):
    """
    Evaluates the final, best-performing model on the test set.
    """
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory
    )
    device = cfg.base.device
    print('--- Running final evaluation on Test set with the best model ---')
    
    model.eval()
    estimator.reset()
    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Testing"):
            if cfg.dataset.preload_path:
                X_side, key_states, value_states, y = test_data
                key_states, value_states = key_states.to(device), value_states.to(device)
                key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
            else:
                X_lpm, X_side, y = test_data
                X_lpm = X_lpm.to(device)
                _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

            X_side, y = X_side.to(device), y.to(device)
            y = select_target_type(y, cfg.train.criterion)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                y_pred = model(X_side, key_states, value_states)
            estimator.update(y_pred, y)

    print('\n================ Test Set Results ================')
    test_scores = estimator.get_scores()
    for metric, score in test_scores.items():
        print(f'{metric}: {score:.4f}')
    print('Confusion Matrix:')
    print(estimator.get_conf_mat())
    print('================================================')

# --- UNCHANGED HELPER FUNCTIONS ---
# These are your original functions that are still required.

def initialize_dataloader(cfg, train_dataset, val_dataset):
    # This function is kept as is from your original code.
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin_memory
    )
    return train_loader, val_loader

def save_weights(cfg, model, save_name):
    # This function is kept as is from your original code.
    save_path = os.path.join(cfg.base.working_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f'Model weights saved to {save_path}')

def resume(cfg, model, optimizer):
    # This function is kept as is from your original code.
    checkpoint_path = cfg.base.checkpoint
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
        return start_epoch
    else:
        print(f'No checkpoint found at {checkpoint_path}. Starting from scratch.')
        return 0

# Dummy select_target_type for standalone execution
def select_target_type(y, criterion):
    return y.long()
