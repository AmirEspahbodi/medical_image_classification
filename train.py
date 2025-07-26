import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- Assume these are your existing utility imports ---
# from src.utils.func import *
# from src.loss import *
# from src.scheduler import *

# It's good practice to have a helper for saving checkpoints
def save_checkpoint(model, optimizer, scheduler, epoch, file_path):
    """Saves model, optimizer, scheduler, and epoch to a file."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, file_path)
    print(f"Checkpoint saved to {file_path}")

# It's also good practice to have a helper for saving final weights
def save_weights(model, file_path):
    """Saves final model weights."""
    torch.save(model.state_dict(), file_path)
    print(f"Best model weights saved to {file_path}")

# --- 1. Sharpness-Aware Minimization (SAM) Optimizer ---
# A state-of-the-art optimizer that finds flatter, more generalizable minima.
# This implementation is a common variant used in many projects.
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # Add a small epsilon for numerical stability
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        # Put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        # Return a default value if no gradients are found
        grads = [p.grad for group in self.param_groups for p in group["params"] if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=shared_device)
        
        # *** FIX: Use float32 for stability critical norm calculation ***
        with torch.amp.autocast('cuda', dtype=torch.float32):
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

class ModernTrainer:
    """
    An advanced trainer class that encapsulates modern training techniques
    to combat overfitting and improve generalization.
    
    Version 4 Changes:
    - Increased AdamW epsilon for stability with AMP.
    - Forced SAM's grad_norm calculation to float32.
    - Added loss sanity check to skip unstable steps.
    """
    def __init__(self, cfg, model, frozen_encoder, train_dataset, val_dataset, estimator):
        self.cfg = cfg
        self.device = cfg.base.device
        self.model = model.to(self.device)
        self.frozen_encoder = frozen_encoder.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.estimator = estimator

        # --- Optimizer Setup with Differential Learning Rates ---
        backbone_params = list(self.model.cnn_backbone.parameters())
        head_params = [p for p in self.model.parameters() if not any(id(p) == id(bp) for bp in backbone_params)]
        
        param_groups = [
            {'params': backbone_params, 'lr': self.cfg.train.backbone_lr},
            {'params': head_params, 'lr': self.cfg.train.learning_rate}
        ]

        # --- Base optimizer is AdamW, wrapped by SAM ---
        self.optimizer = SAM(
            param_groups, 
            torch.optim.AdamW, 
            rho=self.cfg.train.sam_rho,
            adaptive=self.cfg.train.sam_adaptive,
            weight_decay=self.cfg.train.weight_decay,
            # *** FIX: Increased epsilon for stability in mixed precision ***
            eps=1e-7 
        )

        # --- Loss Function: CrossEntropy with Label Smoothing ---
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg.train.label_smoothing)

        # --- DataLoaders & Gradient Accumulation ---
        self.train_loader, self.val_loader = self.initialize_dataloader()
        self.accumulation_steps = self.cfg.train.get('accumulation_steps', 1)
        self.clip_grad_norm_value = self.cfg.train.get('clip_grad_norm', 1.0)

        # --- Scheduler: OneCycleLR ---
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer.base_optimizer,
            max_lr=[self.cfg.train.backbone_lr, self.cfg.train.learning_rate],
            steps_per_epoch=len(self.train_loader) // self.accumulation_steps,
            epochs=self.cfg.train.epochs,
            pct_start=0.3
        )

        # --- Automatic Mixed Precision (AMP) - UPDATED API ---
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.train.amp)

        # --- Early Stopping ---
        self.early_stopping_patience = self.cfg.train.early_stopping_patience
        self.epochs_no_improve = 0
        self.best_metric = -1.0

    def initialize_dataloader(self):
        train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers, pin_memory=True)
        return train_loader, val_loader

    def _get_encoder_states(self, data):
        if self.cfg.dataset.preload_path:
            X_side, key_states, value_states, y = data
            key_states, value_states = key_states.to(self.device, non_blocking=True).transpose(0, 1), value_states.to(self.device, non_blocking=True).transpose(0, 1)
        else:
            X_lpm, X_side, y = data
            X_lpm = X_lpm.to(self.device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.cfg.train.amp):
                _, key_states, value_states = self.frozen_encoder(X_lpm, interpolate_pos_encoding=True)
        X_side, y = X_side.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        return X_side, key_states, value_states, y

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.estimator.reset()
        epoch_loss = 0.0
        progress = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.cfg.train.epochs} [TRAIN]', unit='batch')
        
        for step, train_data in enumerate(progress):
            X_side, key_states, value_states, y = self._get_encoder_states(train_data)
            y_for_loss = select_target_type(y, self.cfg.train.criterion)

            # --- First forward/backward pass (ascent step) ---
            with torch.amp.autocast('cuda', enabled=self.cfg.train.amp):
                y_pred = self.model(X_side, key_states, value_states)
                loss1 = self.loss_fn(y_pred, y_for_loss)

            # *** FIX: Sanity check the loss to prevent propagating NaNs ***
            if not torch.isfinite(loss1):
                print(f"Warning: Unstable loss detected: {loss1.item()}. Skipping step.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss1).backward()
            
            # This step perturbs the weights to find a region of flat loss
            self.optimizer.first_step(zero_grad=True)

            # --- Second forward/backward pass (descent step) ---
            with torch.amp.autocast('cuda', enabled=self.cfg.train.amp):
                y_pred_2 = self.model(X_side, key_states, value_states)
                loss2 = self.loss_fn(y_pred_2, y_for_loss)
            self.scaler.scale(loss2).backward()

            # The second step does the actual parameter update
            self.scaler.unscale_(self.optimizer.base_optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            
            # Only step if the gradients are finite after clipping
            if torch.isfinite(grad_norm):
                self.optimizer.second_step(zero_grad=True)
                self.scaler.update()
                self.scheduler.step()
            else:
                print(f"Warning: Grad norm is not finite: {grad_norm}. Skipping optimizer step.")
                # If grads are bad, we still need to zero them before the next step
                self.optimizer.zero_grad(set_to_none=True)

            
            # Logging uses the first loss, as it's representative
            loss_val = loss1.item()
            epoch_loss += loss_val
            avg_loss = epoch_loss / (step + 1)
            
            self.estimator.update(y_pred.detach(), y)
            progress.set_postfix({'Loss': f'{avg_loss:.4f}', 'LR': f'{self.optimizer.param_groups[1]["lr"]:.2e}'})
            
        train_scores = self.estimator.get_scores()
        return avg_loss, train_scores

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        self.estimator.reset()
        val_loss = 0.0
        progress = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.cfg.train.epochs} [VAL]', unit='batch')

        with torch.no_grad():
            for step, val_data in enumerate(progress):
                X_side, key_states, value_states, y = self._get_encoder_states(val_data)
                y_for_loss = select_target_type(y, self.cfg.train.criterion)
                with torch.amp.autocast('cuda', enabled=self.cfg.train.amp):
                    y_pred = self.model(X_side, key_states, value_states)
                    loss = self.loss_fn(y_pred, y_for_loss)
                val_loss += loss.item()
                self.estimator.update(y_pred, y)
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_scores = self.estimator.get_scores()
        return avg_val_loss, val_scores

    def train(self):
        print("--- Starting SOTA Training (v4 - Stabilized) ---")
        print(f"Optimizer: SAM(AdamW) | Scheduler: OneCycleLR | Loss: LabelSmoothing")
        print(f"Differential LRs | Grad Accum: {self.accumulation_steps} | AMP: {self.cfg.train.amp} | Clip Norm: {self.clip_grad_norm_value}")
        
        for epoch in range(self.cfg.train.epochs):
            train_loss, train_scores = self._train_one_epoch(epoch)
            scores_txt = ', '.join([f'{m}: {s:.4f}' for m, s in train_scores.items()])
            print(f"Epoch {epoch+1} Train | Loss: {train_loss:.4f} | {scores_txt}")

            val_loss, val_scores = self._validate_one_epoch(epoch)
            scores_txt = ', '.join([f'{m}: {s:.4f}' for m, s in val_scores.items()])
            print(f"Epoch {epoch+1} Valid | Loss: {val_loss:.4f} | {scores_txt}")

            indicator_metric = val_scores[self.cfg.train.indicator]
            if indicator_metric > self.best_metric:
                print(f"Validation metric improved from {self.best_metric:.4f} to {indicator_metric:.4f}.")
                self.best_metric = indicator_metric
                self.epochs_no_improve = 0
                save_weights(self.model, os.path.join(self.cfg.dataset.save_path, 'best_validation_weights.pt'))
            else:
                self.epochs_no_improve += 1
                print(f"Validation metric did not improve. Count: {self.epochs_no_improve}/{self.early_stopping_patience}")

            if self.epochs_no_improve >= self.early_stopping_patience:
                print("--- Early stopping triggered! ---")
                break
        
        print("--- Finished Training ---")
        save_weights(self.model, os.path.join(self.cfg.dataset.save_path, 'final_weights.pt'))

def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    trainer = ModernTrainer(cfg, model, frozen_encoder, train_dataset, val_dataset, estimator)
    trainer.train()

def select_target_type(y, criterion):
    # This function should be defined elsewhere in your project.
    return y
