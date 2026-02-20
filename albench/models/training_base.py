"""
Training utilities for sequence-to-function models.

Includes:
- Training loop with OneCycleLR scheduler
- Metric computation (Pearson R, Spearman R, MSE)
- Checkpointing
- Logging
"""

from typing import Dict, Any, Optional, Callable, Tuple
import time
from pathlib import Path
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        Dictionary with metrics:
        - pearson_r: Pearson correlation coefficient
        - spearman_r: Spearman correlation coefficient
        - mse: Mean squared error
    """
    # Remove any NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) < 2:
        return {
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "mse": float('inf')
        }
    
    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    mse = np.mean((predictions - targets) ** 2)
    
    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "mse": float(mse)
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[Any] = None,
    use_reverse_complement: bool = False
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scheduler: Optional learning rate scheduler (called after each batch)
        use_reverse_complement: Whether to average predictions with reverse complement
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (sequences, targets) in enumerate(pbar):
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (don't use predict() during training as it may disable gradients)
        predictions = model(sequences)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    metrics["loss"] = avg_loss
    
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_reverse_complement: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        use_reverse_complement: Whether to average predictions with reverse complement
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # Check if this is yeast mode (KL divergence loss)
    is_yeast_mode = hasattr(model, 'task_mode') and model.task_mode == 'yeast'
    
    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            if use_reverse_complement and hasattr(model, 'predict'):
                predictions = model.predict(sequences, use_reverse_complement=True)
            else:
                predictions = model(sequences)
            
            # Compute loss
            # For yeast: need logits for KL divergence
            # For K562: use predictions directly for MSE
            if is_yeast_mode:
                # Get logits for KL divergence loss
                logits = model.get_logits(sequences)
                loss = criterion(logits, targets)
            else:
                loss = criterion(predictions, targets)
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    metrics["loss"] = avg_loss
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[Any] = None,
    checkpoint_dir: Optional[Path] = None,
    use_reverse_complement: bool = True,
    early_stopping_patience: Optional[int] = None,
    metric_for_best: str = "pearson_r"
) -> Dict[str, Any]:
    """
    Train a model with validation and checkpointing.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of epochs to train
        device: Device to train on
        scheduler: Optional learning rate scheduler
        checkpoint_dir: Directory to save checkpoints
        use_reverse_complement: Whether to average predictions with reverse complement
        early_stopping_patience: Stop if no improvement for this many epochs
        metric_for_best: Metric to use for saving best model ('pearson_r', 'spearman_r', or 'loss')
        
    Returns:
        Dictionary with training history
    """
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        "train_loss": [],
        "train_pearson_r": [],
        "train_spearman_r": [],
        "val_loss": [],
        "val_pearson_r": [],
        "val_spearman_r": [],
        "learning_rates": []
    }
    
    best_metric = -float('inf') if metric_for_best != "loss" else float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rates"].append(current_lr)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler,
            use_reverse_complement=use_reverse_complement
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            use_reverse_complement=use_reverse_complement
        )
        
        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_pearson_r"].append(train_metrics["pearson_r"])
        history["train_spearman_r"].append(train_metrics["spearman_r"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_pearson_r"].append(val_metrics["pearson_r"])
        history["val_spearman_r"].append(val_metrics["spearman_r"])
        
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Pearson R: {train_metrics['pearson_r']:.4f}, "
              f"Spearman R: {train_metrics['spearman_r']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Pearson R: {val_metrics['pearson_r']:.4f}, "
              f"Spearman R: {val_metrics['spearman_r']:.4f}")
        
        # Check if this is the best model
        if metric_for_best == "loss":
            current_metric = val_metrics["loss"]
            is_best = current_metric < best_metric
        else:
            current_metric = val_metrics[metric_for_best]
            is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            
            if checkpoint_dir is not None:
                best_path = checkpoint_dir / "best_model.pt"
                model.save_checkpoint(
                    str(best_path),
                    epoch=epoch,
                    optimizer_state_dict=optimizer.state_dict(),
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )
                print(f"  → Saved best model (val {metric_for_best}: {best_metric:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best {metric_for_best}: {best_metric:.4f} at epoch {best_epoch+1}")
            break
        
        print("-" * 60)
    
    # Save final model
    if checkpoint_dir is not None:
        final_path = checkpoint_dir / "final_model.pt"
        model.save_checkpoint(
            str(final_path),
            epoch=num_epochs-1,
            optimizer_state_dict=optimizer.state_dict()
        )
        
        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best {metric_for_best}: {best_metric:.4f} at epoch {best_epoch+1}")
    
    return history


def create_optimizer_and_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float = 0.005,
    lr_lstm: float = 0.001,
    weight_decay: float = 0.01,
    pct_start: float = 0.3
) -> Tuple[Optimizer, Any]:
    """
    Create optimizer and OneCycleLR scheduler for DREAM-RNN.
    
    Uses different learning rates for CNN and LSTM components,
    as attention-like components (LSTM) are sensitive to high learning rates.
    
    Args:
        model: Model to optimize
        train_loader: Training data loader (for calculating steps)
        num_epochs: Number of training epochs
        lr: Learning rate for CNN layers
        lr_lstm: Learning rate for LSTM layers
        weight_decay: Weight decay for AdamW
        pct_start: Percentage of cycle spent increasing learning rate
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Separate parameters for different learning rates
    lstm_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'lstm' in name:
            lstm_params.append(param)
        else:
            other_params.append(param)
    
    # Create optimizer with parameter groups
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr},
        {'params': lstm_params, 'lr': lr_lstm}
    ], weight_decay=weight_decay)
    
    # Create OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr, lr_lstm],
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        cycle_momentum=False,  # AdamW doesn't use momentum
        div_factor=25.0,  # Initial lr = max_lr / div_factor
        final_div_factor=1e4  # Final lr = max_lr / final_div_factor
    )
    
    return optimizer, scheduler
