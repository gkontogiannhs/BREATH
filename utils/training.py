"""
Training Utilities
Train and validation epoch loops, checkpoint saving/loading,
and learning rate scheduling.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from .metrics import run_inference, icbhi_score


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    spec_augment=None,
    grad_clip: float = 1.0,
    mixup_fn=None,
    normal_pool=None,
    abnormal_pool=None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The classifier.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CUDA or CPU device.
        spec_augment: Optional SpecAugment callable.
        grad_clip: Gradient clipping norm.
        mixup_fn: Optional CycleMixup instance.
        normal_pool: List of normal audio arrays for mixup.
        abnormal_pool: List of (audio, label) tuples for mixup.

    Returns:
        dict with "loss" and "accuracy".
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Train", leave=False, dynamic_ncols=True)

    for features, targets in pbar:
        # features: (B, 4, n_mels, T), targets: (B,)

        # SpecAugment on the batch (numpy → apply → back to tensor)
        if spec_augment is not None:
            feat_np = features.numpy()
            feat_np = np.stack([spec_augment(f) for f in feat_np], axis=0)
            features = torch.from_numpy(feat_np)

        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(features)
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict]:
    """
    Run validation for one epoch.

    Returns:
        Tuple of (loss_metrics, icbhi_metrics)
    """
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="  Valid", leave=False, dynamic_ncols=True)

    for features, targets in pbar:
        features = features.to(device)
        targets_device = targets.to(device)

        logits, _ = model(features)
        loss = criterion(logits, targets_device)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    metrics = icbhi_score(np.array(all_targets), np.array(all_preds))

    return {"loss": avg_loss}, metrics


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    save_path: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model.pt")
        torch.save(state, best_path)
        print(f"  ✓ Saved best model (ICBHI: {metrics.get('icbhi_score', 0):.4f})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = None,
) -> Dict:
    """Load model checkpoint. Returns the metadata dict."""
    if device is None:
        device = torch.device("cpu")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    print(f"  Loaded checkpoint from epoch {state['epoch']}")
    return state


# ---------------------------------------------------------------------------
# Optimizer and Scheduler Factories
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build optimizer."""
    params = model.parameters()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warm",
    epochs: int = 100,
    t0: int = 10,
    eta_min: float = 1e-6,
):
    """Build learning rate scheduler."""
    if scheduler_type == "cosine_warm":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=1, eta_min=eta_min
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
