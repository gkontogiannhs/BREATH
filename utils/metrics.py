"""
Metrics and Evaluation Utilities
Implements the official ICBHI Score = (Sensitivity + Specificity) / 2
plus per-class F1, confusion matrix, and related helpers.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
)


CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


def icbhi_score(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4
) -> Dict[str, float]:
    """
    Compute the official ICBHI Score.

    ICBHI Score = (Sensitivity + Specificity) / 2
    where Sensitivity and Specificity are computed as the macro-average
    across all classes (each class treated as one-vs-rest).

    Args:
        y_true: (N,) integer true labels.
        y_pred: (N,) integer predicted labels.
        n_classes: Number of classes.

    Returns:
        Dictionary with:
            - "icbhi_score": main metric
            - "sensitivity": macro-averaged sensitivity (recall)
            - "specificity": macro-averaged specificity
            - "per_class_sensitivity": per-class sensitivity
            - "per_class_specificity": per-class specificity
            - "per_class_f1": per-class F1
            - "macro_f1": macro-averaged F1
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    per_class_sensitivity = []
    per_class_specificity = []

    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        per_class_sensitivity.append(sensitivity)
        per_class_specificity.append(specificity)

    macro_sensitivity = np.mean(per_class_sensitivity)
    macro_specificity = np.mean(per_class_specificity)
    score = (macro_sensitivity + macro_specificity) / 2.0

    per_class_f1 = f1_score(y_true, y_pred, labels=list(range(n_classes)),
                            average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "icbhi_score": float(score),
        "sensitivity": float(macro_sensitivity),
        "specificity": float(macro_specificity),
        "per_class_sensitivity": {
            CLASS_NAMES[i]: float(per_class_sensitivity[i]) for i in range(n_classes)
        },
        "per_class_specificity": {
            CLASS_NAMES[i]: float(per_class_specificity[i]) for i in range(n_classes)
        },
        "per_class_f1": {
            CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(n_classes)
        },
        "macro_f1": float(macro_f1),
    }


def format_metrics(metrics: Dict) -> str:
    """Pretty-print metrics dictionary."""
    lines = []
    lines.append(f"  ICBHI Score:   {metrics['icbhi_score']:.4f}")
    lines.append(f"  Sensitivity:   {metrics['sensitivity']:.4f}")
    lines.append(f"  Specificity:   {metrics['specificity']:.4f}")
    lines.append(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    lines.append("  Per-class F1:")
    for cls, val in metrics["per_class_f1"].items():
        lines.append(f"    {cls:<10}: {val:.4f}")
    return "\n".join(lines)


class MetricTracker:
    """
    Tracks training and validation metrics across epochs.
    Maintains best model state based on ICBHI score.
    """

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_icbhi_score": [],
            "val_sensitivity": [],
            "val_specificity": [],
            "val_macro_f1": [],
        }
        self.best_icbhi_score = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict,
    ) -> bool:
        """
        Update tracker. Returns True if this is the best epoch so far.
        """
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_icbhi_score"].append(val_metrics["icbhi_score"])
        self.history["val_sensitivity"].append(val_metrics["sensitivity"])
        self.history["val_specificity"].append(val_metrics["specificity"])
        self.history["val_macro_f1"].append(val_metrics["macro_f1"])

        is_best = val_metrics["icbhi_score"] > self.best_icbhi_score
        if is_best:
            self.best_icbhi_score = val_metrics["icbhi_score"]
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return is_best

    def should_stop_early(self, patience: int) -> bool:
        return self.epochs_without_improvement >= patience

    def summary(self) -> str:
        return (
            f"Best ICBHI Score: {self.best_icbhi_score:.4f} "
            f"at epoch {self.best_epoch}"
        )


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    return_attention: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Run full inference over a dataloader.

    Returns:
        all_preds: (N,) predicted labels
        all_targets: (N,) true labels
        all_attn: list of (T,) attention weight arrays (if return_attention=True)
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_attn = []

    for batch in dataloader:
        features, targets = batch
        features = features.to(device)

        logits, attn_weights = model(features)
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())

        if return_attention and attn_weights is not None:
            all_attn.extend([w.cpu().numpy() for w in attn_weights])

    return np.array(all_preds), np.array(all_targets), all_attn


@torch.no_grad()
def test_time_augmentation(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    n_augments: int = 5,
) -> torch.Tensor:
    """
    Test-Time Augmentation (TTA).
    Apply small random time shifts and noise, average softmax probabilities.

    Args:
        features: (B, 4, n_mels, T)
        n_augments: number of augmented copies to average

    Returns:
        averaged_probs: (B, n_classes)
    """
    import torch.nn.functional as F

    model.eval()
    features = features.to(device)
    all_probs = []

    for i in range(n_augments):
        aug = features.clone()

        if i > 0:
            # Small random time shift (circular)
            shift = torch.randint(-5, 6, (1,)).item()
            aug = torch.roll(aug, shift, dims=-1)

            # Small additive noise
            noise_scale = 0.02 * torch.rand(1).item()
            aug = aug + noise_scale * torch.randn_like(aug)

        logits, _ = model(aug)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs)

    return torch.stack(all_probs, dim=0).mean(dim=0)  # (B, n_classes)
