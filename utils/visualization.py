"""
Visualization Utilities
- Attention weight maps (event localization)
- Training curves
- Confusion matrix
- Per-channel spectrogram viewer
"""

import numpy as np
import os
from typing import List, Optional, Dict


def plot_training_curves(
    history: Dict,
    save_path: str,
    show: bool = False,
):
    """Plot training loss, validation loss, and ICBHI score curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ICBHI Score
    axes[1].plot(epochs, history["val_icbhi_score"], label="ICBHI Score",
                 color="seagreen", linewidth=2)
    axes[1].plot(epochs, history["val_sensitivity"], label="Sensitivity",
                 color="cornflowerblue", linestyle="--")
    axes[1].plot(epochs, history["val_specificity"], label="Specificity",
                 color="darkorange", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"  Saved training curves → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    show: bool = False,
):
    """Plot normalized confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("matplotlib/sklearn not available; skipping confusion matrix.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})",
                    ha="center", va="center", fontsize=9, color=text_color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


def plot_attention_map(
    features: np.ndarray,
    attn_weights: np.ndarray,
    predicted_label: int,
    true_label: int,
    class_names: List[str],
    save_path: str,
    sample_rate: int = 22050,
    hop_length: int = 220,
    show: bool = False,
):
    """
    Visualize the 4-channel spectrogram alongside the temporal attention weights.
    The attention map shows where in the respiratory cycle the model is looking.

    Args:
        features: (4, n_mels, T) feature array.
        attn_weights: (T',) temporal attention weights (may be downsampled from T).
        predicted_label: predicted class index.
        true_label: true class index.
        class_names: list of class name strings.
        save_path: where to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.ndimage import zoom
    except ImportError:
        print("matplotlib not available; skipping attention map.")
        return

    channel_names = [
        "Log Mel (Ch 0)",
        "Harmonic / Wheeze (Ch 1)",
        "Percussive / Crackle (Ch 2)",
        "Delta / Onset (Ch 3)",
    ]
    channel_cmaps = ["inferno", "plasma", "viridis", "seismic"]

    n_channels = features.shape[0]
    T = features.shape[-1]

    # Upsample attention weights to match T if needed
    if len(attn_weights) != T:
        scale = T / len(attn_weights)
        attn_upsampled = zoom(attn_weights, scale, order=1)
        attn_upsampled = np.clip(attn_upsampled, 0, None)
    else:
        attn_upsampled = attn_weights

    # Normalize attention for display
    attn_upsampled = attn_upsampled / (attn_upsampled.max() + 1e-8)

    # Time axis in seconds
    time_axis = np.arange(T) * hop_length / sample_rate

    fig, axes = plt.subplots(
        n_channels + 1, 1,
        figsize=(14, 3 * (n_channels + 1)),
        gridspec_kw={"height_ratios": [3] * n_channels + [1.5]}
    )

    for c in range(n_channels):
        im = axes[c].imshow(
            features[c],
            aspect="auto",
            origin="lower",
            cmap=channel_cmaps[c],
            extent=[time_axis[0], time_axis[-1], 0, features.shape[1]],
        )
        axes[c].set_ylabel("Mel Bin")
        axes[c].set_title(channel_names[c])
        plt.colorbar(im, ax=axes[c], fraction=0.02)

    # Attention weights
    axes[-1].fill_between(time_axis, attn_upsampled, alpha=0.7, color="royalblue")
    axes[-1].plot(time_axis, attn_upsampled, color="navy", linewidth=1.5)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Attn Weight")
    axes[-1].set_title(
        f"Temporal Attention  |  "
        f"True: {class_names[true_label]}  |  "
        f"Pred: {class_names[predicted_label]}"
    )
    axes[-1].set_xlim(time_axis[0], time_axis[-1])
    axes[-1].grid(True, alpha=0.3)

    plt.suptitle(
        "Multi-Channel Features and Event Localization Map",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Saved attention map → {save_path}")


def plot_per_class_metrics(
    metrics_history: List[Dict],
    class_names: List[str],
    save_path: str,
    show: bool = False,
):
    """Plot per-class F1 score evolution across epochs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = ["steelblue", "tomato", "seagreen", "darkorchid"]
    epochs = range(1, len(metrics_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cls in enumerate(class_names):
        f1_vals = [m["per_class_f1"].get(cls, 0) for m in metrics_history]
        ax.plot(epochs, f1_vals, label=cls, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"  Saved per-class F1 curves → {save_path}")
