"""
evaluate.py
Task 3 (evaluate) & Task 5 - Analysis

evaluate()        -> no-grad pass over a loader, returns (loss, accuracy)
get_predictions() -> run inference over a loader, return (y_true, y_pred)
plot_history()    -> compare val accuracy/loss curves across strategies
plot_confusion_matrix() / classification_report_str() -> Task 5 analysis
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """One pass over a data loader with no gradient updates. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference over `loader`. Returns (y_true, y_pred) numpy arrays."""
    model.eval()
    all_true, all_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_true.append(labels.numpy())
        all_pred.append(preds)

    return np.concatenate(all_true), np.concatenate(all_pred)


def plot_history(history, title="Training History", save_path=None):
    """
    Plot train/val loss and accuracy curves for a single run.

    Handles partial history gracefully: if this run was resumed from a
    saved checkpoint that had no matching history_*.json (see
    train.run_all_experiments' skip_existing fallback), `history` may only
    contain "val_acc" with a single value. In that case we plot what's
    available and print a note instead of crashing.
    """
    has_full_curves = all(k in history and len(history[k]) > 0
                           for k in ("train_loss", "train_acc", "val_loss", "val_acc"))

    if not has_full_curves:
        print(f"Note: '{title}' has no full per-epoch history (likely resumed from a "
              f"checkpoint saved before history logging, or trained with an older "
              f"version of run_all_experiments). Only plotting what's available: "
              f"{list(history.keys())}")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        for key, values in history.items():
            if len(values) > 1:
                ax.plot(range(1, len(values) + 1), values, label=key)
            else:
                ax.axhline(values[0], linestyle="--", label=f"{key} (single value: {values[0]:.4f})")
        ax.set_title(f"{title} (partial history)")
        ax.set_xlabel("Epoch")
        ax.legend()
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot -> {save_path}")
        return fig

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {save_path}")
    return fig


def plot_strategy_comparison(history_feat, history_ft, model_name="model", save_path=None):
    """
    Compare validation accuracy/loss curves for feature extraction vs
    fine-tuning on the same model, as required in Task 5.
    """
    for label, h in (("feature-extraction", history_feat), ("fine-tuning", history_ft)):
        if "val_loss" not in h or "val_acc" not in h or len(h.get("val_acc", [])) == 0:
            raise ValueError(
                f"plot_strategy_comparison needs val_loss/val_acc curves for {label}, "
                f"but only found keys {list(h.keys())}. This run was likely resumed from "
                f"a checkpoint with no saved history_*.json — retrain it (skip_existing=False "
                f"for just this model) to get real per-epoch curves."
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(range(1, len(history_feat["val_loss"]) + 1), history_feat["val_loss"],
                 label="Feature extraction")
    axes[0].plot(range(1, len(history_ft["val_loss"]) + 1), history_ft["val_loss"],
                 label="Fine-tuning")
    axes[0].set_title(f"{model_name}: Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(range(1, len(history_feat["val_acc"]) + 1), history_feat["val_acc"],
                 label="Feature extraction")
    axes[1].plot(range(1, len(history_ft["val_acc"]) + 1), history_ft["val_acc"],
                 label="Fine-tuning")
    axes[1].set_title(f"{model_name}: Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {save_path}")
    return fig





def plot_confusion_matrix(y_true, y_pred, class_names, top_n=None, save_path=None,
                           title="Confusion Matrix"):
    """
    Plot a confusion matrix. With 102 classes, a full matrix is unreadable,
    so by default `top_n` restricts the plot to the N classes with the most
    test samples. Pass top_n=None to plot the full 102x102 matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if top_n is not None:
        counts = cm.sum(axis=1)
        top_idx = np.argsort(counts)[::-1][:top_n]
        cm = cm[np.ix_(top_idx, top_idx)]
        labels = [class_names[i] for i in top_idx]
    else:
        labels = class_names

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), max(6, len(labels) * 0.35)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate each cell with its value; text color flips based on
    # cell darkness so numbers stay readable on both light and dark cells.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value == 0:
                continue  # skip zeros to reduce clutter on large matrices
            ax.text(
                j, i, format(value, "d"),
                ha="center", va="center",
                fontsize=6,
                color="white" if value > thresh else "black"
            )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix -> {save_path}")
    return fig, cm


def classification_report_str(y_true, y_pred, class_names):
    """Return the sklearn classification report as a printable string."""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def most_confused_pairs(cm, class_names, top_k=10):
    """
    From a confusion matrix, return the top_k (true_class, predicted_class, count)
    off-diagonal pairs — i.e. the most frequently confused class pairs (Task 5, #13/#20).
    """
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def save_predictions_csv(y_true, y_pred, class_names, save_path):
    """Save raw true/predicted labels (and names) to a CSV for the results/predictions folder."""
    import csv

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label", "true_class", "pred_class"])
        for t, p in zip(y_true, y_pred):
            writer.writerow([int(t), int(p), class_names[t], class_names[p]])
    print(f"Saved predictions -> {save_path}")
