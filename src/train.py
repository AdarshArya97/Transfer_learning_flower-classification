"""
train.py
Task 3 - Training Functions & Task 4 - Running Experiments

train_one_epoch()     -> one pass over training data with backprop
fit()                  -> full training loop for one (model, strategy) combo
run_all_experiments() -> loops over every model in MODEL_REGISTRY and both
                          strategies (feature extraction + fine-tuning),
                          so you can compare all of them and pick the best.
run_two_phase_sweep() -> cheap screen (all models, feature extraction, few
                          epochs) to shortlist candidates, then a full
                          run_all_experiments() on just the shortlist —
                          use this instead of run_all_experiments() when
                          training all 7 models x 2 strategies end-to-end
                          is too slow.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders, NUM_CLASSES
from evaluate import evaluate
from model import build_model, get_param_groups
from utils import (
    MODEL_REGISTRY, get_device, save_checkpoint, load_checkpoint,
    save_json, load_json,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One training pass. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def fit(model, train_loader, val_loader, model_name, fine_tune, device,
        epochs=10, head_lr=1e-3, backbone_lr=1e-4, patience=None):
    """
    Full training loop.

    fine_tune=True uses two optimizer parameter groups: `head_lr` for the
    new classification head and the lower `backbone_lr` for the unfrozen
    backbone (see model.get_param_groups for the one-sentence justification).
    fine_tune=False only trains the head, so a single param group at
    head_lr is used (the backbone has no trainable params to group).
    """
    criterion = nn.CrossEntropyLoss()

    if fine_tune:
        param_groups = get_param_groups(model, model_name, head_lr, backbone_lr)
        optimizer = optim.Adam(param_groups)
    else:
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=head_lr
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{model_name} | {'fine_tune' if fine_tune else 'feature_extract'}] "
              f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience is not None and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_acc


def run_all_experiments(
    data_dir="../data",
    results_dir="../results",
    saved_models_dir="../saved_models",
    model_names=None,
    strategies=(False, True),
    epochs=10,
    batch_size=32,
    head_lr=1e-3,
    backbone_lr=1e-4,
    seed=42,
    skip_existing=True,
    
):
    """
    Loop over every model architecture x every strategy in `strategies`
    (False = feature extraction, True = fine-tuning), train each, and
    collect results into a single comparison table. Use this to run the
    full "try every model, then pick the best" sweep from the notebook.

    skip_existing=True (default): if a checkpoint for a given
        (model_name, strategy) already exists in `saved_models_dir` (e.g.
        from a previous, interrupted run), that model is loaded from disk
        instead of retrained from scratch. Its saved training history is
        loaded too if present (`{results_dir}/history_*.json`); otherwise
        just its validation accuracy is recomputed with one quick forward
        pass. Set skip_existing=False to force retraining everything.

    Returns a dict: {(model_name, fine_tune): {"history": ..., "best_val_acc": ...}}
    and also writes a leaderboard JSON to results_dir.
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    device = get_device()
    print(f"Using device: {device}")

    all_results = {}
    leaderboard = []

    for model_name in model_names:
        train_loader, val_loader, _test_loader, _class_names = get_dataloaders(
            model_name=model_name, data_dir=data_dir, batch_size=batch_size
        )

        for fine_tune in strategies:
            strategy_name = "finetune" if fine_tune else "feature_extract"
            ckpt_path = f"{saved_models_dir}/{model_name}_{strategy_name}.pth"
            history_path = f"{results_dir}/history_{model_name}_{strategy_name}.json"

            if skip_existing and Path(ckpt_path).exists():
                print(f"Found existing checkpoint for {model_name} ({strategy_name}) "
                      f"-> loading instead of retraining: {ckpt_path}")
                model = build_model(model_name, NUM_CLASSES, fine_tune=fine_tune).to(device)
                model = load_checkpoint(model, ckpt_path, map_location=device)

                if Path(history_path).exists():
                    history = load_json(history_path)
                    best_val_acc = max(history["val_acc"]) if history.get("val_acc") else None

                if not Path(history_path).exists() or best_val_acc is None:
                    # No saved history — recompute val accuracy with one quick pass
                    criterion = nn.CrossEntropyLoss()
                    _, best_val_acc = evaluate(model, val_loader, criterion, device)
                    history = {"val_acc": [best_val_acc]}
            else:
                torch.manual_seed(seed)
                model = build_model(model_name, NUM_CLASSES, fine_tune=fine_tune).to(device)

                model, history, best_val_acc = fit(
                    model, train_loader, val_loader, model_name, fine_tune, device,
                    epochs=epochs, head_lr=head_lr, backbone_lr=backbone_lr,
                )

                save_checkpoint(model, ckpt_path)
                save_json(history, history_path)

            all_results[(model_name, fine_tune)] = {
                "history": history,
                "best_val_acc": best_val_acc,
            }
            leaderboard.append({
                "model": model_name,
                "strategy": strategy_name,
                "best_val_acc": best_val_acc,
            })

    leaderboard.sort(key=lambda r: r["best_val_acc"], reverse=True)
    save_json(leaderboard, f"{results_dir}/leaderboard.json")

    print("\n=== Leaderboard (sorted by best validation accuracy) ===")
    for rank, row in enumerate(leaderboard, start=1):
        print(f"{rank}. {row['model']:22s} {row['strategy']:16s} val_acc={row['best_val_acc']:.4f}")

    return all_results, leaderboard


def run_two_phase_sweep(
    data_dir="../data",
    results_dir="../results",
    saved_models_dir="../saved_models",
    model_names=None,
    top_k=3,
    screen_epochs=3,
    full_epochs=10,
    screen_strategy=False,
    full_strategies=(False, True),
    batch_size=32,
    head_lr=1e-3,
    backbone_lr=1e-4,
    seed=42,
    skip_existing=True,
):
    """
    Two-phase version of run_all_experiments(), built to cut total training
    time when sweeping every architecture in MODEL_REGISTRY is too slow.

    Phase 1 (screen): every model in `model_names` (default: all of
        MODEL_REGISTRY) is trained briefly — `screen_epochs` epochs, and
        only with `screen_strategy` (feature extraction by default, since
        it's much cheaper than fine-tuning: the backbone never gets a
        backward pass). This ranks all candidates fast.

    Phase 2 (full): only the top `top_k` models from the screen move on to
        full training, using `full_strategies` (both feature extraction
        and fine-tuning by default) for `full_epochs` epochs each.

    This trades a small amount of ranking accuracy (the screen might
    occasionally mis-rank two close architectures) for a large cut in
    total compute — e.g. with 7 models, top_k=3, this runs 7 cheap
    screens + 6 full trainings instead of 14 full trainings.

    Returns (screen_results, screen_leaderboard, full_results, full_leaderboard).
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    print(f"=== Phase 1: screening {len(model_names)} models "
          f"({screen_epochs} epochs, strategy={'fine_tune' if screen_strategy else 'feature_extract'}) ===")
    screen_results, screen_leaderboard = run_all_experiments(
        data_dir=data_dir,
        results_dir=results_dir,
        saved_models_dir=saved_models_dir,
        model_names=model_names,
        strategies=(screen_strategy,),
        epochs=screen_epochs,
        batch_size=batch_size,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        seed=seed,
        skip_existing=skip_existing,
    )
    save_json(screen_leaderboard, f"{results_dir}/screen_leaderboard.json")

    top_models = [row["model"] for row in screen_leaderboard[:top_k]]
    print(f"\n=== Phase 1 complete. Shortlisted top {top_k}: {top_models} ===\n")

    print(f"=== Phase 2: full training on shortlist "
          f"({full_epochs} epochs, strategies={full_strategies}) ===")
    full_results, full_leaderboard = run_all_experiments(
        data_dir=data_dir,
        results_dir=results_dir,
        saved_models_dir=saved_models_dir,
        model_names=top_models,
        strategies=full_strategies,
        epochs=full_epochs,
        batch_size=batch_size,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        seed=seed,
        skip_existing=skip_existing,
    )

    return screen_results, screen_leaderboard, full_results, full_leaderboard
