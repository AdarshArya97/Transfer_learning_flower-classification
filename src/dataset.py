"""
dataset.py
Task 1 - Data Pipeline

build_transforms(model_name)   -> (train_transform, eval_transform)
get_dataloaders(model_name, ...) -> train_loader, val_loader, test_loader, class_names
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102

from utils import MODEL_REGISTRY, FLOWER_CLASSES

NUM_CLASSES = 102  # Oxford Flowers-102


def build_transforms(model_name: str):
    """
    Return (train_transform, eval_transform) sized/normalized for the chosen
    pretrained model.

    We pull the exact preprocessing recipe the pretrained weights were
    trained with via weights.transforms() instead of guessing input size /
    normalization constants. A light augmentation pipeline is layered on
    top for the *training* transform only; the eval transform uses the
    pretrained preprocessing as-is.
    """
    weights = MODEL_REGISTRY[model_name]["weights"]
    base_eval_transform = weights.transforms()

    # Recover the resize/crop size the pretrained transform uses so our
    # augmentations operate at the same resolution before final resize/norm.
    crop_size = base_eval_transform.crop_size[0] if hasattr(base_eval_transform, "crop_size") else 224
    resize_size = base_eval_transform.resize_size[0] if hasattr(base_eval_transform, "resize_size") else 256

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=base_eval_transform.mean, std=base_eval_transform.std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=base_eval_transform.mean, std=base_eval_transform.std),
    ])

    return train_transform, eval_transform


def get_dataloaders(
    model_name: str,
    data_dir: str = "../data",
    batch_size: int = 32,
    num_workers: int = 2,  # on macOS, set to 0 to silence "MallocStackLogging" worker-exit noise
):
    """
    Load Oxford Flowers-102 and return train/val/test DataLoaders + class names.

    Flowers102 ships with its own official train/val/test splits (via the
    `split=` argument), so we use those directly rather than carving up
    the data ourselves.
    """
    train_transform, eval_transform = build_transforms(model_name)

    train_set = Flowers102(root=data_dir, split="train", transform=train_transform, download=True)
    val_set = Flowers102(root=data_dir, split="val", transform=eval_transform, download=True)
    test_set = Flowers102(root=data_dir, split="test", transform=eval_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    class_names = FLOWER_CLASSES  # 0-indexed, matches torchvision's labels

    print(f"[{model_name}] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}  "
          f"classes={len(class_names)}")

    return train_loader, val_loader, test_loader, class_names
