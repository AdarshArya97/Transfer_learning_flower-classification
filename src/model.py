"""
model.py
Task 2 - Model Construction

build_model(model_name, num_classes, fine_tune) loads any of the seven
architectures from the assignment's Model Options table, freezes/unfreezes
the backbone based on `fine_tune`, and replaces the final classification
layer per the Appendix table.
"""

import torch.nn as nn

from utils import MODEL_REGISTRY


def _freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _replace_head(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    """
    Swap the final classification layer per the assignment's Appendix table:

        ResNet (18/34/50/101/152)        model.fc
        VGG (11/13/16/19)                model.classifier[6]
        DenseNet (121/169/201)           model.classifier
        EfficientNet / EfficientNetV2    model.classifier[1]
        MobileNetV3                      model.classifier[3]
        ConvNeXt                         model.classifier[2]
    """
    head_attr = MODEL_REGISTRY[model_name]["head_attr"]

    if head_attr == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif head_attr == "classifier":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif head_attr.startswith("classifier["):
        idx = int(head_attr.split("[")[1].rstrip("]"))
        in_features = model.classifier[idx].in_features
        model.classifier[idx] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unhandled head_attr '{head_attr}' for model '{model_name}'")

    return model


def build_model(model_name: str, num_classes: int, fine_tune: bool) -> nn.Module:
    """
    Load `model_name` with pretrained ImageNet weights, adapt it to
    `num_classes` outputs, and set up its trainable parameters:

        fine_tune=False (feature extraction): backbone frozen, only the
            new head is trained.
        fine_tune=True (fine-tuning): backbone unfrozen (trainable at a
            lower LR, see train.py's two param groups), head trainable
            at a higher LR.

    In both cases the new classification head is freshly initialized and
    always trainable.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                          f"Choose from {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[model_name]
    model = entry["ctor"](weights=entry["weights"])

    # Freeze the whole backbone first (applies to both strategies as the
    # starting point); the new head is replaced afterwards, which makes
    # it trainable by construction (fresh nn.Linear params require grad).
    _freeze_backbone(model)

    if fine_tune:
        # Unfreeze the backbone so it can be fine-tuned at a low LR.
        for param in model.parameters():
            param.requires_grad = True

    model = _replace_head(model, model_name, num_classes)

    # Freshly-initialized head must always be trainable, even in the
    # feature-extraction (frozen backbone) case.
    head_attr = entry["head_attr"]
    if head_attr == "fc":
        head_module = model.fc
    elif head_attr == "classifier":
        head_module = model.classifier
    else:
        idx = int(head_attr.split("[")[1].rstrip("]"))
        head_module = model.classifier[idx]

    for param in head_module.parameters():
        param.requires_grad = True

    return model


def get_param_groups(model: nn.Module, model_name: str, head_lr: float, backbone_lr: float):
    """
    Build the two optimizer parameter groups required for fine-tuning:
    a higher LR for the newly-initialized head and a lower LR for the
    unfrozen backbone. New head trains faster because its weights are
    random and need larger updates to learn useful features quickly,
    while the pretrained backbone only needs small nudges to adapt
    without destroying the ImageNet features it already learned.
    """
    head_attr = MODEL_REGISTRY[model_name]["head_attr"]
    if head_attr == "fc":
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
    elif head_attr == "classifier":
        head_params = list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
    else:
        idx = int(head_attr.split("[")[1].rstrip("]"))
        head_params = list(model.classifier[idx].parameters())
        head_ids = {id(p) for p in head_params}

    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    param_groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    return param_groups
