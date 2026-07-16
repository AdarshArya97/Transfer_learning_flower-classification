"""
utils.py
Shared helpers: reproducibility, device selection, checkpointing,
the Oxford Flowers-102 class-name list, and the model registry that
maps a model name -> (constructor, pretrained weights, classifier-head attribute).
"""

import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model weights -> {path}")


def load_checkpoint(model: nn.Module, path: str, map_location=None) -> nn.Module:
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Model registry
# One entry per architecture family from the assignment's Model Options table.
# head_attr tells build_model() (in model.py) which attribute holds the
# final classification layer, per the assignment's Appendix table.
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    "resnet18": dict(
        ctor=models.resnet18,
        weights=models.ResNet18_Weights.DEFAULT,
        head_attr="fc",
    ),
    "resnet50": dict(
        ctor=models.resnet50,
        weights=models.ResNet50_Weights.DEFAULT,
        head_attr="fc",
    ),
    # "vgg16": dict(
    #     ctor=models.vgg16,
    #     weights=models.VGG16_Weights.DEFAULT,
    #     head_attr="classifier[6]",
    # ),
    # "densenet121": dict(
    #     ctor=models.densenet121,
    #     weights=models.DenseNet121_Weights.DEFAULT,
    #     head_attr="classifier",
    # ),
    # "efficientnet_v2_s": dict(
    #     ctor=models.efficientnet_v2_s,
    #     weights=models.EfficientNet_V2_S_Weights.DEFAULT,
    #     head_attr="classifier[1]",
    # ),
    # "mobilenet_v3_large": dict(
    #     ctor=models.mobilenet_v3_large,
    #     weights=models.MobileNet_V3_Large_Weights.DEFAULT,
    #     head_attr="classifier[3]",
    # ),
    # "convnext_tiny": dict(
    #     ctor=models.convnext_tiny,
    #     weights=models.ConvNeXt_Tiny_Weights.DEFAULT,
    #     head_attr="classifier[2]",
    # ),
}


# --------------------------------------------------------------------------
# Oxford Flowers-102 class names (index 0-101, matches torchvision's
# 0-indexed labels). torchvision does not ship human-readable names for
# this dataset, so we provide the commonly-used mapping ourselves purely
# for nicer plots / reports. Functionally the pipeline only needs the
# integer labels, so double check this list if you rely on it for grading.
# --------------------------------------------------------------------------
FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily",
]

assert len(FLOWER_CLASSES) == 102, "FLOWER_CLASSES must have exactly 102 entries"
