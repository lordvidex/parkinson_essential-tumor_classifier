"""
inference.py — Model loading, prediction, and GradCAM for the Streamlit app.
Keeps all PyTorch/MONAI logic separate from the UI layer.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from monai.networks.nets import EfficientNetBN
from monai.transforms import (
    EnsureChannelFirst, ScaleIntensity, Resize, ToTensor, Compose, Lambda
)


# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["ET", "PD"]
IMAGE_SIZE   = 224
DEFAULT_META = {
    "model_name" : "efficientnet-b0",
    "in_channels": 1,
    "image_size" : IMAGE_SIZE,
    "num_classes": 2,
    "class_names": CLASS_NAMES,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Transforms ───────────────────────────────────────────────────────────────
def build_transforms(in_channels: int, image_size: int) -> Compose:
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),  # (H,W) → (1,H,W)
        Lambda(lambda x: x[:in_channels]),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(image_size, image_size)),
        ToTensor(),
    ])


# ─── Model loader ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str):
    """
    Load MONAI EfficientNetBN from a .pth checkpoint.

    Supports two checkpoint formats:
      1. Full dict  — saved with metadata keys (from the training notebook)
      2. State dict — raw state_dict only
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Determine format
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        meta = {k: raw.get(k, DEFAULT_META[k]) for k in DEFAULT_META}
    else:
        state_dict = raw
        meta = DEFAULT_META.copy()

    model = EfficientNetBN(
        model_name  = meta["model_name"],
        pretrained  = False,              # weights come from checkpoint
        spatial_dims= 2,
        in_channels = meta["in_channels"],
        num_classes = meta["num_classes"],
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, meta


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(img_array: np.ndarray, meta: dict) -> torch.Tensor:
    """
    Convert a raw numpy (H, W) float32 array to a model-ready (1, C, H, W) tensor.
    """
    transforms = build_transforms(meta["in_channels"], meta["image_size"])
    tensor = transforms(img_array)         # (C, H, W)
    return tensor.unsqueeze(0).to(DEVICE)  # (1, C, H, W)


# ─── Prediction ───────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, img_array: np.ndarray, meta: dict) -> np.ndarray:
    """
    Returns softmax probabilities [P(ET), P(PD)] as a float32 numpy array.
    """
    tensor = preprocess(img_array, meta)
    logits = model(tensor)                             # (1, 2)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (2,)
    return probs.astype(np.float32)


# ─── GradCAM ──────────────────────────────────────────────────────────────────
class _GradCAMHook:
    """Minimal GradCAM — no extra dependency beyond PyTorch."""

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._handles    = []

        # Target: last conv block in EfficientNet-B0
        target = model._blocks[-1]

        self._handles.append(
            target.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            target.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(tensor)                        # (1, C)
        score  = output[0, class_idx]
        score.backward()

        # Global average pool gradients → weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, Ch, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam     = torch.relu(cam).squeeze().cpu().numpy()

        # Resize to input resolution
        from PIL import Image as PILImage
        cam_img  = PILImage.fromarray(cam).resize(
            (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR
        )
        cam_np   = np.array(cam_img, dtype=np.float32)

        # Normalise to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove(self):
        for h in self._handles:
            h.remove()


def generate_gradcam(
    model: nn.Module,
    img_array: np.ndarray,
    meta: dict,
    class_idx: int,
) -> np.ndarray:
    """
    Compute GradCAM heatmap for the given class index.
    Returns a (H, W) float32 array in [0, 1].
    """
    tensor = preprocess(img_array, meta)
    tensor.requires_grad_(True)

    gcam   = _GradCAMHook(model)

    # Temporarily set to train mode so backward works
    model.train()
    heatmap = gcam(tensor, class_idx)
    model.eval()

    gcam.remove()
    return heatmap
