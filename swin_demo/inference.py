"""
inference.py — Model loading, prediction, GradCAM & Attention Rollout
                for the Swin-B/S 3-class MRI Streamlit app.
"""

from __future__ import annotations
import os
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage

# ── torchvision Swin imports (both variants) ──────────────────────────────────
from torchvision.models import (
    swin_b, Swin_B_Weights,
    swin_s, Swin_S_Weights,
)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["ET", "PD", "Healthy"]
NUM_CLASSES  = 3
CLASS_TO_IDX = {"ET": 0, "PD": 1, "Healthy": 2}
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Swin-B embed dim = 1024 ; Swin-S embed dim = 768
EMBED_DIM_MAP = {"swin_b": 1024, "swin_s": 768}

HEATMAP_ALPHA        = 0.45
BRAIN_MASK_DILATE_PX = 8
BRAIN_MASK_SMOOTH_PX = 5


# ─── Transforms ───────────────────────────────────────────────────────────────
VAL_TRANSFORM = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

TTA_HFLIP = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=1.0),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ─── Brain masking helpers ────────────────────────────────────────────────────
def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-6:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def compute_brain_mask(image: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY) \
            if image.ndim == 3 else image.copy()
    gray8 = _to_uint8(gray)
    _, binary = cv2.threshold(gray8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.zeros(gray8.shape, dtype=np.float32)
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    min_a = 0.02 * gray8.size
    keep  = [i + 1 for i, s in enumerate(sizes) if s >= min_a] or [int(np.argmax(sizes)) + 1]
    mask  = np.isin(labeled, keep).astype(np.uint8)
    mask  = ndimage.binary_fill_holes(mask).astype(np.uint8)
    if BRAIN_MASK_DILATE_PX > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
               (BRAIN_MASK_DILATE_PX * 2 + 1, BRAIN_MASK_DILATE_PX * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    soft = mask.astype(np.float32)
    if BRAIN_MASK_SMOOTH_PX > 0:
        soft = cv2.GaussianBlur(soft,
               (BRAIN_MASK_SMOOTH_PX * 2 + 1, BRAIN_MASK_SMOOTH_PX * 2 + 1),
               BRAIN_MASK_SMOOTH_PX)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def apply_mask_to_heatmap(heatmap: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    masked = heatmap * soft_mask
    hi = masked.max()
    return (masked / hi if hi > 1e-6 else masked).astype(np.float32)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = HEATMAP_ALPHA) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    out = (1 - alpha) * image_rgb.astype(np.float32) + alpha * hm.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─── Model definition (mirrors training notebook) ────────────────────────────
class ResidualHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.norm  = nn.LayerNorm(in_dim)
        self.fc1   = nn.Linear(in_dim, 512)
        self.fc2   = nn.Linear(512, 256)
        self.fc3   = nn.Linear(256, num_classes)
        self.skip  = nn.Linear(in_dim, num_classes)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        shortcut = self.skip(x)
        x = self.drop1(F.gelu(self.fc1(x)))
        x = self.drop2(F.gelu(self.fc2(x)))
        return self.fc3(x) + shortcut


class SwinClassifier(nn.Module):
    """
    Supports both Swin-B (variant='swin_b') and Swin-S (variant='swin_s').
    """
    def __init__(self, num_classes: int = NUM_CLASSES, variant: str = "swin_b"):
        super().__init__()
        embed_dim = EMBED_DIM_MAP[variant]

        if variant == "swin_b":
            base = swin_b(weights=None)
        elif variant == "swin_s":
            base = swin_s(weights=None)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.features = base.features
        self.norm     = base.norm
        self.avgpool  = base.avgpool
        self.head     = ResidualHead(embed_dim, num_classes)
        self.variant  = variant

        self._feat_store: Optional[torch.Tensor] = None
        self._grad_store: Optional[torch.Tensor] = None
        self._hooks: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.avgpool(x.permute(0, 3, 1, 2))
        x = torch.flatten(x, 1)
        return self.head(x)

    def register_gradcam_hooks(self):
        def fwd(m, inp, out):
            self._feat_store = out.clone()
        def bwd(m, gi, go):
            self._grad_store = go[0].detach()
        self._hooks = [
            self.norm.register_forward_hook(fwd),
            self.norm.register_full_backward_hook(bwd),
        ]

    def remove_gradcam_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._feat_store = self._grad_store = None


# ─── Model loader ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, variant: str = "swin_b") -> SwinClassifier:
    """
    Load SwinClassifier from a .pt checkpoint.
    Supports:
      - raw state_dict
      - dict with key 'state_dict' or 'model_state_dict'
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        if "state_dict" in raw:
            state_dict = raw["state_dict"]
        elif "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        else:
            state_dict = raw          # assume it is already a state_dict
    else:
        state_dict = raw

    model = SwinClassifier(num_classes=NUM_CLASSES, variant=variant)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    meta = {"variant": variant, "num_classes": NUM_CLASSES, "device": str(DEVICE)}
    return model, meta


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(pil_image: Image.Image, use_tta: bool = False) -> torch.Tensor:
    """
    Returns (2, 3, H, W) if TTA, else (1, 3, H, W).
    """
    t_orig  = VAL_TRANSFORM(pil_image).unsqueeze(0)
    if use_tta:
        t_flip  = TTA_HFLIP(pil_image).unsqueeze(0)
        return torch.cat([t_orig, t_flip], dim=0).to(DEVICE)
    return t_orig.to(DEVICE)


# ─── Prediction ───────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: SwinClassifier,
            pil_image: Image.Image,
            use_tta: bool = True) -> np.ndarray:
    """
    Returns softmax probabilities [P(ET), P(PD), P(Healthy)] as float32.
    If TTA, averages orig + hflip.
    """
    tensors = preprocess(pil_image, use_tta=use_tta)  # (N, 3, H, W)
    logits  = model(tensors)                           # (N, 3)
    probs   = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs.mean(axis=0).astype(np.float32)       # (3,)


# ─── GradCAM ──────────────────────────────────────────────────────────────────
def generate_gradcam(model: SwinClassifier,
                     image,
                     class_idx: Optional[int] = None,
                     apply_brain_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (heatmap_0_1, overlay_rgb).
    Accepts a PIL Image or a numpy array (H, W) or (H, W, C).
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image
    tensor = VAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    # Get reference image for mask/overlay
    orig_np = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                       dtype=np.uint8)

    model.eval()
    model.register_gradcam_hooks()

    tensor.requires_grad_(True)
    logits   = model(tensor)
    pred_cls = int(logits.argmax(1).item())
    target   = class_idx if class_idx is not None else pred_cls

    model.zero_grad()
    logits[0, target].backward()

    feats = model._feat_store
    grads = model._grad_store
    model.remove_gradcam_hooks()

    if feats is None or grads is None:
        blank = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        return blank, orig_np

    feats = feats.detach().squeeze(0)   # (H', W', C)
    grads = grads.squeeze(0)            # (H', W', C)
    alpha = grads.mean(dim=(0, 1))      # (C,)
    cam   = torch.clamp((alpha * feats).sum(dim=-1), min=0).cpu().numpy()
    cam   = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    hi    = cam.max()
    cam   = cam / hi if hi > 1e-6 else cam

    if apply_brain_mask:
        mask = compute_brain_mask(orig_np)
        cam  = apply_mask_to_heatmap(cam, mask)
    overlay = overlay_heatmap(orig_np, cam)
    return cam.astype(np.float32), overlay


# ─── Attention Rollout ────────────────────────────────────────────────────────
class AttentionRollout:
    """Aggregates feature-activation magnitude across all SwinTransformerBlocks."""

    def __init__(self, model: SwinClassifier):
        self.model  = model
        self._maps: list = []
        self._hooks: list = []
        self._register()

    def _register(self):
        from torchvision.models.swin_transformer import SwinTransformerBlock
        def hook(m, inp, out):
            self._maps.append(out.detach().cpu())
        for m in self.model.features.modules():
            if isinstance(m, SwinTransformerBlock):
                self._hooks.append(m.register_forward_hook(hook))

    def __call__(self, pil_image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (rollout_0_1, overlay_rgb)."""
        tensor = VAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
        orig_np = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                           dtype=np.uint8)
        self._maps.clear()
        self.model.eval()
        with torch.no_grad():
            self.model(tensor)

        if not self._maps:
            blank = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
            return blank, orig_np

        aggregated = None
        for block_out in self._maps:
            mag = block_out.abs().mean(dim=-1).squeeze(0).numpy()
            lo, hi = mag.min(), mag.max()
            mag = (mag - lo) / (hi - lo + 1e-8)
            mag_r = cv2.resize(mag.astype(np.float32), (IMG_SIZE, IMG_SIZE))
            aggregated = mag_r if aggregated is None else aggregated + mag_r

        lo, hi = aggregated.min(), aggregated.max()
        rollout = ((aggregated - lo) / (hi - lo + 1e-8)).astype(np.float32)

        mask    = compute_brain_mask(orig_np)
        rollout = apply_mask_to_heatmap(rollout, mask)
        overlay = overlay_heatmap(orig_np, rollout)
        return rollout, overlay

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._maps.clear()


def generate_attention_rollout(model: SwinClassifier, image) -> np.ndarray:
    """
    Convenience wrapper around AttentionRollout.
    Accepts a PIL Image or a numpy array (H, W) or (H, W, C).
    Returns rollout heatmap (0-1 float32, shape IMG_SIZE x IMG_SIZE).
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    ar = AttentionRollout(model)
    try:
        rollout, _overlay = ar(image)
    finally:
        ar.close()
    return rollout