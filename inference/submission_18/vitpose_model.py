"""ViTPose (plain ViT backbone + classic deconvolution head) in pure PyTorch.

Used both for training on Colab and inside the submission (torch 2.4.1, no
transformers/mmpose), so it only relies on torch. Weights can be loaded from
the Hugging Face port (usyd-community/vitpose-base, COCO only) or from the
MMPose 1.x COCO checkpoints (td-hm_ViTPose-{base,large,huge}_..._coco-256x192).
ViTPose+ (mixture-of-experts, multi-dataset) checkpoints are rejected: the
Track 3 rules only allow COCO / PoseTrack18 pose pre-training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

PRESETS = {
    "base": dict(embed_dim=768, depth=12, num_heads=12),
    "large": dict(embed_dim=1024, depth=24, num_heads=16),
    "huge": dict(embed_dim=1280, depth=32, num_heads=16),
}

# PoseTrack18 slot -> COCO channel used to initialise the head (slots 1/2 have no
# COCO counterpart, start them from the nose channel).
PT18_FROM_COCO = (0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop_path: float, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.fc2(F.gelu(self.fc1(self.norm2(x)))))


class ViTPose(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 img_size: tuple[int, int] = (256, 192), patch_size: int = 16, num_keypoints: int = 17,
                 drop_path_rate: float = 0.0, deconv_channels: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.depth = depth
        self.grad_checkpoint = False  # recompute block activations in backward (saves memory at 384x288)
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size, padding=2)
        hp = (img_size[0] + 4 - patch_size) // patch_size + 1
        wp = (img_size[1] + 4 - patch_size) // patch_size + 1
        self.grid = (hp, wp)
        self.pos_embed = nn.Parameter(torch.zeros(1, hp * wp + 1, embed_dim))
        rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio, r, eps) for r in rates)
        self.last_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.deconv1 = nn.ConvTranspose2d(embed_dim, deconv_channels, 4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(deconv_channels)
        self.deconv2 = nn.ConvTranspose2d(deconv_channels, deconv_channels, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(deconv_channels)
        self.final = nn.Conv2d(deconv_channels, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.last_norm(x).transpose(1, 2).reshape(b, c, hp, wp)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        return self.final(x)

    def layer_id(self, name: str) -> int:
        """Depth index for layer-wise learning-rate decay."""
        if name.startswith(("patch_embed", "pos_embed")):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return self.depth + 1


def build(preset: str, **kwargs) -> ViTPose:
    return ViTPose(**PRESETS[preset], **kwargs)


def build_for_config(cfg: dict, **kwargs) -> ViTPose:
    """Model matching an exported model_config.json (preset + input size; old configs are 256x192)."""
    img_size = (int(cfg.get("input_height", 256)), int(cfg.get("input_width", 192)))
    return build(cfg["preset"], img_size=img_size, **kwargs)


# ---------------------------------------------------------------- weight conversion

def _head_from(src: dict, prefix: dict) -> dict:
    return {
        "deconv1.weight": src[prefix["deconv1"] + ".weight"],
        "deconv2.weight": src[prefix["deconv2"] + ".weight"],
        **{f"bn1.{k}": src[f"{prefix['bn1']}.{k}"] for k in ("weight", "bias", "running_mean", "running_var")},
        **{f"bn2.{k}": src[f"{prefix['bn2']}.{k}"] for k in ("weight", "bias", "running_mean", "running_var")},
        "final.weight": src[prefix["final"] + ".weight"],
        "final.bias": src[prefix["final"] + ".bias"],
    }


def from_hf(src: dict, depth: int) -> dict:
    """Hugging Face VitPoseForPoseEstimation (non-MoE) state dict -> ViTPose."""
    if any("experts" in k for k in src):
        raise ValueError("ViTPose+ (MoE) checkpoint: not allowed by the Track 3 rules")
    out = {
        "patch_embed.weight": src["backbone.embeddings.patch_embeddings.projection.weight"],
        "patch_embed.bias": src["backbone.embeddings.patch_embeddings.projection.bias"],
        "pos_embed": src["backbone.embeddings.position_embeddings"],
        "last_norm.weight": src["backbone.layernorm.weight"],
        "last_norm.bias": src["backbone.layernorm.bias"],
    }
    for i in range(depth):
        p = f"backbone.encoder.layer.{i}."
        a = p + "attention.attention."
        out[f"blocks.{i}.attn.qkv.weight"] = torch.cat([src[a + "query.weight"], src[a + "key.weight"], src[a + "value.weight"]])
        out[f"blocks.{i}.attn.qkv.bias"] = torch.cat([src[a + "query.bias"], src[a + "key.bias"], src[a + "value.bias"]])
        out[f"blocks.{i}.attn.proj.weight"] = src[p + "attention.output.dense.weight"]
        out[f"blocks.{i}.attn.proj.bias"] = src[p + "attention.output.dense.bias"]
        for mine, theirs in (("norm1", "layernorm_before"), ("norm2", "layernorm_after"), ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2")):
            out[f"blocks.{i}.{mine}.weight"] = src[p + theirs + ".weight"]
            out[f"blocks.{i}.{mine}.bias"] = src[p + theirs + ".bias"]
    out.update(_head_from(src, {"deconv1": "head.deconv1", "deconv2": "head.deconv2", "bn1": "head.batchnorm1",
                                "bn2": "head.batchnorm2", "final": "head.conv"}))
    return out


def from_mmpose(src: dict, depth: int) -> dict:
    """MMPose 1.x ViTPose COCO checkpoint (mmpretrain VisionTransformer backbone + HeatmapHead) -> ViTPose."""
    src = src.get("state_dict", src)
    if any("experts" in k for k in src):
        raise ValueError("ViTPose+ (MoE) checkpoint: not allowed by the Track 3 rules")
    out = {
        "patch_embed.weight": src["backbone.patch_embed.projection.weight"],
        "patch_embed.bias": src["backbone.patch_embed.projection.bias"],
        "pos_embed": src["backbone.pos_embed"],
        "last_norm.weight": src["backbone.ln1.weight"] if "backbone.ln1.weight" in src else src["backbone.pre_norm.weight"],
        "last_norm.bias": src["backbone.ln1.bias"] if "backbone.ln1.bias" in src else src["backbone.pre_norm.bias"],
    }
    for i in range(depth):
        p = f"backbone.layers.{i}."
        for mine, theirs in (("norm1", "ln1"), ("norm2", "ln2"), ("attn.qkv", "attn.qkv"), ("attn.proj", "attn.proj"),
                             ("fc1", "ffn.layers.0.0"), ("fc2", "ffn.layers.1")):
            out[f"blocks.{i}.{mine}.weight"] = src[p + theirs + ".weight"]
            out[f"blocks.{i}.{mine}.bias"] = src[p + theirs + ".bias"]
    out.update(_head_from(src, {"deconv1": "head.deconv_layers.0", "deconv2": "head.deconv_layers.3",
                                "bn1": "head.deconv_layers.1", "bn2": "head.deconv_layers.4", "final": "head.final_layer"}))
    return out


def coco_head_to_pt18(state: dict) -> dict:
    """Re-order the final 1x1 conv from COCO-17 channels to PoseTrack18-17 slots."""
    idx = torch.tensor(PT18_FROM_COCO)
    state = dict(state)
    state["final.weight"] = state["final.weight"][idx].clone()
    state["final.bias"] = state["final.bias"][idx].clone()
    return state


def resize_pos_embed(pos_embed: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Bicubic resize of the patch position embeddings (keeps the extra first token)."""
    cls, patches = pos_embed[:, :1], pos_embed[:, 1:]
    n = patches.shape[1]
    if n == grid[0] * grid[1]:
        return pos_embed
    old = int(round(math.sqrt(n * 4 / 3))), int(round(math.sqrt(n * 3 / 4)))  # 256x192 grids are 4:3
    if old[0] * old[1] != n:
        raise ValueError(f"cannot infer pos_embed grid from {n} tokens")
    patches = patches.reshape(1, old[0], old[1], -1).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=grid, mode="bicubic", align_corners=False)
    return torch.cat([cls, patches.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1)], dim=1)


def load_checkpoint_file(path: str) -> dict:
    """Load a .pth checkpoint without executing arbitrary pickle code.

    MMPose checkpoints (legacy torch format) pickle a few numpy objects in their
    metadata, which torch's weights_only loader rejects under numpy 2. A restricted
    unpickler admits exactly the tensor-rebuild, OrderedDict and numpy array/dtype
    globals and refuses everything else."""
    import collections
    import importlib
    import pickle
    import types

    allowed_modules = {"numpy.core.multiarray": "_reconstruct", "numpy._core.multiarray": "_reconstruct"}
    allowed = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
               ("torch._utils", "_rebuild_parameter"), ("numpy", "ndarray"), ("numpy", "dtype"),
               ("_codecs", "encode")}

    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch" and name.endswith("Storage"):
                return getattr(importlib.import_module("torch"), name)
            if module == "numpy.dtypes" and name.endswith("DType"):
                return getattr(importlib.import_module("numpy.dtypes"), name)
            if allowed_modules.get(module) == name or (module, name) in allowed:
                if module.startswith("numpy") and "core" in module:
                    try:
                        return importlib.import_module("numpy._core.multiarray")._reconstruct
                    except ImportError:
                        return importlib.import_module("numpy.core.multiarray")._reconstruct
                if module == "collections":
                    return collections.OrderedDict
                return getattr(importlib.import_module(module), name)
            raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")

    safe_pickle = types.SimpleNamespace(
        __name__="safe_pickle", Unpickler=SafeUnpickler,
        load=lambda f, **kw: SafeUnpickler(f, **kw).load())
    return torch.load(path, map_location="cpu", pickle_module=safe_pickle, weights_only=False)
