"""Training data for top-down PoseTrack18 heatmap regression (runs on Colab).

Augmentation follows MMPose's ViTPose COCO recipe (random flip, half body,
bbox scale/shift/rotation, UDP affine, UDP Gaussian heatmaps sigma 2) plus the
organizers' cross-domain DA (Cormier, KIT 2026, Sec. 6.2): photometric
distortion and channel swap (p=0.5 each), Gaussian/median blur (p=0.1), one
coarse-dropout hole up to 40% of height/width. Grayscale and down/up-sampling
are added for unknown target domains.

Training uses OpenCV (available on Colab); the submission does not.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

PT18_FLIP_PAIRS = ((3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
UPPER_BODY = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
IGNORED_SLOTS = (3, 4)  # ears: not annotated, not scored

DEFAULT_AUG = dict(
    flip=0.5, half_body=0.3, scale=(0.75, 1.25), shift=0.16, rotate=40.0, rotate_p=0.6,
    photometric=0.5, blur=0.1, median_blur=0.1, dropout=1.0, dropout_max=0.4,
    grayscale=0.1, lowres=0.2,
)


def udp_warp_matrix(center, size, rot_deg: float, out_w: int, out_h: int) -> np.ndarray:
    """MMPose get_udp_warp_matrix: maps the (rotated) box of `size` around `center`
    onto the (out_w-1) x (out_h-1) input grid."""
    theta = math.radians(rot_deg)
    sx = (out_w - 1.0) / size[0]
    sy = (out_h - 1.0) / size[1]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c * sx, -s * sx, sx * (-center[0] * c + center[1] * s + 0.5 * size[0])],
        [s * sy, c * sy, sy * (-center[0] * s - center[1] * c + 0.5 * size[1])],
    ], dtype=np.float32)


def box_center_size(box, aspect: float, padding: float = 1.25):
    x, y, w, h = [float(v) for v in box[:4]]
    center = np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], dtype=np.float32) * padding


def udp_heatmaps(kpts_in: np.ndarray, weights: np.ndarray, in_size, hm_size, sigma: float = 2.0):
    """Gaussian targets at the exact (sub-pixel) UDP heatmap location."""
    w_hm, h_hm = hm_size
    feat = np.array([(w_hm - 1.0) / (in_size[0] - 1.0), (h_hm - 1.0) / (in_size[1] - 1.0)], dtype=np.float32)
    mu = kpts_in * feat
    xs = np.arange(w_hm, dtype=np.float32)[None, None, :]
    ys = np.arange(h_hm, dtype=np.float32)[None, :, None]
    hm = np.exp(-((xs - mu[:, 0, None, None]) ** 2 + (ys - mu[:, 1, None, None]) ** 2) / (2 * sigma ** 2))
    inside = (mu[:, 0] >= -0.5) & (mu[:, 0] <= w_hm - 0.5) & (mu[:, 1] >= -0.5) & (mu[:, 1] <= h_hm - 0.5)
    weights = weights * inside
    return (hm * weights[:, None, None]).astype(np.float32), weights.astype(np.float32)


def _photometric(img: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    img = img.astype(np.float32)
    if rng.random() < p:
        img += rng.uniform(-32, 32)
    contrast_first = rng.random() < 0.5
    if contrast_first and rng.random() < p:
        img *= rng.uniform(0.5, 1.5)
    if rng.random() < p or rng.random() < p:
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        if rng.random() < p:
            hsv[..., 1] *= rng.uniform(0.5, 1.5)
        if rng.random() < p:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-9, 9)) % 180  # OpenCV hue is 0..179 (= +/-18 deg)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if not contrast_first and rng.random() < p:
        img *= rng.uniform(0.5, 1.5)
    if rng.random() < p:
        img = img[..., rng.permutation(3)]
    return np.clip(img, 0, 255).astype(np.uint8)


class PoseDataset(torch.utils.data.Dataset):
    """images: list of HxWx3 uint8 RGB arrays (kept in RAM, shared with fork workers).
    anns: list of dicts with 'image_index', 'bbox' and 'keypoints' (51 floats, PT18 order)."""

    def __init__(self, images, anns, cfg: dict, train: bool, aug: dict | None = None, seed: int = 0,
                 sigma: float = 2.0) -> None:
        self.images, self.anns, self.cfg, self.train = images, anns, cfg, train
        self.aug = {**DEFAULT_AUG, **(aug or {})}
        self.in_w, self.in_h = int(cfg["input_width"]), int(cfg["input_height"])
        self.hm_w, self.hm_h = self.in_w // 4, self.in_h // 4
        self.mean = np.asarray(cfg["image_mean"], dtype=np.float32) * 255.0
        self.std = np.asarray(cfg["image_std"], dtype=np.float32) * 255.0
        self.seed = seed
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, i: int):
        ann = self.anns[i]
        img = self.images[ann["image_index"]]
        kpts = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        vis = (kpts[:, 2] > 0).astype(np.float32)
        vis[list(IGNORED_SLOTS)] = 0.0
        aspect = self.in_w / self.in_h
        center, size = box_center_size(ann["bbox"], aspect)
        rot, flip = 0.0, False
        rng = np.random.default_rng() if self.train else None

        if self.train:
            a = self.aug
            if rng.random() < a["half_body"] and vis.sum() >= 8:
                part = [j for j in (UPPER_BODY if rng.random() < 0.5 else range(11, 17)) if vis[j] > 0]
                if len(part) >= 2:
                    pts = kpts[part, :2]
                    lo, hi = pts.min(0), pts.max(0)
                    center = (lo + hi) / 2
                    w, h = hi - lo
                    w, h = (h * aspect, h) if w < aspect * h else (w, w / aspect)
                    size = np.array([w, h], dtype=np.float32) * 1.5
            size = size * rng.uniform(*a["scale"])
            center = center + size * rng.uniform(-a["shift"], a["shift"], 2) * 0.5
            if rng.random() < a["rotate_p"]:
                rot = float(np.clip(rng.normal(0, a["rotate"] / 2), -a["rotate"], a["rotate"]))
            flip = rng.random() < a["flip"]

        m = udp_warp_matrix(center, size, rot, self.in_w, self.in_h)
        crop = cv2.warpAffine(img, m, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        k_in = kpts[:, :2] @ m[:, :2].T + m[:, 2]
        if flip:
            crop = crop[:, ::-1]
            k_in[:, 0] = (self.in_w - 1) - k_in[:, 0]
            for l, r in PT18_FLIP_PAIRS:
                k_in[[l, r]] = k_in[[r, l]]
                vis[[l, r]] = vis[[r, l]]

        if self.train:
            crop = self._appearance(np.ascontiguousarray(crop), rng)
        target, weight = udp_heatmaps(k_in, vis, (self.in_w, self.in_h), (self.hm_w, self.hm_h), self.sigma)
        x = ((crop.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1)
        meta = {"center": center, "size": size, "index": i}
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(target), torch.from_numpy(weight), meta

    def _appearance(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        a = self.aug
        crop = _photometric(crop, rng, a["photometric"])
        if rng.random() < a["grayscale"]:
            g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            crop = np.repeat(g[..., None], 3, axis=2)
        if rng.random() < a["lowres"]:
            f = rng.uniform(2.0, 4.0)
            small = cv2.resize(crop, (max(8, int(self.in_w / f)), max(8, int(self.in_h / f))), interpolation=cv2.INTER_AREA)
            crop = cv2.resize(small, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < a["blur"]:
            crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.5, 2.0))
        if rng.random() < a["median_blur"]:
            crop = cv2.medianBlur(crop, int(rng.choice([3, 5])))
        if rng.random() < a["dropout"]:
            hh = int(rng.uniform(0.05, a["dropout_max"]) * self.in_h)
            ww = int(rng.uniform(0.05, a["dropout_max"]) * self.in_w)
            y0 = int(rng.integers(0, self.in_h - hh + 1))
            x0 = int(rng.integers(0, self.in_w - ww + 1))
            crop = crop.copy()
            crop[y0:y0 + hh, x0:x0 + ww] = 0
        return crop


def collate(batch):
    xs, ts, ws, metas = zip(*batch)
    return (torch.stack(xs), torch.stack(ts), torch.stack(ws),
            {"center": np.stack([m["center"] for m in metas]), "size": np.stack([m["size"] for m in metas]),
             "index": np.array([m["index"] for m in metas])})
