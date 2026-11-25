"""Top-down heatmap pose inference shared by the Colab experiments and run.py.

numpy only (plus a model callable), so it runs unchanged in the evaluation
image (torch 2.4.1, numpy, Pillow; no OpenCV/scipy). Preprocessing and DARK
decoding follow the organizers' example submission; the Gaussian blur is
vectorised and flip test-time augmentation is added.

The model is passed in as `forward(batch) -> heatmaps`, with batch a float32
array (B, 3, H, W) and heatmaps a float32 array (B, K, h, w).
"""
from __future__ import annotations

import numpy as np

# COCO-17 left/right pairs (nose=0, eyes 1/2, ears 3/4, shoulders 5/6, ...).
COCO_FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
# PoseTrack18-17: nose, head_bottom and head_top are centre joints.
PT18_FLIP_PAIRS = ((3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
# COCO joints whose positions predict head_bottom / head_top.
HEAD_FEATURE_JOINTS = (0, 1, 2, 3, 4, 5, 6)


# ---------------------------------------------------------------- preprocessing

def box_to_center_scale(box, out_w: int, out_h: int, normalize_factor: float = 200.0,
                        padding_factor: float = 1.25):
    x, y, w, h = [float(v) for v in box[:4]]
    aspect = out_w / out_h
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w / normalize_factor, h / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor


def warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    size_input = center * 2.0
    size_dst = np.array([out_w, out_h], dtype=np.float32) - 1.0
    size_target = scale * 200.0
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] + 0.5 * size_target[0])
    matrix[1, 1] = scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[1] + 0.5 * size_target[1])
    return matrix


def warp_affine(image: np.ndarray, matrix: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear warp with zero fill outside the image (numpy stand-in for cv2.warpAffine)."""
    bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    inverse = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32), np.arange(out_w, dtype=np.float32), indexing="ij")
    src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]

    height, width = image.shape[:2]
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = src_x - x0, src_y - y0
    inside = (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    top = image[y0c, x0c] * (1 - wx)[..., None] + image[y0c, x1c] * wx[..., None]
    bot = image[y1c, x0c] * (1 - wx)[..., None] + image[y1c, x1c] * wx[..., None]
    return (top * (1 - wy)[..., None] + bot * wy[..., None]) * inside[..., None]


def preprocess(image: np.ndarray, box, cfg: dict):
    """image: HxWx3 float32 RGB in [0, 255]. Returns (CHW tensor, center, scale)."""
    out_w, out_h = int(cfg["input_width"]), int(cfg["input_height"])
    center, scale = box_to_center_scale(box, out_w, out_h, float(cfg.get("normalize_factor", 200.0)),
                                        float(cfg.get("padding_factor", 1.25)))
    crop = warp_affine(image, warp_matrix(center, scale, out_w, out_h), out_h, out_w)
    mean = np.asarray(cfg["image_mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(cfg["image_std"], dtype=np.float32).reshape(3, 1, 1)
    tensor = crop.transpose(2, 0, 1) * float(cfg.get("rescale_factor", 1.0 / 255.0))
    return ((tensor - mean) / std).astype(np.float32), center, scale


# ---------------------------------------------------------------- decoding

def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 / (sigma * sigma) * x * x)
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_blur(heatmaps: np.ndarray, kernel_size: int = 11, sigma: float = 0.8) -> np.ndarray:
    """Separable blur over the last two axes with symmetric padding (vectorised)."""
    radius = (kernel_size - 1) // 2
    kernel = _gaussian_kernel1d(sigma, radius)
    h, w = heatmaps.shape[-2:]
    padded = np.pad(heatmaps, ((0, 0), (0, 0), (radius, radius), (radius, radius)), mode="symmetric")
    rows = sum(kernel[k] * padded[..., :, k:k + w] for k in range(kernel_size))
    return sum(kernel[k] * rows[..., k:k + h, :] for k in range(kernel_size))


def decode_heatmaps(heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray, dark_kernel: int = 11):
    """Argmax + DARK refinement. Returns image coords (B, K, 2) and max values (B, K)."""
    batch, num_kpts, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, num_kpts, -1)
    idx = np.argmax(flat, axis=2)
    scores = np.max(flat, axis=2)
    coords = np.zeros((batch, num_kpts, 2), dtype=np.float32)
    coords[:, :, 0] = idx % width
    coords[:, :, 1] = idx // width

    blurred = np.clip(gaussian_blur(heatmaps, dark_kernel), 0.001, 50.0)
    padded = np.pad(np.log(blurred), ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").reshape(-1)
    stride = (width + 2) * (height + 2)
    base = coords[..., 0] + 1 + (coords[..., 1] + 1) * (width + 2)
    base = base + stride * np.arange(batch * num_kpts).reshape(batch, num_kpts)
    index = base.astype(np.int64).reshape(-1, 1)

    center_v = padded[index]
    x_plus, x_minus = padded[index + 1], padded[index - 1]
    y_plus, y_minus = padded[index + width + 2], padded[index - 2 - width]
    xy_plus, xy_minus = padded[index + width + 3], padded[index - width - 3]
    dx = 0.5 * (x_plus - x_minus)
    dy = 0.5 * (y_plus - y_minus)
    derivative = np.concatenate([dx, dy], axis=1).reshape(batch, num_kpts, 2, 1)
    dxx = x_plus - 2 * center_v + x_minus
    dyy = y_plus - 2 * center_v + y_minus
    dxy = 0.5 * (xy_plus - x_plus - y_plus + center_v + center_v - x_minus - y_minus + xy_minus)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(batch, num_kpts, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords = coords - np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze(-1)

    out = np.zeros_like(coords)
    for i in range(batch):
        size = scales[i] * 200.0
        out[i, :, 0] = coords[i, :, 0] * size[0] / (width - 1.0) + centers[i][0] - size[0] * 0.5
        out[i, :, 1] = coords[i, :, 1] * size[1] / (height - 1.0) + centers[i][1] - size[1] * 0.5
    return out, scores


def flip_back(heatmaps: np.ndarray, flip_pairs, shift: bool = False) -> np.ndarray:
    """Undo a horizontal input flip on heatmaps: mirror and swap left/right channels.
    ViTPose uses UDP, so no one-pixel shift (MMPose: shift_heatmap=False); `shift=True`
    is only for non-UDP models. Measured on the mock test: no shift 0.8866, shift 0.8700."""
    out = heatmaps[..., ::-1].copy()
    for a, b in flip_pairs:
        out[:, [a, b]] = out[:, [b, a]]
    if shift:
        out[..., 1:] = out[..., :-1].copy()
    return out


def predict_boxes(image: np.ndarray, boxes, forward, cfg: dict, flip: bool = False,
                  flip_pairs=COCO_FLIP_PAIRS, batch_size: int = 16, flip_shift: bool = False):
    """Run the model on every usable box. Returns (keypoints (N, K, 2), joint_scores (N, K),
    usable mask (N,)). Boxes with w or h <= 0 are left as NaN for the caller to fill."""
    num_kpts = int(cfg["num_keypoints"])
    n = len(boxes)
    keypoints = np.full((n, num_kpts, 2), np.nan, dtype=np.float32)
    joint_scores = np.zeros((n, num_kpts), dtype=np.float32)
    usable = np.array([float(b[2]) > 0 and float(b[3]) > 0 for b in boxes], dtype=bool)
    order = np.flatnonzero(usable)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        tensors, centers, scales = zip(*(preprocess(image, boxes[i], cfg) for i in chunk))
        batch = np.stack(tensors)
        heatmaps = forward(batch)
        if flip:
            heatmaps = 0.5 * (heatmaps + flip_back(forward(np.ascontiguousarray(batch[..., ::-1])), flip_pairs, flip_shift))
        coords, maxvals = decode_heatmaps(heatmaps, np.stack(centers), np.stack(scales))
        keypoints[chunk] = coords
        joint_scores[chunk] = maxvals
    return keypoints, joint_scores, usable



def _corner_aligned_matrix(n_out: int, n_in: int) -> np.ndarray:
    pos = np.arange(n_out, dtype=np.float64) * (n_in - 1) / max(n_out - 1, 1)
    lo = np.clip(np.floor(pos).astype(np.int64), 0, n_in - 2)
    frac = (pos - lo).astype(np.float32)
    m = np.zeros((n_out, n_in), dtype=np.float32)
    m[np.arange(n_out), lo] = 1.0 - frac
    m[np.arange(n_out), lo + 1] += frac
    return m


def resize_heatmaps(heatmaps: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize with aligned corners. decode_heatmaps maps heatmap pixel 0 and w-1 to
    the crop edges, so heatmaps of the same crop at different resolutions (e.g. 64x48 from a
    256x192 model, 96x72 from a 384x288 one) line up after this resize."""
    h, w = heatmaps.shape[-2:]
    if (h, w) == (height, width):
        return heatmaps
    return np.einsum("yi,bkij,xj->bkyx", _corner_aligned_matrix(height, h), heatmaps,
                     _corner_aligned_matrix(width, w), optimize=True).astype(np.float32)


def predict_boxes_ensemble(image: np.ndarray, boxes, members, flip: bool = False,
                           flip_pairs=COCO_FLIP_PAIRS, batch_size: int = 16, flip_shift: bool = False):
    """predict_boxes for a heatmap ensemble. members: list of (forward, cfg, weight); the configs
    may differ in input size but must share aspect ratio and padding, so every member sees the
    same crop. Each member's (flip-averaged) heatmaps are resized to the largest heatmap size,
    weighted, summed and decoded once."""
    cfgs = [cfg for _, cfg, _ in members]
    num_kpts = int(cfgs[0]["num_keypoints"])
    n = len(boxes)
    keypoints = np.full((n, num_kpts, 2), np.nan, dtype=np.float32)
    joint_scores = np.zeros((n, num_kpts), dtype=np.float32)
    usable = np.array([float(b[2]) > 0 and float(b[3]) > 0 for b in boxes], dtype=bool)
    order = np.flatnonzero(usable)
    out_h = max(int(c["input_height"]) // 4 for c in cfgs)
    out_w = max(int(c["input_width"]) // 4 for c in cfgs)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        total, centers, scales = None, None, None
        for forward, cfg, weight in members:
            tensors, c, s = zip(*(preprocess(image, boxes[i], cfg) for i in chunk))
            if centers is None:
                centers, scales = np.stack(c), np.stack(s)
            elif not (np.allclose(centers, np.stack(c)) and np.allclose(scales, np.stack(s))):
                raise ValueError("ensemble members crop different regions (aspect ratio / padding differ)")
            batch = np.stack(tensors)
            heatmaps = forward(batch)
            if flip:
                heatmaps = 0.5 * (heatmaps + flip_back(forward(np.ascontiguousarray(batch[..., ::-1])), flip_pairs, flip_shift))
            heatmaps = float(weight) * resize_heatmaps(heatmaps, out_h, out_w)
            total = heatmaps if total is None else total + heatmaps
        coords, maxvals = decode_heatmaps(total, centers, scales)
        keypoints[chunk] = coords
        joint_scores[chunk] = maxvals
    return keypoints, joint_scores, usable

# ---------------------------------------------------------------- COCO -> PoseTrack18

def head_features(coco_kpts: np.ndarray, box) -> np.ndarray:
    """Box-normalised (u, v) of the head/shoulder COCO joints + bias. coco_kpts: (17, 2)."""
    x, y, w, h = [float(v) for v in box[:4]]
    pts = coco_kpts[list(HEAD_FEATURE_JOINTS)]
    uv = np.stack([(pts[:, 0] - x) / w, (pts[:, 1] - y) / h], axis=1).reshape(-1)
    return np.concatenate([uv, [1.0]]).astype(np.float64)


def coco_to_pt18(coco_kpts: np.ndarray, coco_scores: np.ndarray, box, mapping: dict | None):
    """Map a COCO-17 prediction to the PoseTrack18 slot order. Slots 1/2 (eyes in COCO)
    become head_bottom/head_top predicted by a linear model on head/shoulder joints.
    Without a mapping the eyes are copied through (the organizers' baseline behaviour)."""
    kpts = coco_kpts.astype(np.float64).copy()
    scores = coco_scores.astype(np.float64).copy()
    if mapping is not None:
        x, y, w, h = [float(v) for v in box[:4]]
        f = head_features(coco_kpts, box)
        feature_score = float(np.mean(coco_scores[list(HEAD_FEATURE_JOINTS)]))
        for slot, name in ((1, "head_bottom"), (2, "head_top")):
            coef_u, coef_v = mapping[name]
            kpts[slot] = [x + float(f @ np.asarray(coef_u)) * w, y + float(f @ np.asarray(coef_v)) * h]
            scores[slot] = feature_score
    return kpts, scores


OKS_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], dtype=np.float64)
SCORED_JOINTS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
HEAD_JOINTS = [0, 1, 2]
INV_SIGMA = 1.0 / OKS_SIGMAS[SCORED_JOINTS]
WEIGHTS_SIGMA = INV_SIGMA / INV_SIGMA.sum()


def person_score(joint_scores: np.ndarray, mode: str = "mean_scored", threshold: float = 0.2,
                 ignore=(3, 4), gamma: float = 0.15) -> float:
    """Per-person confidence used to rank detections in OKS-AP."""
    s = np.asarray(joint_scores, dtype=np.float64)
    if mode == "mean_all":
        return float(s.mean())
    if mode == "calibrated_oks":
        scored_scores = s[SCORED_JOINTS]
        weighted_mean = float(np.sum(scored_scores * WEIGHTS_SIGMA))
        if gamma > 0.0:
            min_head = float(np.clip(np.min(s[HEAD_JOINTS]), 1e-4, 1.0))
            return weighted_mean * (min_head ** gamma)
        return weighted_mean
    keep = np.ones(len(s), dtype=bool)
    keep[list(ignore)] = False
    s = s[keep]
    if mode == "mean_scored":
        return float(s.mean())
    if mode == "mean_thresh":
        good = s[s > threshold]
        return float(good.mean()) if len(good) else float(s.mean())
    raise ValueError(f"unknown score mode {mode}")
