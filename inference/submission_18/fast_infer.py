"""Device-side (torch) version of pose_infer.predict_boxes_ensemble for the submission.

The numpy pipeline spends most of the per-image time on the CPU: the bilinear crop
(~12 ms at 256x192, ~22 ms at 384x288) and the heatmap resize (~10 ms). Here the crop,
both flip views (one batch per member instead of two forward calls), the flip-back and
the corner-aligned resize run on the model's device; decoding stays in numpy
(pose_infer.decode_heatmaps). Same formulas as pose_infer, so the output matches the numpy
path up to float rounding (tests/test_fast_infer.py). Only torch + numpy are needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import pose_infer as pi


def warp_affine(image: torch.Tensor, matrix: np.ndarray, out_h: int, out_w: int) -> torch.Tensor:
    """pose_infer.warp_affine on a (H, W, 3) float tensor: bilinear, zero outside the image."""
    bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    inv = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    dev = image.device
    ys, xs = torch.meshgrid(torch.arange(out_h, dtype=torch.float32, device=dev),
                            torch.arange(out_w, dtype=torch.float32, device=dev), indexing="ij")
    src_x = float(inv[0, 0]) * xs + float(inv[0, 1]) * ys + float(inv[0, 2])
    src_y = float(inv[1, 0]) * xs + float(inv[1, 1]) * ys + float(inv[1, 2])
    height, width = image.shape[:2]
    x0, y0 = torch.floor(src_x), torch.floor(src_y)
    wx, wy = (src_x - x0)[..., None], (src_y - y0)[..., None]
    inside = ((src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1))[..., None]
    x0c, x1c = x0.long().clamp(0, width - 1), (x0.long() + 1).clamp(0, width - 1)
    y0c, y1c = y0.long().clamp(0, height - 1), (y0.long() + 1).clamp(0, height - 1)
    top = image[y0c, x0c] * (1 - wx) + image[y0c, x1c] * wx
    bot = image[y1c, x0c] * (1 - wx) + image[y1c, x1c] * wx
    return (top * (1 - wy) + bot * wy) * inside


def _flip_perm(num_kpts: int, flip_pairs) -> list[int]:
    perm = list(range(num_kpts))
    for a, b in flip_pairs:
        perm[a], perm[b] = b, a
    return perm


def build_gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device,
                             dtype: torch.dtype = torch.float32) -> torch.Tensor:
    radius = (kernel_size - 1) // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k.to(dtype)


def pad_symmetric_2d(t: torch.Tensor, radius: int) -> torch.Tensor:
    left_w = torch.flip(t[..., :radius], dims=[-1])
    right_w = torch.flip(t[..., -radius:], dims=[-1])
    t_w = torch.cat([left_w, t, right_w], dim=-1)
    top_h = torch.flip(t_w[..., :radius, :], dims=[-2])
    bot_h = torch.flip(t_w[..., -radius:, :], dims=[-2])
    return torch.cat([top_h, t_w, bot_h], dim=-2)


def gaussian_blur_2d(heatmaps: torch.Tensor, kernel_size: int = 11, sigma: float = 1.8) -> torch.Tensor:
    if kernel_size <= 1:
        return heatmaps
    radius = (kernel_size - 1) // 2
    b, k, h, w = heatmaps.shape
    device = heatmaps.device
    k1d = build_gaussian_kernel_1d(kernel_size, sigma, device, heatmaps.dtype)
    padded = pad_symmetric_2d(heatmaps, radius)
    k_row = k1d.view(1, 1, 1, kernel_size).repeat(k, 1, 1, 1)
    k_col = k1d.view(1, 1, kernel_size, 1).repeat(k, 1, 1, 1)
    row_conv = F.conv2d(padded, k_row, groups=k)
    return F.conv2d(row_conv, k_col, groups=k)


@torch.no_grad()
def decode_heatmaps_dark(heatmaps: torch.Tensor, centers, scales, kernel_size: int = 11,
                         sigma: float = 1.8, clamp_shift: float | None = 0.75) -> tuple[torch.Tensor, torch.Tensor]:
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype

    if isinstance(centers, np.ndarray):
        centers_t = torch.from_numpy(centers).to(device=device, dtype=dtype)
    else:
        centers_t = centers.to(device=device, dtype=dtype)

    if isinstance(scales, np.ndarray):
        scales_t = torch.from_numpy(scales).to(device=device, dtype=dtype)
    else:
        scales_t = scales.to(device=device, dtype=dtype)

    flat = heatmaps.view(B, K, -1)
    max_idx = flat.argmax(dim=-1)
    maxvals = flat.gather(dim=-1, index=max_idx.unsqueeze(-1))

    x0 = (max_idx % W).float()
    y0 = (max_idx // W).float()
    coords = torch.stack([x0, y0], dim=-1)

    blurred = gaussian_blur_2d(heatmaps, kernel_size=kernel_size, sigma=sigma)
    blurred = blurred.clamp(min=1e-3, max=50.0)
    log_blurred = torch.log(blurred)
    padded = F.pad(log_blurred, (1, 1, 1, 1), mode="replicate")

    stride = (W + 2) * (H + 2)
    b_idx = torch.arange(B * K, device=device).view(B, K)
    base = (x0.long() + 1) + (y0.long() + 1) * (W + 2) + stride * b_idx
    flat_padded = padded.reshape(-1)

    c_v = flat_padded[base]
    x_p = flat_padded[base + 1]
    x_m = flat_padded[base - 1]
    y_p = flat_padded[base + (W + 2)]
    y_m = flat_padded[base - (W + 2)]
    xy_p = flat_padded[base + (W + 3)]
    xy_m = flat_padded[base - (W + 3)]

    dx = 0.5 * (x_p - x_m)
    dy = 0.5 * (y_p - y_m)
    dxx = x_p - 2.0 * c_v + x_m
    dyy = y_p - 2.0 * c_v + y_m
    dxy = 0.5 * (xy_p - x_p - y_p + c_v + c_v - x_m - y_m + xy_m)

    eps = float(np.finfo(np.float32).eps)
    h11 = dxx + eps
    h22 = dyy + eps
    h12 = dxy
    det = h11 * h22 - h12 * h12

    det_safe = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    inv_det = 1.0 / det_safe

    shift_x = inv_det * (h22 * dx - h12 * dy)
    shift_y = inv_det * (-h12 * dx + h11 * dy)

    if clamp_shift is not None and clamp_shift > 0.0:
        shift_x = shift_x.clamp(-clamp_shift, clamp_shift)
        shift_y = shift_y.clamp(-clamp_shift, clamp_shift)

    coords_hm_x = coords[..., 0] - shift_x
    coords_hm_y = coords[..., 1] - shift_y

    size = scales_t * 200.0
    coords_img_x = coords_hm_x * size[:, 0:1] / (W - 1.0) + centers_t[:, 0:1] - size[:, 0:1] * 0.5
    coords_img_y = coords_hm_y * size[:, 1:2] / (H - 1.0) + centers_t[:, 1:2] - size[:, 1:2] * 0.5
    out_coords = torch.stack([coords_img_x, coords_img_y], dim=-1)

    return out_coords, maxvals


@torch.no_grad()
def predict_boxes_ensemble(image: np.ndarray, boxes, members, flip: bool = False, flip_pairs=pi.COCO_FLIP_PAIRS,
                           batch_size: int = 16, half: bool = False,
                           padding_scales: tuple | list = (1.0,), dark_kernel: int = 11, dark_sigma: float = 1.8,
                           clamp_shift: float | None = 0.75, scale_weights=None):
    """Drop-in for pose_infer.predict_boxes_ensemble. members: list of (model, cfg, weight) with
    the torch model already on its device (all members on the same one). half=True runs the
    model under fp16 autocast (CUDA only); crop, resize and decode stay in fp32.
    padding_scales: multipliers of cfg padding_factor. Either a single tuple/list of scales
    shared by all members (e.g. (0.94, 1.0, 1.06)), or a tuple/list of tuples giving scales
    per member (e.g. ((0.88, 0.94, 1.0, 1.06), (0.94, 1.0, 1.06))).
    dark_kernel: DARK blur kernel size. dark_sigma: Gaussian blur standard deviation."""
    cfgs = [cfg for _, cfg, _ in members]
    if padding_scales and isinstance(padding_scales[0], (tuple, list)):
        if len(padding_scales) != len(members):
            raise ValueError(f"expected {len(members)} scale lists, got {len(padding_scales)}")
        member_scales = [tuple(float(s) for s in ms) for ms in padding_scales]
    else:
        common = tuple(float(s) for s in padding_scales)
        member_scales = [common for _ in members]

    unique_scales = []
    for ms in member_scales:
        for s in ms:
            if s not in unique_scales:
                unique_scales.append(s)

    if scale_weights is not None:
        if len(scale_weights) != len(unique_scales):
            raise ValueError(f"expected {len(unique_scales)} scale weights, got {len(scale_weights)}")
        sw_dict = {s: float(w) for s, w in zip(unique_scales, scale_weights)}
    else:
        sw_dict = {
            s: sum(float(w) for (_, _, w), ms in zip(members, member_scales) if s in ms)
            for s in unique_scales
        }
    total_sw = sum(sw_dict.values())
    sw_norm = {s: sw_dict[s] / total_sw for s in unique_scales}

    num_kpts = int(cfgs[0]["num_keypoints"])
    device = next(members[0][0].parameters()).device
    n = len(boxes)
    keypoints = np.full((n, num_kpts, 2), np.nan, dtype=np.float32)
    joint_scores = np.zeros((n, num_kpts), dtype=np.float32)
    usable = np.array([float(b[2]) > 0 and float(b[3]) > 0 for b in boxes], dtype=bool)
    order = np.flatnonzero(usable)
    if not len(order):
        return keypoints, joint_scores, usable
    out_h = max(int(c["input_height"]) // 4 for c in cfgs)
    out_w = max(int(c["input_width"]) // 4 for c in cfgs)
    perm = _flip_perm(num_kpts, flip_pairs)
    img = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).to(device)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=half and device.type == "cuda")
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        coords, maxvals = 0.0, 0.0
        for ps in unique_scales:
            wps = sw_norm[ps]
            total, centers, scales = None, None, None
            active = [(m, c, w) for (m, c, w), ms in zip(members, member_scales) if ps in ms]
            w_active_sum = sum(float(w) for _, _, w in active)
            for model, cfg, weight in active:
                in_w, in_h = int(cfg["input_width"]), int(cfg["input_height"])
                cs = [pi.box_to_center_scale(boxes[i], in_w, in_h, float(cfg.get("normalize_factor", 200.0)),
                                             float(cfg.get("padding_factor", 1.25)) * float(ps)) for i in chunk]
                c, s = np.stack([x[0] for x in cs]), np.stack([x[1] for x in cs])
                if centers is None:
                    centers, scales = c, s
                elif not (np.allclose(centers, c) and np.allclose(scales, s)):
                    raise ValueError("ensemble members crop different regions (aspect ratio / padding differ)")
                crops = torch.stack([warp_affine(img, pi.warp_matrix(ci, si, in_w, in_h), in_h, in_w) for ci, si in zip(c, s)])
                mean = torch.tensor(cfg["image_mean"], dtype=torch.float32, device=device)
                std = torch.tensor(cfg["image_std"], dtype=torch.float32, device=device)
                x = ((crops * float(cfg.get("rescale_factor", 1.0 / 255.0)) - mean) / std).permute(0, 3, 1, 2).contiguous()
                b = x.shape[0]
                if flip:
                    x = torch.cat([x, x.flip(-1)])
                with autocast:
                    hm = model(x).float()
                if flip:  # flip_back without the 1 px shift (UDP), then average the two views
                    hm = 0.5 * (hm[:b] + hm[b:].flip(-1)[:, perm])
                if hm.shape[-2:] != (out_h, out_w):
                    hm = F.interpolate(hm, size=(out_h, out_w), mode="bilinear", align_corners=True)
                norm_w = float(weight) / w_active_sum
                total = norm_w * hm if total is None else total + norm_w * hm
            c_ps, m_ps = decode_heatmaps_dark(total, centers, scales, kernel_size=dark_kernel,
                                              sigma=dark_sigma, clamp_shift=clamp_shift)
            coords = coords + wps * c_ps
            maxvals = maxvals + wps * m_ps.squeeze(-1)
        keypoints[chunk] = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
        joint_scores[chunk] = maxvals.cpu().numpy() if isinstance(maxvals, torch.Tensor) else maxvals
    return keypoints, joint_scores, usable

