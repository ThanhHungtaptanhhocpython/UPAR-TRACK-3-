"""Dependency-light COCO keypoint evaluation (OKS-AP), numpy only.

Reimplements pycocotools.cocoeval.COCOeval(iouType='keypoints') for a single
category, following the organizers' description: AP over OKS 0.50:0.95,
maxDets = 20, areas all / medium / large. Ground truth with no labelled
keypoints and crowd regions are ignored. Joints with v=0 in the ground truth
(e.g. the unannotated ear slots) do not enter the OKS.

The per-joint OKS sigmas used by the challenge are not published. The
default below is the COCO set, which is also what mmpose uses for
PoseTrack18 (head_bottom / head_top get the eye sigmas). Override with
--sigmas or a "sigmas" entry in the task file if the organizers publish theirs.

Usage:
  python -I coco_kpt_eval.py --gt ground_truth.json --dt predictions.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

DEFAULT_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])
IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 20
AREA_RNG = {"all": (0.0, 1e10), "medium": (32.0 ** 2, 96.0 ** 2), "large": (96.0 ** 2, 1e10)}


def _dt_area_from_keypoints(kpts: np.ndarray) -> float:
    # loadRes() derives the detection area from the keypoint extent.
    xs, ys = kpts[0::3], kpts[1::3]
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def compute_oks(gts: list[dict], dts: list[dict], sigmas: np.ndarray) -> np.ndarray:
    if not gts or not dts:
        return np.zeros((len(dts), len(gts)))
    variances = (sigmas * 2) ** 2
    ious = np.zeros((len(dts), len(gts)))
    for j, gt in enumerate(gts):
        g = np.asarray(gt["keypoints"], dtype=np.float64)
        xg, yg, vg = g[0::3], g[1::3], g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bx, by, bw, bh = gt["bbox"]
        x0, x1, y0, y1 = bx - bw, bx + bw * 2, by - bh, by + bh * 2
        for i, dt in enumerate(dts):
            d = np.asarray(dt["keypoints"], dtype=np.float64)
            xd, yd = d[0::3], d[1::3]
            if k1 > 0:
                dx, dy = xd - xg, yd - yg
            else:
                z = np.zeros(len(sigmas))
                dx = np.maximum(z, x0 - xd) + np.maximum(z, xd - x1)
                dy = np.maximum(z, y0 - yd) + np.maximum(z, yd - y1)
            e = (dx ** 2 + dy ** 2) / variances / (gt["area"] + np.spacing(1)) / 2
            if k1 > 0:
                e = e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious


def _evaluate_image(gts, dts, ious, area_rng):
    """Returns (dt_scores, dt_matched[T,D], dt_ignore[T,D], num_gt_not_ignored) or None."""
    if not gts and not dts:
        return None
    gt_ignore = np.array([1 if (g["_ignore"] or g["area"] < area_rng[0] or g["area"] > area_rng[1]) else 0
                          for g in gts], dtype=int)
    gt_order = np.argsort(gt_ignore, kind="mergesort")
    gts = [gts[i] for i in gt_order]
    gt_ignore = gt_ignore[gt_order]
    is_crowd = np.array([g.get("iscrowd", 0) for g in gts], dtype=int)

    dt_order = np.argsort([-d["score"] for d in dts], kind="mergesort")[:MAX_DETS]
    dts = [dts[i] for i in dt_order]
    ious = ious[dt_order][:, gt_order] if len(ious) else ious

    T, G, D = len(IOU_THRS), len(gts), len(dts)
    gt_matched = np.zeros((T, G))
    dt_matched = np.zeros((T, D))
    dt_ignore = np.zeros((T, D))
    if len(ious):
        for t_index, t in enumerate(IOU_THRS):
            for d_index in range(D):
                best = min(t, 1 - 1e-10)
                m = -1
                for g_index in range(G):
                    if gt_matched[t_index, g_index] > 0 and not is_crowd[g_index]:
                        continue
                    if m > -1 and gt_ignore[m] == 0 and gt_ignore[g_index] == 1:
                        break
                    if ious[d_index, g_index] < best:
                        continue
                    best = ious[d_index, g_index]
                    m = g_index
                if m == -1:
                    continue
                dt_ignore[t_index, d_index] = gt_ignore[m]
                dt_matched[t_index, d_index] = 1
                gt_matched[t_index, m] = 1
    dt_out_of_range = np.array([d["area"] < area_rng[0] or d["area"] > area_rng[1] for d in dts]).reshape(1, D)
    dt_ignore = np.logical_or(dt_ignore, np.logical_and(dt_matched == 0, np.repeat(dt_out_of_range, T, 0)))
    scores = np.array([d["score"] for d in dts])
    return scores, dt_matched, dt_ignore, int(np.count_nonzero(gt_ignore == 0))


def _accumulate(per_image):
    """AP and AR at each IoU threshold for one area range."""
    per_image = [r for r in per_image if r is not None]
    T = len(IOU_THRS)
    if not per_image:
        return np.full(T, -1.0), np.full(T, -1.0)
    scores = np.concatenate([r[0] for r in per_image])
    order = np.argsort(-scores, kind="mergesort")
    dtm = np.concatenate([r[1] for r in per_image], axis=1)[:, order]
    dtig = np.concatenate([r[2] for r in per_image], axis=1)[:, order]
    npig = sum(r[3] for r in per_image)
    if npig == 0:
        return np.full(T, -1.0), np.full(T, -1.0)

    tps = np.logical_and(dtm, np.logical_not(dtig))
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig))
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    ap = np.zeros(T)
    ar = np.zeros(T)
    for t in range(T):
        tp, fp = tp_sum[t], fp_sum[t]
        nd = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        ar[t] = rc[-1] if nd else 0.0
        pr = pr.tolist()
        for i in range(nd - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        inds = np.searchsorted(rc, REC_THRS, side="left")
        q = np.zeros(len(REC_THRS))
        for ri, pi in enumerate(inds):
            if pi >= nd:
                break
            q[ri] = pr[pi]
        ap[t] = q.mean()
    return ap, ar


def evaluate(gt: dict, predictions: list[dict], sigmas: np.ndarray = DEFAULT_SIGMAS) -> dict:
    """gt: COCO dict with keypoints. predictions: COCO results list."""
    gts_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        ann = dict(ann)
        num_kpts = ann.get("num_keypoints", int(np.count_nonzero(np.asarray(ann["keypoints"])[2::3] > 0)))
        ann["_ignore"] = bool(ann.get("iscrowd", 0)) or num_kpts == 0
        gts_by_image[ann["image_id"]].append(ann)
    dts_by_image = defaultdict(list)
    for det in predictions:
        det = dict(det)
        det["area"] = _dt_area_from_keypoints(np.asarray(det["keypoints"], dtype=np.float64))
        dts_by_image[det["image_id"]].append(det)

    image_ids = sorted(img["id"] for img in gt["images"])
    ious = {i: compute_oks(gts_by_image[i], sorted(dts_by_image[i], key=lambda d: -d["score"])[:MAX_DETS], sigmas)
            for i in image_ids}
    # compute_oks above already sees score-sorted detections; keep the same order for matching.
    for i in image_ids:
        dts_by_image[i] = sorted(dts_by_image[i], key=lambda d: -d["score"])[:MAX_DETS]

    results = {}
    for name, rng in AREA_RNG.items():
        per_image = [_evaluate_image(gts_by_image[i], dts_by_image[i], ious[i], rng) for i in image_ids]
        ap, ar = _accumulate(per_image)
        results[name] = (ap, ar)

    def mean_valid(x):
        x = x[x > -1]
        return float(x.mean()) if len(x) else -1.0

    ap_all, ar_all = results["all"]
    return {
        "AP": mean_valid(ap_all),
        "AP50": float(ap_all[0]),
        "AP75": float(ap_all[5]),
        "AP_medium": mean_valid(results["medium"][0]),
        "AP_large": mean_valid(results["large"][0]),
        "AR": mean_valid(ar_all),
        "AR50": float(ar_all[0]),
        "AR75": float(ar_all[5]),
        "AR_medium": mean_valid(results["medium"][1]),
        "AR_large": mean_valid(results["large"][1]),
    }


def per_keypoint_similarity(gt: dict, predictions: list[dict], sigmas: np.ndarray = DEFAULT_SIGMAS) -> dict:
    """Diagnostic: mean per-joint OKS term exp(-e) and PCK@0.5 over labelled joints,
    pairing each prediction with the ground truth box it was made for (box_id)."""
    names = gt["categories"][0]["keypoints"]
    variances = (sigmas * 2) ** 2
    anns = {a["id"]: a for a in gt["annotations"]}
    sums, hits, counts = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
    for det in predictions:
        ann = anns.get(det.get("box_id"))
        if ann is None:
            continue
        g = np.asarray(ann["keypoints"], dtype=np.float64)
        d = np.asarray(det["keypoints"], dtype=np.float64)
        vis = g[2::3] > 0
        e = ((d[0::3] - g[0::3]) ** 2 + (d[1::3] - g[1::3]) ** 2) / variances / (ann["area"] + np.spacing(1)) / 2
        ks = np.exp(-e)
        sums[vis] += ks[vis]
        hits[vis] += ks[vis] > 0.5
        counts[vis] += 1
    out = {}
    for j, name in enumerate(names):
        if counts[j]:
            out[name] = {"oks": float(sums[j] / counts[j]), "pck50": float(hits[j] / counts[j]), "n": int(counts[j])}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--dt", required=True)
    parser.add_argument("--sigmas", type=float, nargs=17, default=None)
    parser.add_argument("--per-keypoint", action="store_true")
    args = parser.parse_args()

    with open(args.gt, "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    with open(args.dt, "r", encoding="utf-8") as fh:
        dt = json.load(fh)
    sigmas = np.asarray(args.sigmas) if args.sigmas else DEFAULT_SIGMAS
    metrics = evaluate(gt, dt, sigmas)
    print(json.dumps(metrics, indent=2))
    if args.per_keypoint:
        for name, row in per_keypoint_similarity(gt, dt, sigmas).items():
            print(f"  {name:15s} oks={row['oks']:.3f} pck50={row['pck50']:.3f} n={row['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
