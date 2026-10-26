"""Local stand-in for the Codabench ingestion program (Track 3).

Mirrors the documented contract: the working directory is the submission
folder, run.py is imported and predict_image(sample) is called once per image
with the sample dict from the Submission page. The return value may be a list
of 17x3 (or flat 51) keypoints per box, or {"keypoints": [...], "scores": [...]};
numpy arrays and tensors are accepted.

Runs inside the evaluation image, so it only uses the standard library + numpy.

Assumption (not documented by the organizers): when no "scores" are returned,
the person score is the mean of the third keypoint column.

Usage (inside the container):
  python ingestion.py --input /app/input_data --submission /app/submission \
      --output /app/output/predictions.json [--time-limit 7200]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

NUM_KEYPOINTS = 17


def to_list(value):
    if hasattr(value, "detach"):  # torch tensor, without importing torch here
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_list(v) for v in value]
    return value


def normalize_output(output, num_boxes: int, image_id) -> tuple[list[list[float]], list[float]]:
    scores = None
    if isinstance(output, dict):
        if "keypoints" not in output:
            raise ValueError(f"image {image_id}: dict output needs a 'keypoints' key")
        keypoints = to_list(output["keypoints"])
        if output.get("scores") is not None:
            scores = [float(s) for s in to_list(output["scores"])]
    else:
        keypoints = to_list(output)

    if len(keypoints) != num_boxes:
        raise ValueError(f"image {image_id}: got {len(keypoints)} predictions for {num_boxes} boxes")
    if scores is not None and len(scores) != num_boxes:
        raise ValueError(f"image {image_id}: got {len(scores)} scores for {num_boxes} boxes")

    flat_all = []
    for b, person in enumerate(keypoints):
        flat = np.asarray(person, dtype=np.float64).reshape(-1)
        if flat.size != NUM_KEYPOINTS * 3:
            raise ValueError(f"image {image_id}, box {b}: expected 17x3 or 51 values, got {flat.size}")
        if not np.all(np.isfinite(flat)):
            raise ValueError(f"image {image_id}, box {b}: non-finite keypoint values")
        flat_all.append(flat.tolist())
    if scores is None:
        scores = [float(np.mean(np.asarray(f).reshape(-1, 3)[:, 2])) for f in flat_all]
    for b, s in enumerate(scores):
        if not math.isfinite(s):
            raise ValueError(f"image {image_id}, box {b}: non-finite score")
    return flat_all, scores


def load_submission(submission_dir: str):
    # The real ingestion program only guarantees that the working directory is the
    # submission folder; do not put it on sys.path, so run.py must handle its own imports.
    os.chdir(submission_dir)
    spec = importlib.util.spec_from_file_location("run", os.path.join(submission_dir, "run.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["run"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "predict_image"):
        raise AttributeError("run.py does not define predict_image(sample)")
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-limit", type=float, default=7200.0)
    args = parser.parse_args()

    with open(os.path.join(args.input, "boxes.json"), "r", encoding="utf-8") as fh:
        boxes = json.load(fh)
    with open(os.path.join(args.input, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    print(f"[ingestion] set={task.get('name')} images={len(boxes['images'])} "
          f"boxes={len(boxes['annotations'])}", flush=True)

    by_image = {}
    for ann in boxes["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    start = time.perf_counter()
    module = load_submission(os.path.abspath(args.submission))
    if hasattr(module, "load_model"):
        module.load_model()
    load_time = time.perf_counter() - start
    print(f"[ingestion] submission loaded in {load_time:.1f}s", flush=True)

    predictions, timings = [], []
    images = boxes["images"]
    for n, image in enumerate(images, 1):
        anns = by_image.get(image["id"], [])
        sample = {
            "image_id": image["id"],
            "file_name": image["file_name"],
            "image_path": os.path.join(os.path.abspath(args.input), "images", image["file_name"]),
            "width": image["width"],
            "height": image["height"],
            "boxes": [list(a["bbox"]) for a in anns],
            "box_ids": [a["id"] for a in anns],
        }
        t0 = time.perf_counter()
        output = module.predict_image(sample)
        timings.append(time.perf_counter() - t0)
        keypoints, scores = normalize_output(output, len(anns), image["id"])
        for ann, kpts, score in zip(anns, keypoints, scores):
            predictions.append({"image_id": image["id"], "category_id": 1, "box_id": ann["id"],
                                "keypoints": kpts, "score": score})

        elapsed = time.perf_counter() - start
        if elapsed > args.time_limit:
            raise TimeoutError(f"wall-clock limit of {args.time_limit:.0f}s exceeded after {n} images")
        if n % 500 == 0 or n == len(images):
            print(f"[ingestion] {n}/{len(images)} images, {elapsed:.1f}s elapsed", flush=True)

    total = time.perf_counter() - start
    t = np.asarray(timings) if timings else np.zeros(1)
    report = {
        "set": task.get("name"),
        "images": len(images),
        "boxes": len(boxes["annotations"]),
        "load_seconds": load_time,
        "total_seconds": total,
        "per_image_ms": {"mean": float(t.mean() * 1e3), "p50": float(np.percentile(t, 50) * 1e3),
                         "p95": float(np.percentile(t, 95) * 1e3), "max": float(t.max() * 1e3)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.output)), "timing.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[ingestion] done: {len(predictions)} predictions in {total:.1f}s "
          f"(mean {report['per_image_ms']['mean']:.1f} ms/image)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
